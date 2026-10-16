"""
Read-only duplicate audit: verifies hash matches against the actual image files.

Nothing in data/ is modified. All outputs go to --out-dir (default results/dup_audit).

    python scripts/audit_duplicates.py \
        --features data/processed/aptos_train_features_v2.csv \
                   data/processed/aptos_val_features_v2.csv \
                   data/processed/aptos_test_features_v2.csv \
        --image-dir train=data/raw/aptos/train_images \
        --image-dir val=data/raw/aptos/val_images \
        --image-dir test=data/raw/aptos/test_images

Stages
  A  Hash sanity (CSV only): format, multiplicity, popcounts, nearest-neighbour histogram,
     exact-hash sharing between splits.
  B  Candidate pairs: every pair with hash distance <= --candidate-bits (default 40, wider
     than make_splits' 24, because re-encoded / resized copies can exceed 24 bits).
  C  Image verification of each pair, independent of the z-map hash:
       exact_file    identical file bytes (SHA-256)
       exact_pixels  identical decoded pixels (SHA-256 of the pixel array)
       ORB keypoints + RANSAC similarity transform on the retina crops -> inlier count,
       then high-pass correlation of the aligned images (ncc_aligned).
  D  Calibration: negative controls (random pairs with hash distance > 60) and positive
     controls (in-memory resized + JPEG-recompressed + brightness-shifted copies).
     Verdict thresholds come from these, and are printed.
  E  Verified groups (union-find over verified pairs only), per-image annotation file,
     side-by-side contact sheets for visual review.
"""

import argparse
import hashlib
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2                                                                  # noqa: E402
import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402

from src.drgrade.config import DEFAULT                                      # noqa: E402
from src.drgrade.datasets import _find_image                                # noqa: E402
from src.drgrade.dedup import hashes_to_words, hamming_matrix               # noqa: E402
from src.drgrade.preprocess import crop_to_retina, decode, preprocess       # noqa: E402

VERIFY_SIZE = 512
VERIFY_CFG = replace(DEFAULT, work_size=VERIFY_SIZE)
VERIFIED = {"exact_file", "exact_pixels", "near_duplicate"}


# ------------------------------------------------------------------ per-image work
def _signature(rgb):
    """Retina crop -> ORB keypoints/descriptors (512 px) + green channel and FOV mask (256 px)."""
    crop, _ = crop_to_retina(rgb, VERIFY_CFG)
    green = crop[:, :, 1]
    fov = (crop.max(axis=2) > 10).astype(np.uint8)
    fov = cv2.erode(fov, np.ones((9, 9), np.uint8))
    eq = cv2.createCLAHE(2.0, (8, 8)).apply(green)
    orb = cv2.ORB_create(nfeatures=1000, fastThreshold=10)
    kps, desc = orb.detectAndCompute(eq, fov)
    pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
    half = (VERIFY_SIZE // 2, VERIFY_SIZE // 2)
    return {"g": cv2.resize(green, half, interpolation=cv2.INTER_AREA),
            "fov": np.packbits(cv2.resize(fov, half, interpolation=cv2.INTER_NEAREST).astype(bool)),
            "pts": pts, "desc": desc if desc is not None else np.zeros((0, 32), np.uint8)}


def _highpass(g):
    g = g.astype(np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 4)            # remove illumination, keep structure


def _to_rgb8(pixels):
    """Decoded array (any channel count / bit depth) -> 8-bit RGB."""
    if pixels.dtype == np.uint16:
        pixels = (pixels >> 8).astype(np.uint8)
    if pixels.ndim == 2:
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
    if pixels.shape[2] == 4:
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)


def _image_info(item):
    """One decode per image: file checksum, pixel checksum, verification signature."""
    key, path = item
    try:
        data = Path(path).read_bytes()
        pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise IOError("could not decode")
        info = {"file_sha256": hashlib.sha256(data).hexdigest(),
                "pixel_sha256": hashlib.sha256(
                    str(pixels.shape).encode() + str(pixels.dtype).encode() + pixels.tobytes()).hexdigest(),
                "width": pixels.shape[1], "height": pixels.shape[0], "file_bytes": len(data)}
        info.update(_signature(_to_rgb8(pixels)))
        return key, info, None
    except Exception as e:                                      # report, never crash the audit
        return key, None, repr(e)


def _perturbed_info(item):
    """Positive control: a modified copy of an image, built in memory only."""
    key, path, seed = item
    rng = np.random.default_rng(seed)
    rgb, _, _ = decode(path, DEFAULT)
    s = rng.uniform(0.55, 0.85)
    img = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.15), 0, 255).astype(np.uint8)
    dy, dx = rng.integers(0, 15, 2)
    img = img[dy:, dx:]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 92))])
    img = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return key, _signature(img)


def compare(a, b):
    """ORB + RANSAC similarity transform, then correlation of aligned high-pass images."""
    out = {"orb_good": 0, "orb_inliers": 0, "inlier_ratio": 0.0, "ncc_aligned": np.nan,
           "scale": np.nan, "rotation_deg": np.nan}
    if len(a["desc"]) < 8 or len(b["desc"]) < 8:
        return out
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a["desc"], b["desc"], k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    out["orb_good"] = len(good)
    if len(good) < 6:
        return out
    src = a["pts"][[m.queryIdx for m in good]]
    dst = b["pts"][[m.trainIdx for m in good]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=4.0, maxIters=4000)
    if M is None or inl is None:
        return out
    n_in = int(inl.sum())
    out.update(orb_inliers=n_in, inlier_ratio=n_in / len(good),
               scale=float(np.hypot(M[0, 0], M[1, 0])),
               rotation_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
    # warp a into b's frame (transform estimated at 512 px, applied at 256 px);
    # correlate high-pass images where both retinas overlap
    M2 = M.copy()
    M2[:, 2] /= 2.0
    half = VERIFY_SIZE // 2
    hp_a = cv2.warpAffine(_highpass(a["g"]), M2, (half, half), flags=cv2.INTER_LINEAR)
    unpack = lambda f: np.unpackbits(f)[:half * half].reshape(half, half)
    fov_a = cv2.warpAffine(unpack(a["fov"]), M2, (half, half), flags=cv2.INTER_NEAREST) > 0
    both = fov_a & (unpack(b["fov"]) > 0)
    if both.sum() > 0.2 * half ** 2:
        x, y = hp_a[both], _highpass(b["g"])[both]
        x, y = x - x.mean(), y - y.mean()
        den = np.sqrt((x * x).sum() * (y * y).sum())
        out["ncc_aligned"] = float((x * y).sum() / den) if den > 0 else np.nan
    return out


# ------------------------------------------------------------------ helpers
def parse_image_dirs(items):
    dirs = {}
    for it in items:
        split, _, path = it.partition("=")
        p = Path(path)
        nested = p / p.name
        dirs[split] = nested if nested.is_dir() else p
        if not dirs[split].is_dir():
            sys.exit(f"Image folder not found for split '{split}': {path}")
    return dirs


def union_groups(n, pairs):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def contact_sheet(row, df, out_path):
    tiles = []
    for side in ("a", "b"):
        r = df.loc[row[f"idx_{side}"]]
        rgb, _, _ = decode(r["path"], DEFAULT)
        crop, _ = crop_to_retina(rgb, replace(DEFAULT, work_size=420))
        bar = np.zeros((54, 420, 3), np.uint8)
        cv2.putText(bar, f"{r['image_id']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bar, f"split={r['split']}  grade={r['grade']}  {int(r['q_orig_w'])}x{int(r['q_orig_h'])}",
                    (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, crop]))
    footer = np.zeros((30, 840, 3), np.uint8)
    cv2.putText(footer, f"hash d={row['hash_dist']}  verdict={row['verdict']}  inliers={row['orb_inliers']}  "
                        f"ncc={row['ncc_aligned']:.2f}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    sheet = np.vstack([np.hstack(tiles), footer])
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--image-dir", action="append", required=True, metavar="SPLIT=DIR")
    ap.add_argument("--out-dir", default="results/dup_audit")
    ap.add_argument("--candidate-bits", type=int, default=40)
    ap.add_argument("--nearest-hash-k", type=int, default=1,
                    help="also verify each image's k nearest hash neighbours, whatever the distance")
    ap.add_argument("--nearest-feature-k", type=int, default=1,
                    help="also verify each image's k nearest neighbours in standardized feature space")
    ap.add_argument("--n-neg-controls", type=int, default=300)
    ap.add_argument("--n-pos-controls", type=int, default=40)
    ap.add_argument("--recheck", type=int, default=30, help="recompute the hash from the file for N images")
    ap.add_argument("--max-sheets", type=int, default=250)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    for f in args.features:
        if Path(f).resolve().parent == out_dir:
            sys.exit("--out-dir must not be the folder holding the input feature CSVs (read-only audit)")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sheets").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.concat([pd.read_csv(f, dtype={"q_dhash": str}) for f in args.features], ignore_index=True)
    dirs = parse_image_dirs(args.image_dir)
    df["path"] = [_find_image(dirs[s], i) if s in dirs else None for s, i in zip(df["split"], df["image_id"])]
    missing = df["path"].isna()
    if missing.any():
        sys.exit(f"{missing.sum()} images not found, e.g. {df.loc[missing, ['split', 'image_id']].head(3).values.tolist()}")
    df["path"] = df["path"].astype(str)
    n = len(df)

    # ============================ A. hash sanity ============================
    print("=" * 70, "\nA. HASH SANITY (from CSVs only)\n" + "=" * 70)
    h = df["q_dhash"].fillna("")
    valid = h.str.fullmatch(r"[0-9a-f]{64}")
    print(f"images: {n} | valid 64-hex hashes: {valid.sum()} | invalid: {(~valid).sum()}")
    if (~valid).any():
        print("  examples:", h[~valid].head(3).tolist())
        sys.exit("Fix invalid hashes before auditing")
    words = hashes_to_words(list(h))
    pop = np.array([bin(int(x, 16)).count("1") for x in h])
    mult = h.map(h.value_counts())
    print(f"distinct hashes: {h.nunique()} | images sharing their exact hash with >=1 other: {(mult > 1).sum()}")
    print("hash multiplicity (images per hash value):",
          dict(sorted(h.value_counts().value_counts().to_dict().items())))
    print(f"popcount (bits set of 256): min {pop.min()} | p1 {np.percentile(pop, 1):.0f} | "
          f"median {np.median(pop):.0f} | max {pop.max()}")
    top = h.value_counts().head(5)
    print("most frequent hashes:")
    for val, c in top.items():
        print(f"  x{c}  popcount {bin(int(val, 16)).count('1'):3d}  {val[:24]}...")
    shared_pop = pop[(mult > 1).values]
    if len(shared_pop):
        print(f"popcount of exactly-shared hashes: min {shared_pop.min()} median {np.median(shared_pop):.0f} "
              "(a constant / degenerate hash would show very low or very high values)")

    # full distance matrix in chunks: nearest neighbour + candidate pairs
    nn = np.empty(n, int)
    cand = []
    knn_hash = []
    for s in range(0, n, 512):
        d = hamming_matrix(words, slice(s, s + 512))
        rows = np.arange(d.shape[0])
        d[rows, s + rows] = 10 ** 6
        nn[s:s + d.shape[0]] = d.min(axis=1)
        if args.nearest_hash_k > 0:
            nb = np.argsort(d, axis=1)[:, :args.nearest_hash_k]
            for r_, cols in enumerate(nb):
                knn_hash += [(s + r_, int(c), int(d[r_, c])) for c in cols]
        ii, jj = np.nonzero(d <= args.candidate_bits)
        ii = ii + s
        keep = ii < jj
        cand += list(zip(ii[keep], jj[keep], d[ii[keep] - s, jj[keep]]))
    bins = [0, 1, 5, 9, 17, 25, 33, 41, 49, 57, 65, 81, 257]
    hist = pd.cut(pd.Series(nn), bins=bins, right=False).value_counts().sort_index()
    print("\nnearest-neighbour distance histogram (bits):")
    for interval, c in hist.items():
        print(f"  [{interval.left:>3}, {interval.right:>3})  {c:5d}  {'#' * int(60 * c / n)}")
    pd.DataFrame({"nn_dist": nn}).value_counts().sort_index().to_csv(out_dir / "nn_hist.csv")

    print("\nimages sharing an EXACT hash, by split pair:")
    ex = df[mult > 1].groupby("q_dhash")["split"].apply(lambda s: tuple(sorted(set(s))))
    print("  " + ex.value_counts().to_string().replace("\n", "\n  "))

    # ============================ B/C. candidate verification ============================
    src = [pd.DataFrame(cand, columns=["idx_a", "idx_b", "hash_dist"]).assign(source="hash_threshold")]
    if knn_hash:
        src.append(pd.DataFrame(knn_hash, columns=["idx_a", "idx_b", "hash_dist"]).assign(source="nearest_hash"))
    if args.nearest_feature_k > 0:
        from src.drgrade.features import FEATURE_NAMES
        feats = df[[c for c in FEATURE_NAMES if c in df.columns]].to_numpy(float)
        feats = (feats - np.nanmean(feats, 0)) / (np.nanstd(feats, 0) + 1e-9)
        feats = np.nan_to_num(feats)
        fk = []
        for s0 in range(0, n, 512):
            blk = feats[s0:s0 + 512]
            d2 = (blk ** 2).sum(1)[:, None] + (feats ** 2).sum(1)[None, :] - 2 * blk @ feats.T
            d2[np.arange(len(blk)), s0 + np.arange(len(blk))] = np.inf
            for r_, cols in enumerate(np.argsort(d2, axis=1)[:, :args.nearest_feature_k]):
                fk += [(s0 + r_, int(c)) for c in cols]
        fk = pd.DataFrame(fk, columns=["idx_a", "idx_b"])
        from src.drgrade.dedup import _popcount64
        x = words[fk["idx_a"].to_numpy()] ^ words[fk["idx_b"].to_numpy()]
        fk["hash_dist"] = _popcount64(x).sum(axis=1).astype(int)
        src.append(fk.assign(source="nearest_feature"))
    allc = pd.concat(src, ignore_index=True)
    a_, b_ = allc[["idx_a", "idx_b"]].min(axis=1), allc[["idx_a", "idx_b"]].max(axis=1)
    allc["idx_a"], allc["idx_b"] = a_, b_
    pairs = (allc.groupby(["idx_a", "idx_b"], as_index=False)
                 .agg(hash_dist=("hash_dist", "min"), source=("source", lambda x: "+".join(sorted(set(x))))))
    print(f"\n{'=' * 70}\nB/C. IMAGE VERIFICATION of {len(pairs)} candidate pairs\n{'=' * 70}")
    print(pairs["source"].value_counts().to_string())

    neg = []
    tries = 0
    while len(neg) < args.n_neg_controls and tries < 50 * args.n_neg_controls:
        i, j = rng.integers(0, n, 2)
        tries += 1
        if i != j and int(hamming_matrix(words[[i, j]], slice(0, 1))[0, 1]) > 60:
            neg.append((min(i, j), max(i, j)))
    neg = pd.DataFrame(sorted(set(neg)), columns=["idx_a", "idx_b"])
    pos_idx = rng.choice(n, size=min(args.n_pos_controls, n), replace=False)

    needed = sorted(set(pairs["idx_a"]) | set(pairs["idx_b"]) | set(neg["idx_a"]) | set(neg["idx_b"]) | set(pos_idx))
    print(f"decoding {len(needed)} images ...", flush=True)
    info, errors = {}, []
    with Pool(args.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        for k, inf, err in pool.imap_unordered(_image_info, [(i, df.at[i, "path"]) for i in needed], chunksize=4):
            if err:
                errors.append((df.at[k, "image_id"], err))
            else:
                info[k] = inf
        pos_sig = dict(pool.map(_perturbed_info, [(int(i), df.at[i, "path"], int(i)) for i in pos_idx if i in info]))
    if errors:
        print(f"{len(errors)} images could not be read, e.g. {errors[:2]}")

    def run(frame):
        rows = []
        for r in frame.itertuples(index=False):
            a, b = info.get(r.idx_a), info.get(r.idx_b)
            if a is None or b is None:
                continue
            m = compare(a, b)
            m.update(exact_file=a["file_sha256"] == b["file_sha256"],
                     exact_pixels=a["pixel_sha256"] == b["pixel_sha256"])
            rows.append({**r._asdict(), **m})
        return pd.DataFrame(rows)

    negc = run(neg)
    posc = pd.DataFrame([{**compare(info[i], pos_sig[i]), "idx": i} for i in pos_sig])

    # ============================ D. calibration ============================
    inl_thr = max(25, int(np.ceil(np.nanpercentile(negc["orb_inliers"], 99.5))) + 5) if len(negc) else 25
    ncc_neg = np.nanpercentile(negc["ncc_aligned"].fillna(0), 99.5) if len(negc) else 0.3
    ncc_thr = float(min(0.8, max(0.5, ncc_neg + 0.2)))
    print(f"\n{'=' * 70}\nD. CALIBRATION\n{'=' * 70}")
    for name, c in [("negative controls (different images)", negc), ("positive controls (modified copies)", posc)]:
        if len(c):
            print(f"{name}: n={len(c)} | inliers p50 {c.orb_inliers.median():.0f}, p99.5 "
                  f"{np.percentile(c.orb_inliers, 99.5):.0f}, max {c.orb_inliers.max()} | ncc_aligned p50 "
                  f"{c.ncc_aligned.median():.2f}, p99.5 {np.nanpercentile(c.ncc_aligned.fillna(0), 99.5):.2f}")
    print(f"thresholds: near_duplicate = inliers >= {inl_thr} AND ncc_aligned >= {ncc_thr:.2f}")
    if len(posc):
        rec = ((posc.orb_inliers >= inl_thr) & (posc.ncc_aligned >= ncc_thr)).mean()
        print(f"verifier recall on positive controls: {rec:.1%}")
    if len(negc):
        fpr = ((negc.orb_inliers >= inl_thr) & (negc.ncc_aligned >= ncc_thr)).mean()
        print(f"verifier false-positive rate on negative controls: {fpr:.1%}")

    def verdict(r):
        if r.exact_file:
            return "exact_file"
        if r.exact_pixels:
            return "exact_pixels"
        if r.orb_inliers >= inl_thr and r.ncc_aligned >= ncc_thr:
            return "near_duplicate"
        if r.orb_inliers < max(10, inl_thr // 2) and not (r.ncc_aligned >= 0.5):
            return "different"
        return "needs_review"

    res = run(pairs)
    if len(res):
        res["verdict"] = res.apply(verdict, axis=1)
        for side in ("a", "b"):
            for col in ("image_id", "split", "grade"):
                res[f"{col}_{side}"] = df.loc[res[f"idx_{side}"], col].values
        res["cross_split"] = res["split_a"] != res["split_b"]
        res["label_conflict"] = res["grade_a"] != res["grade_b"]
        res["within_make_splits_threshold"] = res["hash_dist"] <= 24

    # ============================ E. results ============================
    print(f"\n{'=' * 70}\nE. RESULTS\n{'=' * 70}")
    if len(res):
        print("verdict by hash distance band:")
        band = pd.cut(res["hash_dist"], [-1, 0, 8, 16, 24, 32, 40, 64, 256],
                      labels=["0", "1-8", "9-16", "17-24", "25-32", "33-40", "41-64", "65+"])
        print(pd.crosstab(band, res["verdict"], margins=True).to_string())
        ver = res[res["verdict"].isin(VERIFIED)]
        print("\nverified duplicate pairs by split pair:")
        print(pd.crosstab((ver["split_a"] + "-" + ver["split_b"]).rename("split pair"),
                          ver["label_conflict"].map({True: "conflicting grade", False: "same grade"}),
                          margins=True).to_string())
        print("\nverified pairs by candidate source:")
        print(ver["source"].value_counts().to_string())
        print(f"\nverified pairs missed by make_splits (hash distance > 24): "
              f"{int((~ver['within_make_splits_threshold']).sum())}")
        print(f"make_splits pairs (<= 24 bits) NOT verified: "
              f"{int((res['within_make_splits_threshold'] & ~res['verdict'].isin(VERIFIED)).sum())}")
    else:
        ver = res

    groups = union_groups(n, ver[["idx_a", "idx_b"]].values if len(ver) else [])
    df["verified_dup_group"] = groups
    size = df.groupby("verified_dup_group")["image_id"].transform("size")
    g = df[size > 1].groupby("verified_dup_group").agg(
        n_images=("image_id", "size"), image_ids=("image_id", list),
        splits=("split", lambda s: sorted(set(s))), grades=("grade", lambda s: sorted(set(s))))
    if len(g):
        g["cross_split"] = g["splits"].apply(len) > 1
        g["label_conflict"] = g["grades"].apply(len) > 1
        print(f"\nverified groups: {len(g)} | cross-split: {int(g.cross_split.sum())} | "
              f"conflicting grades: {int(g.label_conflict.sum())} | images in groups: {int(g.n_images.sum())}")
    ann = df[["image_id", "split", "grade"]].copy()
    ann["verified_dup_group"] = np.where(size > 1, df["verified_dup_group"], -1)
    ann["group_cross_split"] = ann["verified_dup_group"].map(g["cross_split"]).fillna(False) if len(g) else False
    ann["group_label_conflict"] = ann["verified_dup_group"].map(g["label_conflict"]).fillna(False) if len(g) else False

    # hash determinism: recompute from the file for a sample of candidate images
    sample = list(rng.choice(needed, size=min(args.recheck, len(needed)), replace=False))
    same = sum(preprocess(df.at[i, "path"])["quality"]["q_dhash"] == df.at[i, "q_dhash"] for i in sample)
    print(f"\nhash recomputed from file matches CSV: {same}/{len(sample)}")

    # contact sheets: cross-split and conflicting first, then needs_review, then the rest
    if len(res):
        # sheets for every verified / needs_review pair (cross-split and conflicting first),
        # plus 10 random "different" pairs as a sanity check of the verifier
        keep = res[res["verdict"] != "different"]
        pri = (keep["cross_split"].astype(int) * 4 + keep["label_conflict"].astype(int) * 2
               + (keep["verdict"] == "needs_review").astype(int))
        order = keep.assign(_p=pri).sort_values(["_p", "hash_dist"], ascending=[False, True]).head(args.max_sheets)
        diff = res[res["verdict"] == "different"]
        order = pd.concat([order, diff.sample(min(10, len(diff)), random_state=args.seed)])
        for k, (_, r) in enumerate(order.iterrows()):
            contact_sheet(r, df, out_dir / "sheets" /
                          f"{k:03d}_{r['verdict']}_{r['split_a']}-{r['split_b']}_{r['image_id_a']}_{r['image_id_b']}.png")

    res.to_csv(out_dir / "pairs.csv", index=False)
    negc.to_csv(out_dir / "neg_controls.csv", index=False)
    posc.to_csv(out_dir / "pos_controls.csv", index=False)
    g.to_csv(out_dir / "verified_groups.csv")
    ann.to_csv(out_dir / "duplicate_annotations.csv", index=False)
    print(f"\nwrote {out_dir}/pairs.csv, verified_groups.csv, duplicate_annotations.csv, "
          f"nn_hist.csv, *_controls.csv, sheets/")
    print("No input files were modified.")


if __name__ == "__main__":
    main()
