"""
Duplicate detection, quality flags and train/test splits.

Two modes:

A. Provided splits (e.g. APTOS with labelled train / val / test CSVs).
   Pass one feature CSV per split; each already has a "split" column.
   The split is KEPT. Duplicates are detected across all splits, and any
   duplicate group that spans splits (train <-> val/test leakage) is reported.

     python scripts/make_splits.py \
         --features data/processed/aptos_train_features_v2.csv \
                    data/processed/aptos_val_features_v2.csv \
                    data/processed/aptos_test_features_v2.csv \
         --out data/processed/aptos_features_v2_split.csv --drop-leaks

B. One labelled pool (Kaggle train.csv only): an 80/20 holdout is created,
   stratified by grade, duplicate groups never split across train and test.

     python scripts/make_splits.py --features data/processed/aptos_features_v2.csv \
         --out data/processed/aptos_features_v2_split.csv

Adds columns:
  dup_group       near-duplicate images share a group id
  label_conflict  True if a duplicate group contains different grades
  leak            True if a duplicate group spans more than one split (mode A)
  q_flag          True if the image fails a basic quality check
  split           train / val / test
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold      # noqa: E402

from src.drgrade.dedup import (                               # noqa: E402
    duplicate_groups, duplicate_report, hashes_to_words, hamming_matrix,
)

SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def quality_flags(df):
    sharp_low = df["q_sharpness"].quantile(0.02)
    return (
        (df["q_fov_coverage"] < 0.5)            # retina mostly missing / bad crop
        | (df["q_dark_frac"] > 0.4)             # badly under-exposed
        | (df["q_saturated_frac"] > 0.2)        # badly over-exposed
        | (df["q_sharpness"] < sharp_low)       # blurriest 2%
        | (df["q_fov_diameter_px"] < 500)       # retina < 500 px in the file: upscaled > 1.5x
    )


def nearest_distances(hashes):
    """Distance from each image to its closest other image (helps choose --max-bits)."""
    words = hashes_to_words(list(hashes))
    out = []
    for s in range(0, len(words), 512):
        d = hamming_matrix(words, slice(s, s + 512))
        d[np.arange(d.shape[0]), np.arange(s, s + d.shape[0])] = 10 ** 6
        out.append(d.min(axis=1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, help="one or more feature CSVs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-bits", type=int, default=24,
                    help="hash bit distance (of 256) counted as near-duplicate")
    ap.add_argument("--drop-leaks", action="store_true",
                    help="mode A: drop val/test images that duplicate an image in an earlier split")
    ap.add_argument("--test-frac", type=float, default=0.2, help="mode B only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(f, dtype={"q_dhash": str}) for f in args.features],
                   ignore_index=True)
    if df["image_id"].duplicated().any():
        sys.exit(f"Same image_id in several inputs: {df.loc[df.image_id.duplicated(), 'image_id'].head(3).tolist()}")
    provided = "split" in df.columns and df["split"].notna().all()

    groups, pairs = duplicate_groups(df["q_dhash"], args.max_bits)
    df["dup_group"] = groups
    report = duplicate_report(df, groups)
    df["label_conflict"] = df["dup_group"].isin(set(report.index[report["label_conflict"]]))
    df["q_flag"] = quality_flags(df)

    nn = nearest_distances(df["q_dhash"])
    print(f"Images: {len(df)}")
    print("Nearest-neighbour hash distance (bits of 256): "
          + ", ".join(f"p{q}={np.percentile(nn, q):.0f}" for q in (1, 5, 25, 50)))
    print(f"Near-duplicate pairs: {len(pairs)} | groups with >1 image: {len(report)} "
          f"| groups with conflicting grades: {int(report['label_conflict'].sum())} "
          f"({int(df['label_conflict'].sum())} images)")

    if provided:
        # ---------- mode A: keep the provided split, check for leakage ----------
        bad = set(df["split"]) - set(SPLIT_ORDER)
        if bad:
            sys.exit(f"Unknown split names: {bad}")
        n_splits_per_group = df.groupby("dup_group")["split"].transform("nunique")
        df["leak"] = n_splits_per_group > 1
        leak_groups = df.loc[df["leak"], "dup_group"].unique()
        print(f"\nDuplicate groups spanning splits (leakage): {len(leak_groups)} "
              f"({int(df['leak'].sum())} images)")
        if len(leak_groups):
            print(pd.crosstab(df.loc[df.leak, "dup_group"], df.loc[df.leak, "split"]).head(10).to_string())
        if args.drop_leaks and len(leak_groups):
            # within each leaking group, keep the members of its earliest split only
            order = df["split"].map(SPLIT_ORDER)
            first = order.groupby(df["dup_group"]).transform("min")
            drop = df["leak"] & (order > first)
            print(f"Dropping {int(drop.sum())} val/test images that duplicate an earlier split")
            df = df[~drop].copy()
            df["leak"] = False
        report["splits"] = [sorted(set(df.loc[df.dup_group == g, "split"])) for g in report.index]
    else:
        # ---------- mode B: create a group-aware stratified holdout ----------
        n_splits = round(1 / args.test_frac)
        if df["dup_group"].nunique() < n_splits:
            sys.exit(f"Only {df['dup_group'].nunique()} distinct groups: --max-bits {args.max_bits} is too loose")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        _, test_idx = next(sgkf.split(df, df["grade"], groups=df["dup_group"]))
        df["split"] = "train"
        df.loc[df.index[test_idx], "split"] = "test"

    print(f"Quality-flagged images: {int(df['q_flag'].sum())}")
    print("\nGrade distribution per split:")
    cols = [c for c in SPLIT_ORDER if c in set(df["split"])]
    print(pd.crosstab(df["grade"], df["split"], margins=True)[cols + ["All"]].to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    report.to_csv(out.with_name(out.stem + "_duplicates.csv"))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
