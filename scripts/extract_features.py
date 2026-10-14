"""
Parallel feature extraction (one process per core).

Examples (run from the repo root):
  python scripts/extract_features.py --dataset aptos --root data/raw/aptos \
      --out data/processed/aptos_features_v2.csv --workers 4

  python scripts/extract_features.py --dataset idrid --split test \
      --root "data/raw/IDRiD/B. Disease Grading" --out data/processed/idrid_test_features_v2.csv

  # APTOS with labelled train / val / test CSVs (one run per split)
  python scripts/extract_features.py --dataset aptos --root data/raw/aptos \
      --labels train_1.csv --images-dir train_images --split-name train \
      --out data/processed/aptos_train_features_v2.csv

  # quick check on 50 images
  python scripts/extract_features.py --dataset aptos --root data/raw/aptos --limit 50 --out /tmp/x.csv
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from src.drgrade.config import DEFAULT                        # noqa: E402
from src.drgrade.datasets import LOADERS                      # noqa: E402
from src.drgrade.features import FEATURE_NAMES, extract_features  # noqa: E402


def _init_worker():
    cv2.setNumThreads(1)          # parallelism comes from processes; avoid oversubscription


def _work(row):
    timings = {}
    t = time.perf_counter()
    try:
        feats, quality = extract_features(row["path"], DEFAULT, timings)
    except Exception as e:                                    # keep going, report at the end
        return {"image_id": row["image_id"], "error": repr(e)}, None
    timings["total"] = (time.perf_counter() - t) * 1000
    out = {"image_id": row["image_id"], "dataset": row["dataset"], "grade": row["grade"],
           **({"split": row["split"]} if "split" in row else {}), **quality, **feats}
    return out, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test", help="IDRiD only: train or test")
    ap.add_argument("--labels", help="APTOS only: labels CSV (path, or file name inside --root)")
    ap.add_argument("--images-dir", help="APTOS only: image folder (path, or folder name inside --root)")
    ap.add_argument("--split-name", choices=["train", "val", "test"],
                    help="APTOS only: which provided split this CSV is; stored in a 'split' column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, help="process only the first N images (testing)")
    args = ap.parse_args()

    if args.dataset == "aptos":
        df = LOADERS["aptos"](args.root, args.labels, args.images_dir, args.split_name)
    else:
        df = LOADERS["idrid"](args.root, args.split)
    if args.limit:
        df = df.groupby("grade", group_keys=False).head(max(1, args.limit // 5)).head(args.limit)

    print(f"{len(df)} images | grades: {df['grade'].value_counts().sort_index().to_dict()} "
          f"| workers: {args.workers}")

    rows, timings, failed = [], [], []
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for k, (out, tm) in enumerate(pool.imap_unordered(_work, df.to_dict("records"), chunksize=4), 1):
            if tm is None:
                failed.append(out)
            else:
                rows.append(out)
                timings.append(tm)
            if k % 100 == 0 or k == len(df):
                rate = k / (time.perf_counter() - t0)
                print(f"  {k}/{len(df)}  ({rate:.1f} img/s)", flush=True)
    wall = time.perf_counter() - t0

    if not rows:
        print("All images failed. First errors:")
        for f in failed[:5]:
            print(" ", f["image_id"], f["error"])
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values("image_id")
    q_cols = sorted(c for c in result.columns if c.startswith("q_"))
    id_cols = ["image_id", "dataset", "grade"] + (["split"] if "split" in result.columns else [])
    result = result[id_cols + q_cols + FEATURE_NAMES]
    result.to_csv(out_path, index=False)

    t = pd.DataFrame(timings)
    t.describe(percentiles=[0.5, 0.95]).T[["mean", "50%", "95%"]].round(2).to_csv(
        out_path.with_name(out_path.stem + "_timing.csv"))
    print(f"\nSaved {len(result)} rows -> {out_path}")
    print(f"Wall time {wall:.1f} s | throughput {len(df) / wall:.2f} img/s with {args.workers} workers")
    print("Per-image stage time (ms, median):")
    print(t.median().round(1).sort_values(ascending=False).to_string())

    if failed:
        fpath = out_path.with_name(out_path.stem + "_failed.csv")
        pd.DataFrame(failed).to_csv(fpath, index=False)
        print(f"\n{len(failed)} images FAILED -> {fpath}")
    if not np.isfinite(result[FEATURE_NAMES].to_numpy(float)).all():
        print("WARNING: non-finite feature values present")


if __name__ == "__main__":
    main()
