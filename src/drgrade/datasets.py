"""
Dataset loaders. Each returns a DataFrame with the same columns:

    image_id | grade | path | dataset

so the rest of the pipeline does not care which dataset an image came from.
"""

from pathlib import Path

import pandas as pd

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    matches = sorted(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def _finish(df, dataset):
    missing = df["path"].isna()
    if missing.any():
        raise FileNotFoundError(
            f"{missing.sum()} labelled images not found, e.g. {df.loc[missing, 'image_id'].head(3).tolist()}"
        )
    df["path"] = df["path"].astype(str)
    df["grade"] = df["grade"].astype(int)
    df["dataset"] = dataset
    if df["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image ids in labels: {df.loc[df['image_id'].duplicated(), 'image_id'].head(3).tolist()}")
    bad = ~df["grade"].between(0, 4)
    if bad.any():
        raise ValueError(f"Grades outside 0-4: {df.loc[bad, 'grade'].unique()}")
    return df[["image_id", "grade", "path", "dataset"]].reset_index(drop=True)


def load_aptos(root, labels_csv=None, images_dir=None, split=None):
    """
    APTOS 2019 (id_code, diagnosis).

    Two layouts are supported:
      * Kaggle competition: root/train.csv + root/train_images/   (split=None:
        one labelled pool, split later by make_splits.py)
      * Pre-split versions with labelled train / val / test CSVs: call once per
        split with labels_csv=<that split's CSV>, images_dir=<its image folder>
        and split="train" | "val" | "test". The split is kept in a "split" column.
    A folder nested inside itself (train_images/train_images) is handled.
    """
    root = Path(root)
    labels_csv = Path(labels_csv) if labels_csv else root / "train.csv"
    images_dir = Path(images_dir) if images_dir else root / "train_images"
    if not labels_csv.is_absolute() and not labels_csv.exists():
        labels_csv = root / labels_csv
    if not images_dir.is_absolute() and not images_dir.exists():
        images_dir = root / images_dir
    nested = images_dir / images_dir.name
    if nested.is_dir():
        images_dir = nested
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")

    labels = pd.read_csv(labels_csv)
    labels.columns = labels.columns.str.strip()
    if not {"id_code", "diagnosis"} <= set(labels.columns):
        raise ValueError(f"{labels_csv} needs id_code and diagnosis columns, has {list(labels.columns)}")
    df = labels.rename(columns={"id_code": "image_id", "diagnosis": "grade"})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["path"] = [_find_image(images_dir, s) for s in df["image_id"]]
    df = _finish(df, "aptos")
    if split is not None:
        df["split"] = split
    return df


def load_idrid(root, split="test"):
    """
    IDRiD B. Disease Grading.  root = ".../IDRiD/B. Disease Grading"
    split: "train" or "test" (official split).
    """
    root = Path(root)
    sub = {"train": ("a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"),
           "test": ("b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv")}[split]
    images_dir = root / "1. Original Images" / sub[0]
    labels = pd.read_csv(root / "2. Groundtruths" / sub[1])
    labels.columns = labels.columns.str.strip()
    df = labels[["Image name", "Retinopathy grade"]].dropna()
    df = df.rename(columns={"Image name": "image_id", "Retinopathy grade": "grade"})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["path"] = [_find_image(images_dir, s) for s in df["image_id"]]
    return _finish(df, f"idrid_{split}")


LOADERS = {"aptos": load_aptos, "idrid": load_idrid}
