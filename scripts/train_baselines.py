from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

BASE = Path("data/processed")
MODEL_DIR = Path("results/models")
METRIC_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": BASE / "aptos_train_features_v2.csv",
    "val": BASE / "aptos_val_features_v2.csv",
    "test": BASE / "aptos_test_features_v2.csv",
}

dfs = {name: pd.read_csv(path) for name, path in FILES.items()}

# Metadata and labels must never be model inputs.
excluded = {"image_id", "dataset", "grade", "split", "q_dhash"}

feature_cols = [
    c for c in dfs["train"].columns
    if c not in excluded
    and pd.api.types.is_numeric_dtype(dfs["train"][c])
]

if not feature_cols:
    raise ValueError("No numeric feature columns found.")

for name, df in dfs.items():
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing features: {sorted(missing)}")

    if df["grade"].isna().any():
        raise ValueError(f"{name} contains missing grades.")

    print(
        f"{name}: {len(df)} rows | "
        f"{df['grade'].nunique()} grades | "
        f"numeric features: {len(feature_cols)}"
    )

X_train = dfs["train"][feature_cols].replace([np.inf, -np.inf], np.nan)
y_train = dfs["train"]["grade"].astype(int)

X_val = dfs["val"][feature_cols].replace([np.inf, -np.inf], np.nan)
y_val = dfs["val"]["grade"].astype(int)

X_test = dfs["test"][feature_cols].replace([np.inf, -np.inf], np.nan)
y_test = dfs["test"]["grade"].astype(int)

# Fail early if training data contains a feature that is entirely missing.
all_missing = X_train.columns[X_train.isna().all()].tolist()
if all_missing:
    raise ValueError(f"Features entirely missing in training: {all_missing}")

models = {
    "logistic_regression": LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
    ),
    "linear_svm": LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000,
    ),
    "rbf_svm": SVC(
        kernel="rbf",
        class_weight="balanced",
        C=1.0,
        gamma="scale",
    ),
}

results = []
fitted = {}

for name, estimator in models.items():
    print(f"\n{'=' * 60}\nTraining: {name}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)
    fitted[name] = pipeline

    pred = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, pred)
    macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_val, pred, average="weighted", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_val, pred)

    results.append({
        "model": name,
        "val_accuracy": accuracy,
        "val_macro_f1": macro_f1,
        "val_weighted_f1": weighted_f1,
        "val_balanced_accuracy": balanced_acc,
    })

    print(f"Validation accuracy:          {accuracy:.4f}")
    print(f"Validation macro-F1:          {macro_f1:.4f}")
    print(f"Validation weighted-F1:       {weighted_f1:.4f}")
    print(f"Validation balanced accuracy: {balanced_acc:.4f}")

    print("\nValidation classification report:")
    print(classification_report(
        y_val, pred, labels=[0, 1, 2, 3, 4], zero_division=0
    ))

    print("Validation confusion matrix (true rows, predicted columns):")
    print(confusion_matrix(y_val, pred, labels=[0, 1, 2, 3, 4]))

    joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")

results_df = pd.DataFrame(results).sort_values(
    "val_macro_f1", ascending=False
)
results_df.to_csv(METRIC_DIR / "baseline_validation_results.csv", index=False)

# Choose the model using validation macro-F1 only.
best_name = results_df.iloc[0]["model"]
best_model = fitted[best_name]

print(f"\n{'=' * 60}")
print("Validation ranking:")
print(results_df.to_string(index=False))
print(f"\nSelected using validation macro-F1: {best_name}")

# This first test evaluation is provisional because duplicate review is ongoing.
test_pred = best_model.predict(X_test)

test_metrics = {
    "selected_model": best_name,
    "status": "provisional_pending_duplicate_audit",
    "test_accuracy": float(accuracy_score(y_test, test_pred)),
    "test_macro_f1": float(
        f1_score(y_test, test_pred, average="macro", zero_division=0)
    ),
    "test_weighted_f1": float(
        f1_score(y_test, test_pred, average="weighted", zero_division=0)
    ),
    "test_balanced_accuracy": float(
        balanced_accuracy_score(y_test, test_pred)
    ),
    "test_confusion_matrix": confusion_matrix(
        y_test, test_pred, labels=[0, 1, 2, 3, 4]
    ).tolist(),
}

with open(METRIC_DIR / "provisional_test_results.json", "w") as f:
    json.dump(test_metrics, f, indent=2)

print("\nProvisional test results (not final publication metrics):")
print(json.dumps(test_metrics, indent=2))

print("\nSaved models to:", MODEL_DIR)
print("Saved metrics to:", METRIC_DIR)
print("Feature columns:", feature_cols)
