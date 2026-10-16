from pathlib import Path
import random

import joblib
import pandas as pd

# Project paths
ROOT = Path.home() / "aiml_hpc"

MODEL_PATH = (
    ROOT / "results" / "models"
    / "partial_oversampling_random_forest.joblib"
)

TEST_CSV = (
    ROOT / "data" / "processed"
    / "aptos_test_features_v2.csv"
)

# Load saved model and processed test data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(TEST_CSV)

# Metadata columns are not model inputs
metadata = {
    "image_id",
    "dataset",
    "grade",
    "split",
    "q_dhash",
}

feature_cols = [c for c in df.columns if c not in metadata]

# Use the exact training feature names if available
if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)

# Check that all expected features exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing test features: {missing}")

# Select a random test row each run
idx = random.randrange(len(df))
sample = df.iloc[[idx]]

X_sample = sample[feature_cols]
original_class = int(sample["grade"].iloc[0])

# Predict using saved weights
predicted_label = int(model.predict(X_sample)[0])

# 1 = correct, 0 = incorrect
correct = int(predicted_label == original_class)

print("=" * 40)
print("DRGrade — Random Test Prediction")
print("=" * 40)
print("Image ID       :", sample["image_id"].iloc[0])
print("Original class :", original_class)
print("Predicted label:", predicted_label)
print("Correct        :", correct)
print("=" * 40)
