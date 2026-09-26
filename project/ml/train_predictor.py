"""
train_predictor.py
==================
Trains two ML models on the feature dataset:

1. CLASSIFIER  — Random Forest + XGBoost ensemble
                 Target: best_device_binary (0=cpu, 1=gpu)

2. REGRESSORS  — Separate models for cpu_mean and gpu_mean runtime prediction
                 Used by scheduler to estimate task duration before running

Saves models to ml/models/ as .pkl files.

Usage:
    python ml/train_predictor.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

FEAT_CSV   = Path(__file__).parent.parent / "data" / "datasets" / "features.csv"
MODELS_DIR = Path(__file__).parent / "models"

# Features used for the classifier
CLASSIFIER_FEATURES = [
    "input_size_mb",
    "compute_intensity",
    "parallelism_degree",
    "data_dep",
    "transfer_bytes_mb",
    "compute_transfer_ratio",
    "transfer_overhead_ratio",
    "log_input_mb",
]

# Features for runtime regressors
REGRESSOR_FEATURES = [
    "input_size_mb",
    "compute_intensity",
    "parallelism_degree",
    "data_dep",
    "transfer_bytes_mb",
]


def load_data(path: Path = FEAT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} samples from {path}")
    return df


def train_classifier(df: pd.DataFrame) -> dict:
    """Train and evaluate device assignment classifier."""
    X = df[CLASSIFIER_FEATURES].values
    y = df["best_device_binary"].values   # 1 = GPU, 0 = CPU

    # Individual models
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0
    )

    # Ensemble via soft voting
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_clf)],
        voting="soft"
    )

    # Cross-validation (leave-one-out style for small datasets)
    cv = StratifiedKFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy")
    print(f"\n[Classifier] CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Final fit on all data
    ensemble.fit(X, y)

    # Feature importance from RF component
    ensemble.estimators_[0].fit(X, y)   # ensure RF is fitted standalone
    fi = pd.Series(
        ensemble.estimators_[0].feature_importances_,
        index=CLASSIFIER_FEATURES
    ).sort_values(ascending=False)
    print("\n[Classifier] Feature importances (RF):")
    print(fi.to_string())

    return {"model": ensemble, "features": CLASSIFIER_FEATURES, "cv_scores": cv_scores}


def train_regressor(df: pd.DataFrame, target_col: str) -> dict:
    """Train a runtime regressor for cpu_mean or gpu_mean."""
    X = df[REGRESSOR_FEATURES].values
    y = df[target_col].values

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    xgb_reg = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=0
    )

    cv = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)

    for name, model in [("RF", rf_reg), ("XGB", xgb_reg)]:
        mae_scores  = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        r2_scores   = cross_val_score(model, X, y, cv=cv, scoring="r2")
        print(f"  [{target_col}] {name}: MAE={mae_scores.mean():.4f}s  R²={r2_scores.mean():.3f}")

    # Use RF as primary regressor (more stable on small datasets)
    rf_reg.fit(X, y)
    return {"model": rf_reg, "features": REGRESSOR_FEATURES}


def save_model(obj: dict, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def main():
    df = load_data()

    if len(df) < 4:
        print("\nWARNING: Very few samples. Run more workloads / profiling runs first.")
        print("Training on synthetic augmentation for development purposes...")
        df = augment_for_dev(df)

    print("\n--- Training Device Classifier ---")
    clf = train_classifier(df)
    save_model(clf, "device_classifier")

    print("\n--- Training CPU Runtime Regressor ---")
    cpu_reg = train_regressor(df, "cpu_mean")
    save_model(cpu_reg, "cpu_runtime_regressor")

    print("\n--- Training GPU Runtime Regressor ---")
    gpu_reg = train_regressor(df, "gpu_mean")
    save_model(gpu_reg, "gpu_runtime_regressor")

    print("\nAll models trained and saved.")


def augment_for_dev(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """
    Synthetic augmentation for development when few real samples exist.
    Adds Gaussian noise to numeric columns to expand the dataset.
    """
    rows = []
    numeric = [c for c in CLASSIFIER_FEATURES + ["cpu_mean", "gpu_mean"] if c in df.columns]
    for _ in range(n):
        base = df.sample(1, replace=True).copy()
        for col in numeric:
            if col in base.columns:
                base[col] = base[col] * np.random.uniform(0.7, 1.3)
        rows.append(base)
    augmented = pd.concat([df] + rows, ignore_index=True)
    augmented["best_device_binary"] = (augmented["speedup_ratio"] >= 1.1).astype(int)
    return augmented


if __name__ == "__main__":
    main()
