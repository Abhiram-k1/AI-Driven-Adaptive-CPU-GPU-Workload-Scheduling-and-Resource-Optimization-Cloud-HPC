"""
ml/train_classifier.py
======================
Trains the Random Forest device prediction models.
Strictly implements:
  - Section 17: Random Forest Classifier (Primary Model)
  - Section 18: Model A vs Model B Ablation Study
      Model A: All pre-execution features + workload_type + workload_domain
      Model B: Intrinsic features only (excludes workload_type and workload_domain)
  - Section 19: Out-of-fold encoding (fit encoders strictly on training data).
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_schema import MODEL_A_FEATURES, MODEL_B_FEATURES, CATEGORICAL_FEATURES

DEFAULT_LAYER3_PATH = Path(__file__).resolve().parent.parent / "dataset" / "layer3_ml_dataset.csv"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "results" / "models"


class DeviceClassifierPipeline:
    """Encapsulates categorical preprocessing and Random Forest classifier."""

    def __init__(self, feature_names: List[str], random_state: int = 42):
        self.feature_names = list(feature_names)
        self.random_state = random_state
        self.cat_cols = [c for c in self.feature_names if c in CATEGORICAL_FEATURES]
        self.num_cols = [c for c in self.feature_names if c not in CATEGORICAL_FEATURES]
        self.encoder: Optional[OrdinalEncoder] = None
        self.classifier: Optional[RandomForestClassifier] = None
        self.classes_: np.ndarray = np.array([0, 1])  # 0=cpu, 1=gpu
        self.single_class_target: Optional[int] = None

    def fit(self, X_df: pd.DataFrame, y: np.ndarray):
        """Fits preprocessing and Random Forest classifier strictly on provided training data."""
        X_encoded = self._fit_transform_features(X_df)

        unique_y = np.unique(y)
        if len(unique_y) < 2:
            # When training fold has only one class, handle gracefully
            self.single_class_target = int(unique_y[0])
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            # Create a minimal fit
            self.classifier.fit(X_encoded, y)
        else:
            self.single_class_target = None
            self.classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state)
            self.classifier.fit(X_encoded, y)
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        """Generates deterministic class predictions."""
        if len(X_df) == 0:
            return np.array([])
        X_encoded = self._transform_features(X_df)
        if self.single_class_target is not None:
            return np.full(len(X_df), self.single_class_target, dtype=int)
        return self.classifier.predict(X_encoded)

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """Generates prediction probabilities for [CPU, GPU]."""
        if len(X_df) == 0:
            return np.empty((0, 2))
        X_encoded = self._transform_features(X_df)
        if self.single_class_target is not None:
            proba = np.zeros((len(X_df), 2))
            proba[:, self.single_class_target] = 1.0
            return proba
        # Ensure 2-class probabilities
        probs = self.classifier.predict_proba(X_encoded)
        if probs.shape[1] == 1:
            full_probs = np.zeros((len(X_df), 2))
            cls = self.classifier.classes_[0]
            full_probs[:, cls] = probs[:, 0]
            return full_probs
        return probs

    def get_feature_importances(self) -> pd.Series:
        """Returns feature importance series sorted descending."""
        if self.classifier is None or not hasattr(self.classifier, "feature_importances_"):
            return pd.Series(0.0, index=self.feature_names)
        importances = self.classifier.feature_importances_
        # Feature names match order: num_cols + cat_cols
        all_cols = self.num_cols + self.cat_cols
        return pd.Series(importances, index=all_cols).sort_values(ascending=False)

    def _fit_transform_features(self, df: pd.DataFrame) -> np.ndarray:
        X_num = df[self.num_cols].values if self.num_cols else np.empty((len(df), 0))
        if self.cat_cols:
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_cat = self.encoder.fit_transform(df[self.cat_cols].astype(str))
            return np.hstack([X_num, X_cat]) if self.num_cols else X_cat
        return X_num

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        X_num = df[self.num_cols].values if self.num_cols else np.empty((len(df), 0))
        if self.cat_cols:
            if self.encoder is None:
                raise RuntimeError("Encoder has not been fitted.")
            X_cat = self.encoder.transform(df[self.cat_cols].astype(str))
            return np.hstack([X_num, X_cat]) if self.num_cols else X_cat
        return X_num


def train_models(
    layer3_path: Path = DEFAULT_LAYER3_PATH,
    models_dir: Path = DEFAULT_MODELS_DIR
) -> Tuple[DeviceClassifierPipeline, DeviceClassifierPipeline]:
    """
    Trains Model A and Model B on Layer 3 ML dataset and saves them to results/models/.
    """
    if not layer3_path.exists():
        raise FileNotFoundError(f"Layer 3 dataset not found at {layer3_path}")

    df_ml = pd.read_csv(layer3_path)
    if len(df_ml) == 0:
        raise ValueError("Layer 3 dataset is empty! Cannot train models.")

    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST CLASSIFIERS (MODEL A & MODEL B)")
    print("=" * 80)

    y = (df_ml["preferred_device"] == "gpu").astype(int).values

    # Train Model A (Full Features)
    print(f"Fitting Model A ({len(MODEL_A_FEATURES)} features)...")
    model_a = DeviceClassifierPipeline(MODEL_A_FEATURES, random_state=42)
    model_a.fit(df_ml, y)

    # Train Model B (Ablated: excludes workload_type and workload_domain)
    print(f"Fitting Model B ({len(MODEL_B_FEATURES)} features)...")
    model_b = DeviceClassifierPipeline(MODEL_B_FEATURES, random_state=42)
    model_b.fit(df_ml, y)

    # Save models
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "model_a.pkl", "wb") as f:
        pickle.dump(model_a, f)
    with open(models_dir / "model_b.pkl", "wb") as f:
        pickle.dump(model_b, f)

    print(f"Models successfully serialized to: {models_dir}")
    print("\nModel A Feature Importances:")
    print(model_a.get_feature_importances())
    print("\nModel B Feature Importances:")
    print(model_b.get_feature_importances())

    return model_a, model_b


if __name__ == "__main__":
    train_models()
