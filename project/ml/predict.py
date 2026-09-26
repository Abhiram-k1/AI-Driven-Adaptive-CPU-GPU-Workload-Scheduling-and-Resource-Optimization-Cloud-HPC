"""
ml/predict.py
=============
Inference module for pre-execution preferred-device prediction.
Uses trained Random Forest models (Model A or Model B).
Strictly pre-execution: accepts only pre-execution attributes.
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_schema import extract_pre_execution_features, MODEL_A_FEATURES, MODEL_B_FEATURES

from ml.train_classifier import DeviceClassifierPipeline

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "results" / "models"


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "DeviceClassifierPipeline":
            return DeviceClassifierPipeline
        return super().find_class(module, name)


def load_model(model_type: str = "model_a", models_dir: Path = DEFAULT_MODELS_DIR):
    """Loads a serialized classifier pipeline."""
    model_file = models_dir / f"{model_type.lower()}.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file '{model_file}' not found. Run train_classifier.py first.")
    with open(model_file, "rb") as f:
        return _ModelUnpickler(f).load()


def predict_device(
    workload: str,
    input_file: str,
    input_size_mib: float,
    model_type: str = "model_a",
    cpu_cores: int = 4,
    gpu_type: str = "Tesla_T4",
    gpu_memory_gb: float = 16.0,
    models_dir: Path = DEFAULT_MODELS_DIR
) -> Dict[str, Any]:
    """
    Predicts the preferred device (CPU vs GPU) using pre-execution features only.
    """
    features = extract_pre_execution_features(
        workload=workload,
        input_file=input_file,
        input_size_mib=input_size_mib,
        cpu_cores=cpu_cores,
        gpu_type=gpu_type,
        gpu_memory_gb=gpu_memory_gb
    )

    pipeline = load_model(model_type, models_dir)
    df_feat = pd.DataFrame([features])

    pred_class_idx = pipeline.predict(df_feat)[0]
    prob_dist = pipeline.predict_proba(df_feat)[0]

    predicted_device = "gpu" if pred_class_idx == 1 else "cpu"
    confidence = float(prob_dist[pred_class_idx])

    return {
        "workload": workload,
        "input_file": input_file,
        "input_size_mib": input_size_mib,
        "model_used": model_type.upper(),
        "predicted_device": predicted_device,
        "confidence": confidence,
        "probabilities": {
            "cpu": float(prob_dist[0]),
            "gpu": float(prob_dist[1])
        },
        "pre_execution_features": features
    }


if __name__ == "__main__":
    res = predict_device("cfd", "fvcorr.domn.193K", 43.54428, model_type="model_a")
    print("\nInference Test Output (Model A):")
    for k, v in res.items():
        if k != "pre_execution_features":
            print(f"  {k:20s}: {v}")
