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


class WorkloadPredictor:
    """
    Object-oriented predictor interface for WorkloadDAG and AI Schedulers.
    Uses trained Random Forest models (Model A or Model B).
    Pre-execution only: takes pre-execution features and provides preferred device,
    confidence probability, and estimated execution metrics.
    """

    def __init__(
        self,
        model_type: str = "model_a",
        models_dir: Path = DEFAULT_MODELS_DIR
    ):
        self.model_type = model_type.lower()
        self.models_dir = Path(models_dir)
        self.pipeline = load_model(self.model_type, self.models_dir)

        # Reference timings (seconds) derived from Layer 2 dataset benchmarks on Tesla T4 + 4 CPU cores
        self.reference_timings = {
            "bfs": {"cpu_s": 0.036, "gpu_s": 0.019, "speedup": 1.93, "transfer_ratio": 0.50},
            "cfd": {"cpu_s": 77.537, "gpu_s": 1.125, "speedup": 68.92, "transfer_ratio": 0.05},
            "hotspot": {"cpu_s": 1.487, "gpu_s": 0.419, "speedup": 3.55, "transfer_ratio": 0.08},
            "kmeans": {"cpu_s": 3.217, "gpu_s": 1.104, "speedup": 2.91, "transfer_ratio": 0.12},
            "lud": {"cpu_s": 2.049, "gpu_s": 0.280, "speedup": 7.33, "transfer_ratio": 0.15},
            "nn": {"cpu_s": 0.018, "gpu_s": 0.029, "speedup": 0.65, "transfer_ratio": 0.60},
            "srad": {"cpu_s": 1.784, "gpu_s": 0.553, "speedup": 3.23, "transfer_ratio": 0.10},
            "backprop": {"cpu_s": 1.250, "gpu_s": 0.350, "speedup": 3.57, "transfer_ratio": 0.15},
            "b_tree": {"cpu_s": 0.850, "gpu_s": 0.220, "speedup": 3.86, "transfer_ratio": 0.18},
            "particlefilter": {"cpu_s": 2.450, "gpu_s": 0.620, "speedup": 3.95, "transfer_ratio": 0.12},
            "leukocyte": {"cpu_s": 5.120, "gpu_s": 0.840, "speedup": 6.10, "transfer_ratio": 0.10},
            "pathfinder": {"cpu_s": 0.950, "gpu_s": 0.420, "speedup": 2.26, "transfer_ratio": 0.20},
        }

    def predict(self, features: Dict[str, Any], workload: Optional[str] = None) -> Dict[str, Any]:
        """
        Predicts preferred device and provides estimated runtimes.
        Supports raw feature dictionaries or pre-execution feature records.
        """
        wl = workload or features.get("workload") or features.get("task_id") or "hotspot"
        input_size_mib = float(features.get("input_size_mib") or features.get("input_size_mb") or 10.0)
        input_file = str(features.get("input_file") or f"{wl}_default.dat")

        pre_feats = extract_pre_execution_features(
            workload=wl,
            input_file=input_file,
            input_size_mib=input_size_mib,
            cpu_cores=int(features.get("cpu_cores_available", 4)),
            gpu_type=str(features.get("gpu_type_encoded", "Tesla_T4")),
            gpu_memory_gb=float(features.get("gpu_memory_gb", 16.0)),
        )

        df_feat = pd.DataFrame([pre_feats])
        pred_class_idx = int(self.pipeline.predict(df_feat)[0])
        prob_dist = self.pipeline.predict_proba(df_feat)[0]

        predicted_device = "gpu" if pred_class_idx == 1 else "cpu"
        confidence = float(prob_dist[pred_class_idx])

        ref = self.reference_timings.get(wl, {
            "cpu_s": 1.5,
            "gpu_s": 0.5 if predicted_device == "gpu" else 2.0,
            "speedup": 3.0 if predicted_device == "gpu" else 0.75,
            "transfer_ratio": 0.15,
        })

        scale = max(0.1, input_size_mib / 10.0)
        cpu_est_s = round(ref["cpu_s"] * (0.8 + 0.2 * scale), 4)
        gpu_est_s = round(ref["gpu_s"] * (0.8 + 0.2 * scale), 4)
        speedup_pred = round(cpu_est_s / max(gpu_est_s, 1e-4), 3)

        return {
            "workload": wl,
            "device": predicted_device,
            "confidence": confidence,
            "cpu_est_s": cpu_est_s,
            "gpu_est_s": gpu_est_s,
            "speedup_pred": speedup_pred,
            "transfer_overhead_ratio": ref.get("transfer_ratio", 0.15),
            "probabilities": {
                "cpu": float(prob_dist[0]),
                "gpu": float(prob_dist[1]),
            },
            "pre_execution_features": pre_feats,
        }


if __name__ == "__main__":
    res = predict_device("cfd", "fvcorr.domn.193K", 43.54428, model_type="model_a")
    print("\nInference Test Output (Model A):")
    for k, v in res.items():
        if k != "pre_execution_features":
            print(f"  {k:20s}: {v}")

    print("\nWorkloadPredictor Class Test:")
    wp = WorkloadPredictor("model_a")
    pred = wp.predict({"input_size_mb": 15.0}, workload="cfd")
    for k, v in pred.items():
        if k != "pre_execution_features":
            print(f"  {k:20s}: {v}")
