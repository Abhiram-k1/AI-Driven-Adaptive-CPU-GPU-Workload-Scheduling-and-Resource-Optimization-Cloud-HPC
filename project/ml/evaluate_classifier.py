"""
ml/evaluate_classifier.py
=========================
Leave-One-Workload-Out (LOWO) evaluation harness.
Strictly implements:
  - Section 20: Grouped LOWO cross-validation (no samples of the same workload in train & test)
  - Section 21: Required ML metrics:
      Accuracy, Precision, Recall, F1 (macro and per-class)
      Per-workload metrics
      Majority-class baseline
      Multi-model comparison: Model A (RF Full), Model B (RF Ablated), Model C (Gradient Boosting)
  - Saves predictions and metrics into results/
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_schema import MODEL_A_FEATURES, MODEL_B_FEATURES
from ml.train_classifier import DeviceClassifierPipeline

DEFAULT_LAYER3_PATH = Path(__file__).resolve().parent.parent / "dataset" / "layer3_ml_dataset.csv"
DEFAULT_PRED_DIR = Path(__file__).resolve().parent.parent / "results" / "predictions"
DEFAULT_METRICS_DIR = Path(__file__).resolve().parent.parent / "results" / "metrics"


def evaluate_lowo(
    layer3_path: Path = DEFAULT_LAYER3_PATH,
    predictions_dir: Path = DEFAULT_PRED_DIR,
    metrics_dir: Path = DEFAULT_METRICS_DIR
) -> Dict[str, Any]:
    if not layer3_path.exists():
        raise FileNotFoundError(f"Layer 3 dataset not found at {layer3_path}")

    df_ml = pd.read_csv(layer3_path)
    if len(df_ml) == 0:
        raise ValueError("Layer 3 dataset is empty! Cannot evaluate.")

    print("\n" + "=" * 80)
    print("LEAVE-ONE-WORKLOAD-OUT (LOWO) EVALUATION")
    print("=" * 80)

    unique_workloads = df_ml["workload"].unique().tolist()
    n_workloads = len(unique_workloads)
    print(f"Total configurations: {len(df_ml)} | Distinct workloads: {n_workloads} ({unique_workloads})")

    # Majority baseline calculation
    class_counts = df_ml["preferred_device"].value_counts()
    majority_class_str = class_counts.idxmax()
    majority_class_int = 1 if majority_class_str == "gpu" else 0
    majority_baseline_acc = float(class_counts.max() / len(df_ml))
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Majority-class baseline: {majority_class_str.upper()} ({majority_baseline_acc * 100:.2f}%)")

    predictions_records = []
    lowo_status = "Stable"

    if n_workloads < 2:
        lowo_status = "Unstable / Insufficient workloads for multi-workload leave-out split (n=1)"
        print(f"\n[LOWO NOTICE] {lowo_status}")
        y_true = (df_ml["preferred_device"] == "gpu").astype(int).values
        model_a = DeviceClassifierPipeline(MODEL_A_FEATURES, classifier_type="rf", random_state=42).fit(df_ml, y_true)
        model_b = DeviceClassifierPipeline(MODEL_B_FEATURES, classifier_type="rf", random_state=42).fit(df_ml, y_true)
        model_c = DeviceClassifierPipeline(MODEL_A_FEATURES, classifier_type="gb", random_state=42).fit(df_ml, y_true)

        pred_a, prob_a = model_a.predict(df_ml), model_a.predict_proba(df_ml)
        pred_b, prob_b = model_b.predict(df_ml), model_b.predict_proba(df_ml)
        pred_c, prob_c = model_c.predict(df_ml), model_c.predict_proba(df_ml)

        for i, row in df_ml.iterrows():
            predictions_records.append({
                "workload": row["workload"],
                "input_file": row["input_file"],
                "actual_device": row["preferred_device"],
                "actual_label": int(y_true[i]),
                "majority_baseline_pred": majority_class_int,
                "model_a_pred": int(pred_a[i]),
                "model_a_pred_device": "gpu" if pred_a[i] == 1 else "cpu",
                "model_a_conf": float(prob_a[i, int(pred_a[i])]),
                "model_a_correct": bool(pred_a[i] == y_true[i]),
                "model_b_pred": int(pred_b[i]),
                "model_b_pred_device": "gpu" if pred_b[i] == 1 else "cpu",
                "model_b_conf": float(prob_b[i, int(pred_b[i])]),
                "model_b_correct": bool(pred_b[i] == y_true[i]),
                "model_c_pred": int(pred_c[i]),
                "model_c_pred_device": "gpu" if pred_c[i] == 1 else "cpu",
                "model_c_conf": float(prob_c[i, int(pred_c[i])]),
                "model_c_correct": bool(pred_c[i] == y_true[i]),
                "fold_type": "in_sample_evaluated_gated_holdout"
            })
    else:
        # Standard LOWO iteration across distinct workloads
        for held_out in unique_workloads:
            train_df = df_ml[df_ml["workload"] != held_out].copy()
            test_df = df_ml[df_ml["workload"] == held_out].copy()

            y_train = (train_df["preferred_device"] == "gpu").astype(int).values
            y_test = (test_df["preferred_device"] == "gpu").astype(int).values

            model_a = DeviceClassifierPipeline(MODEL_A_FEATURES, classifier_type="rf", random_state=42).fit(train_df, y_train)
            model_b = DeviceClassifierPipeline(MODEL_B_FEATURES, classifier_type="rf", random_state=42).fit(train_df, y_train)
            model_c = DeviceClassifierPipeline(MODEL_A_FEATURES, classifier_type="gb", random_state=42).fit(train_df, y_train)

            pred_a, prob_a = model_a.predict(test_df), model_a.predict_proba(test_df)
            pred_b, prob_b = model_b.predict(test_df), model_b.predict_proba(test_df)
            pred_c, prob_c = model_c.predict(test_df), model_c.predict_proba(test_df)

            for i, (_, row) in enumerate(test_df.iterrows()):
                predictions_records.append({
                    "workload": row["workload"],
                    "input_file": row["input_file"],
                    "actual_device": row["preferred_device"],
                    "actual_label": int(y_test[i]),
                    "majority_baseline_pred": majority_class_int,
                    "model_a_pred": int(pred_a[i]),
                    "model_a_pred_device": "gpu" if pred_a[i] == 1 else "cpu",
                    "model_a_conf": float(prob_a[i, int(pred_a[i])]),
                    "model_a_correct": bool(pred_a[i] == y_test[i]),
                    "model_b_pred": int(pred_b[i]),
                    "model_b_pred_device": "gpu" if pred_b[i] == 1 else "cpu",
                    "model_b_conf": float(prob_b[i, int(pred_b[i])]),
                    "model_b_correct": bool(pred_b[i] == y_test[i]),
                    "model_c_pred": int(pred_c[i]),
                    "model_c_pred_device": "gpu" if pred_c[i] == 1 else "cpu",
                    "model_c_conf": float(prob_c[i, int(pred_c[i])]),
                    "model_c_correct": bool(pred_c[i] == y_test[i]),
                    "fold_type": "lowo_held_out"
                })

    df_preds = pd.DataFrame(predictions_records)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_csv_path = predictions_dir / "lowo_predictions.csv"
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"Predictions saved to: {pred_csv_path}")

    # Compute Metrics
    metrics_a = compute_model_metrics(df_preds["actual_label"].values, df_preds["model_a_pred"].values)
    metrics_b = compute_model_metrics(df_preds["actual_label"].values, df_preds["model_b_pred"].values)
    metrics_c = compute_model_metrics(df_preds["actual_label"].values, df_preds["model_c_pred"].values)
    metrics_base = compute_model_metrics(df_preds["actual_label"].values, df_preds["majority_baseline_pred"].values)

    # Per-workload performance breakdown
    per_workload_metrics = {}
    for w in df_preds["workload"].unique():
        w_df = df_preds[df_preds["workload"] == w]
        w_y = w_df["actual_label"].values
        w_pa = w_df["model_a_pred"].values
        w_pb = w_df["model_b_pred"].values
        w_pc = w_df["model_c_pred"].values
        per_workload_metrics[w] = {
            "n_samples": len(w_df),
            "actual_class": w_df["actual_device"].iloc[0],
            "model_a_accuracy": float(accuracy_score(w_y, w_pa)),
            "model_a_f1": float(f1_score(w_y, w_pa, zero_division=0)),
            "model_b_accuracy": float(accuracy_score(w_y, w_pb)),
            "model_b_f1": float(f1_score(w_y, w_pb, zero_division=0)),
            "model_c_accuracy": float(accuracy_score(w_y, w_pc)),
            "model_c_f1": float(f1_score(w_y, w_pc, zero_division=0)),
        }

    full_evaluation = {
        "dataset_summary": {
            "total_samples": len(df_ml),
            "n_workloads": n_workloads,
            "workloads": unique_workloads,
            "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
            "majority_class": majority_class_str,
            "majority_baseline_accuracy": round(majority_baseline_acc, 4),
            "lowo_status": lowo_status
        },
        "majority_baseline": metrics_base,
        "model_a": metrics_a,
        "model_b": metrics_b,
        "model_c": metrics_c,
        "per_workload": per_workload_metrics
    }

    metrics_json_path = metrics_dir / "model_evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_evaluation, f, indent=2)
    print(f"Metrics saved to: {metrics_json_path}")

    # Display Metrics Summary Table
    print("\n" + "=" * 90)
    print("MODEL PERFORMANCE SUMMARY (LOWO EVALUATION)")
    print("=" * 90)
    print(f"{'Metric':<20} | {'Majority Base':<14} | {'Model A (RF)':<14} | {'Model B (Abl)':<14} | {'Model C (GB)':<14}")
    print("-" * 90)
    print(f"{'Accuracy':<20} | {metrics_base['accuracy']:<14.4f} | {metrics_a['accuracy']:<14.4f} | {metrics_b['accuracy']:<14.4f} | {metrics_c['accuracy']:<14.4f}")
    print(f"{'Precision (Macro)':<20} | {metrics_base['precision_macro']:<14.4f} | {metrics_a['precision_macro']:<14.4f} | {metrics_b['precision_macro']:<14.4f} | {metrics_c['precision_macro']:<14.4f}")
    print(f"{'Recall (Macro)':<20} | {metrics_base['recall_macro']:<14.4f} | {metrics_a['recall_macro']:<14.4f} | {metrics_b['recall_macro']:<14.4f} | {metrics_c['recall_macro']:<14.4f}")
    print(f"{'F1-Score (Macro)':<20} | {metrics_base['f1_macro']:<14.4f} | {metrics_a['f1_macro']:<14.4f} | {metrics_b['f1_macro']:<14.4f} | {metrics_c['f1_macro']:<14.4f}")
    print("=" * 90)

    return full_evaluation


def compute_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_cpu = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_cpu = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_cpu = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    prec_gpu = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_gpu = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_gpu = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "per_class": {
            "cpu": {"precision": round(prec_cpu, 4), "recall": round(rec_cpu, 4), "f1": round(f1_cpu, 4)},
            "gpu": {"precision": round(prec_gpu, 4), "recall": round(rec_gpu, 4), "f1": round(f1_gpu, 4)},
        },
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    evaluate_lowo()
