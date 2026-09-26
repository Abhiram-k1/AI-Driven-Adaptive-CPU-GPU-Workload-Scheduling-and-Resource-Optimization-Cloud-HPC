"""
analysis/visualization.py
=========================
Generates all 9 mandatory ML performance visualizations.
Strictly implements Section 22 and Section 23 of the Master Execution Prompt.

Visualizations:
1. class_distribution.png              - CPU vs GPU labelled configurations
2. model_a_confusion_matrix.png        - Model A Confusion Matrix (LOWO predictions)
3. model_b_confusion_matrix.png        - Model B Confusion Matrix (LOWO predictions)
4. model_comparison.png                - Grouped bar chart (Accuracy, Precision, Recall, F1)
5. per_workload_f1.png                 - Per-workload Accuracy and F1 breakdown
6. feature_importance_model_a.png      - Model A Random Forest feature importances
   feature_importance_model_b.png      - Model B Random Forest feature importances
7. lowo_comparison.png                 - Model A vs Model B comparison across held-out workloads
8. prediction_confidence.png           - Confidence distribution for correct vs incorrect
9. actual_vs_predicted_distribution.png- Actual vs Predicted class distribution comparison
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import load_model

DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_PRED_PATH = Path(__file__).resolve().parent.parent / "results" / "predictions" / "lowo_predictions.csv"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics" / "model_evaluation_metrics.json"
DEFAULT_LAYER3_PATH = Path(__file__).resolve().parent.parent / "dataset" / "layer3_ml_dataset.csv"

# Global style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300
})


def generate_all_visualizations(
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    pred_path: Path = DEFAULT_PRED_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    layer3_path: Path = DEFAULT_LAYER3_PATH
):
    """Generates and saves all 9 required figures deterministically."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("GENERATING MANDATORY ML VISUALIZATIONS (SECTION 22 & 23)")
    print("=" * 80)

    df_ml = pd.read_csv(layer3_path)
    df_preds = pd.read_csv(pred_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Class Distribution
    fig1_path = figures_dir / "class_distribution.png"
    plot_class_distribution(df_ml, fig1_path)

    # 2. Model A Confusion Matrix
    fig2_path = figures_dir / "model_a_confusion_matrix.png"
    plot_confusion_matrix(metrics["model_a"]["confusion_matrix"], "Model A (Full Features) Confusion Matrix", fig2_path)

    # 3. Model B Confusion Matrix
    fig3_path = figures_dir / "model_b_confusion_matrix.png"
    plot_confusion_matrix(metrics["model_b"]["confusion_matrix"], "Model B (Ablated Features) Confusion Matrix", fig3_path)

    # 4. Model Comparison Bar Chart
    fig4_path = figures_dir / "model_comparison.png"
    plot_model_comparison(metrics, fig4_path)

    # 5. Per-Workload Performance
    fig5_path = figures_dir / "per_workload_f1.png"
    plot_per_workload_performance(metrics["per_workload"], fig5_path)

    # 6. Feature Importance (Model A and Model B)
    fig6a_path = figures_dir / "feature_importance_model_a.png"
    fig6b_path = figures_dir / "feature_importance_model_b.png"
    plot_feature_importances("model_a", fig6a_path)
    plot_feature_importances("model_b", fig6b_path)

    # 7. Model A vs Model B LOWO Comparison
    fig7_path = figures_dir / "lowo_comparison.png"
    plot_lowo_comparison(metrics["per_workload"], fig7_path)

    # 8. Prediction Confidence Distribution
    fig8_path = figures_dir / "prediction_confidence.png"
    plot_prediction_confidence(df_preds, fig8_path)

    # 9. Actual vs Predicted Class Distribution
    fig9_path = figures_dir / "actual_vs_predicted_distribution.png"
    plot_actual_vs_predicted_distribution(df_preds, fig9_path)

    print("\nAll 9 mandatory visualizations successfully created and saved at:")
    print(f"  {figures_dir}")
    print("=" * 80)


def plot_class_distribution(df: pd.DataFrame, out_path: Path):
    """Visualization 1: Class distribution showing CPU vs GPU labels."""
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["preferred_device"].value_counts()
    classes = ["CPU", "GPU"]
    vals = [counts.get("cpu", 0), counts.get("gpu", 0)]
    colors = ["#4C72B0", "#DD8452"]

    bars = ax.bar(classes, vals, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    ax.set_ylabel("Number of Configurations", fontsize=12)
    ax.set_title("Ground-Truth Preferred-Device Class Distribution", fontsize=13, pad=15)
    ax.set_ylim(0, max(vals) + 1.5 if vals else 2)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{int(h)}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path.name}")


def plot_confusion_matrix(cm_list: list, title: str, out_path: Path):
    """Visualizations 2 & 3: Normalized and count confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array(cm_list)
    labels = ["CPU (0)", "GPU (1)"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=1.5, linecolor="gray", annot_kws={"size": 14, "weight": "bold"})

    ax.set_xlabel("Predicted Device Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Ground-Truth Device", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=15)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path.name}")


def plot_model_comparison(metrics: dict, out_path: Path):
    """Visualization 4: Grouped bar chart comparing Model A vs Model B vs Baseline."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]

    base_vals = [metrics["majority_baseline"][k] for k in metric_keys]
    a_vals = [metrics["model_a"][k] for k in metric_keys]
    b_vals = [metrics["model_b"][k] for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.25

    rects1 = ax.bar(x - width, base_vals, width, label="Majority Baseline", color="#8C8C8C", edgecolor="black")
    rects2 = ax.bar(x, a_vals, width, label="Model A (Full Features)", color="#1F77B4", edgecolor="black")
    rects3 = ax.bar(x + width, b_vals, width, label="Model B (Ablated Identity)", color="#2CA02C", edgecolor="black")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model A vs Model B vs Baseline Performance Comparison", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path.name}")


def plot_per_workload_performance(per_workload: dict, out_path: Path):
    """Visualization 5: Per-workload performance breakdown."""
    fig, ax = plt.subplots(figsize=(7, 5))
    workloads = list(per_workload.keys())
    a_acc = [per_workload[w]["model_a_accuracy"] for w in workloads]
    b_acc = [per_workload[w]["model_b_accuracy"] for w in workloads]

    x = np.arange(len(workloads))
    width = 0.35

    ax.bar(x - width/2, a_acc, width, label="Model A (Accuracy)", color="#1F77B4", edgecolor="black")
    ax.bar(x + width/2, b_acc, width, label="Model B (Accuracy)", color="#2CA02C", edgecolor="black")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Per-Workload Accuracy (Held-Out Evaluation)", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([w.upper() for w in workloads], fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path.name}")


def plot_feature_importances(model_type: str, out_path: Path):
    """Visualization 6: Sorted Random Forest feature importances."""
    pipeline = load_model(model_type)
    fi = pipeline.get_feature_importances()

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(fi))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi)))

    ax.barh(y_pos, fi.values, color=colors, edgecolor="black", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi.index, fontsize=10)
    ax.invert_yaxis()  # top-down highest to lowest
    ax.set_xlabel("Relative Feature Importance (MDI)", fontsize=12)
    label_name = "Model A (Full Features)" if model_type == "model_a" else "Model B (Ablated Features)"
    ax.set_title(f"Random Forest Feature Importances — {label_name}", fontsize=13, pad=15)
    ax.set_xlim(0, max(fi.values) + 0.1 if max(fi.values) > 0 else 1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path.name}")


def plot_lowo_comparison(per_workload: dict, out_path: Path):
    """Visualization 7: Model A vs Model B across held-out workloads."""
    fig, ax = plt.subplots(figsize=(7, 5))
    workloads = list(per_workload.keys())
    a_f1 = [per_workload[w]["model_a_f1"] for w in workloads]
    b_f1 = [per_workload[w]["model_b_f1"] for w in workloads]

    x = np.arange(len(workloads))
    width = 0.35

    ax.bar(x - width/2, a_f1, width, label="Model A F1", color="#1F77B4", edgecolor="black")
    ax.bar(x + width/2, b_f1, width, label="Model B F1", color="#FF7F0E", edgecolor="black")

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Model A vs Model B LOWO F1 Comparison", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([w.upper() for w in workloads], fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path.name}")


def plot_prediction_confidence(df_preds: pd.DataFrame, out_path: Path):
    """Visualization 8: Prediction confidence distribution for Model A."""
    fig, ax = plt.subplots(figsize=(7, 5))

    conf_correct = df_preds[df_preds["model_a_correct"] == True]["model_a_conf"]
    conf_incorrect = df_preds[df_preds["model_a_correct"] == False]["model_a_conf"]

    bins = np.linspace(0.0, 1.0, 11)
    ax.hist([conf_correct, conf_incorrect], bins=bins, label=["Correct", "Incorrect"],
            color=["#2CA02C", "#D62728"], edgecolor="black", stacked=True, width=0.08)

    ax.set_xlabel("Prediction Confidence (Assigned Class Probability)", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Model A Prediction Confidence Distribution", fontsize=13, pad=15)
    ax.set_xlim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path.name}")


def plot_actual_vs_predicted_distribution(df_preds: pd.DataFrame, out_path: Path):
    """Visualization 9: Actual vs Predicted class distribution to detect class bias."""
    fig, ax = plt.subplots(figsize=(7, 5))

    classes = ["CPU", "GPU"]
    actual_counts = [(df_preds["actual_device"] == "cpu").sum(), (df_preds["actual_device"] == "gpu").sum()]
    pred_a_counts = [(df_preds["model_a_pred_device"] == "cpu").sum(), (df_preds["model_a_pred_device"] == "gpu").sum()]
    pred_b_counts = [(df_preds["model_b_pred_device"] == "cpu").sum(), (df_preds["model_b_pred_device"] == "gpu").sum()]

    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, actual_counts, width, label="Actual Ground-Truth", color="#333333", edgecolor="black")
    ax.bar(x, pred_a_counts, width, label="Model A Predicted", color="#1F77B4", edgecolor="black")
    ax.bar(x + width, pred_b_counts, width, label="Model B Predicted", color="#2CA02C", edgecolor="black")

    ax.set_ylabel("Configuration Count", fontsize=12)
    ax.set_title("Actual vs Predicted Class Distribution (Bias Check)", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, max(actual_counts + pred_a_counts + pred_b_counts) + 1.5 if actual_counts else 2)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [SAVED] {out_path.name}")


if __name__ == "__main__":
    generate_all_visualizations()
