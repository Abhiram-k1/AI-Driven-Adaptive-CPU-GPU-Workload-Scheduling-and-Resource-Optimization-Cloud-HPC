"""
visualize_pareto.py
===================
Generates paper-quality visualizations for the project:

  1. CPU vs GPU speedup bar chart (per workload)
  2. Pareto front scatter plot (Makespan vs Energy vs Transfer cost)
  3. Adaptive weight evolution over time
  4. Scheduler comparison: Naive CPU | Naive GPU | Rule-based | AI Scheduler

Usage:
    python analysis/visualize_pareto.py --data data/profiles/rodinia_benchmark_data.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

OUTDIR = Path(__file__).parent.parent / "analysis" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "cpu":        "#4A90D9",
    "gpu":        "#E86A3A",
    "ai":         "#4CAF50",
    "rule":       "#9C27B0",
    "pareto":     "#FF6B6B",
    "dominated":  "#BBBBBB",
}

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":     150,
})


# ── 1. Speedup bar chart ────────────────────────────────────────────────────

def plot_speedup(df: pd.DataFrame, out: Path = None) -> None:
    """Bar chart: CPU mean vs GPU mean runtime per workload."""
    cpu = df[df["device"] == "cpu"].groupby("workload")["elapsed_s"].mean()
    gpu = df[df["device"] == "gpu"].groupby("workload")["elapsed_s"].mean()
    speedup = (cpu / gpu).dropna().sort_values(ascending=False)
    workloads = speedup.index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw runtimes
    ax = axes[0]
    x = np.arange(len(workloads))
    w = 0.35
    ax.bar(x - w/2, [cpu[wl] for wl in workloads], w, label="CPU (OpenMP)",
           color=PALETTE["cpu"], alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, [gpu[wl] for wl in workloads], w, label="GPU (CUDA)",
           color=PALETTE["gpu"], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=35, ha="right")
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("Rodinia 3.1: CPU vs GPU Runtime")
    ax.legend()

    # Right: speedup
    ax2 = axes[1]
    colors = [PALETTE["gpu"] if s > 1 else PALETTE["cpu"] for s in speedup.values]
    bars = ax2.bar(workloads, speedup.values, color=colors, alpha=0.85, edgecolor="white")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Break-even")
    ax2.set_xticklabels(workloads, rotation=35, ha="right")
    ax2.set_ylabel("GPU Speedup (CPU time / GPU time)")
    ax2.set_title("GPU Speedup per Workload")
    ax2.legend()

    # Annotate bars
    for bar, val in zip(bars, speedup.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:.1f}×", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = out or OUTDIR / "speedup_chart.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 2. Pareto front ─────────────────────────────────────────────────────────

def plot_pareto_front(solutions: list[tuple], out: Path = None) -> None:
    """
    Scatter plot of multi-objective solutions.
    solutions: list of (makespan, energy, transfer_cost, label, is_ai)
    """
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111)

    for ms, en, tc, label, is_pareto in solutions:
        color  = PALETTE["pareto"] if is_pareto else PALETTE["dominated"]
        marker = "★" if is_pareto else "o"
        ax.scatter(ms, en / 1000, s=120 if is_pareto else 60,
                   c=color, zorder=5 if is_pareto else 3,
                   edgecolors="black" if is_pareto else "none", linewidths=0.5)
        if is_pareto:
            ax.annotate(label, (ms, en / 1000),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

    pareto_patch   = mpatches.Patch(color=PALETTE["pareto"],   label="Pareto-optimal")
    dominated_patch = mpatches.Patch(color=PALETTE["dominated"], label="Dominated")
    ax.legend(handles=[pareto_patch, dominated_patch])
    ax.set_xlabel("Makespan (s)")
    ax.set_ylabel("Energy (kJ proxy)")
    ax.set_title("Multi-Objective Pareto Front\n(Makespan vs Energy, bubble size ∝ Transfer Cost)")
    plt.tight_layout()

    out = out or OUTDIR / "pareto_front.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 3. Adaptive weight evolution ────────────────────────────────────────────

def plot_weight_evolution(weight_history: list[tuple], out: Path = None) -> None:
    """Line plot of how w1/w2/w3 evolve over scheduling rounds."""
    rounds = list(range(1, len(weight_history) + 1))
    w1s = [w[0] for w in weight_history]
    w2s = [w[1] for w in weight_history]
    w3s = [w[2] for w in weight_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, w1s, "-o", color="#E53935", label="w₁ Makespan",  linewidth=2, markersize=5)
    ax.plot(rounds, w2s, "-s", color="#1E88E5", label="w₂ Energy",    linewidth=2, markersize=5)
    ax.plot(rounds, w3s, "-^", color="#43A047", label="w₃ Transfer",  linewidth=2, markersize=5)
    ax.set_xlabel("Scheduling Round")
    ax.set_ylabel("Objective Weight")
    ax.set_title("Adaptive Objective Weights (Bayesian Online Learning)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    out = out or OUTDIR / "weight_evolution.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 4. Scheduler comparison ─────────────────────────────────────────────────

def plot_scheduler_comparison(
    workloads: list[str],
    cpu_times: list[float],
    gpu_times: list[float],
    ai_times:  list[float],
    out: Path = None,
) -> None:
    """Bar chart comparing four scheduling strategies."""
    x = np.arange(len(workloads))
    w = 0.2

    # Rule-based: always GPU if speedup > 1.5, else CPU
    rule_times = [
        min(c, g) if c / max(g, 1e-6) > 1.5 else c
        for c, g in zip(cpu_times, gpu_times)
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*w, cpu_times,  w, label="CPU-only",   color=PALETTE["cpu"],  alpha=0.85, edgecolor="white")
    ax.bar(x - 0.5*w, gpu_times,  w, label="GPU-only",   color=PALETTE["gpu"],  alpha=0.85, edgecolor="white")
    ax.bar(x + 0.5*w, rule_times, w, label="Rule-based", color=PALETTE["rule"], alpha=0.85, edgecolor="white")
    ax.bar(x + 1.5*w, ai_times,   w, label="AI Scheduler (ours)", color=PALETTE["ai"],  alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=30, ha="right")
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("Scheduler Strategy Comparison\nCPU-only vs GPU-only vs Rule-based vs AI Scheduler")
    ax.legend()
    plt.tight_layout()

    out = out or OUTDIR / "scheduler_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else (
        Path(__file__).parent.parent / "data" / "profiles" / "rodinia_benchmark_data.csv"
    )

    if data_path.exists():
        df = pd.read_csv(data_path)
        df = df[df["success"] == 1]
        print(f"Loaded {len(df)} rows from {data_path}")
        plot_speedup(df)
    else:
        print(f"Data not found at {data_path}. Generating demo plots...")
        _demo_plots()


def _demo_plots():
    """Generate demo plots with synthetic data for development."""
    np.random.seed(42)

    workloads = ["bfs", "cfd", "hotspot", "lud", "kmeans", "srad", "nw", "backprop"]
    cpu_t = [0.85, 2.10, 1.50, 2.10, 3.20, 1.80, 1.30, 1.00]
    gpu_t = [0.95, 0.30, 0.42, 0.28, 1.10, 0.55, 0.70, 0.48]

    # Synthetic raw data for speedup chart
    rows = []
    for wl, ct, gt in zip(workloads, cpu_t, gpu_t):
        for _ in range(6):
            rows.append({"workload": wl, "device": "cpu",
                         "elapsed_s": ct * np.random.uniform(0.95, 1.05), "success": 1})
            rows.append({"workload": wl, "device": "gpu",
                         "elapsed_s": gt * np.random.uniform(0.95, 1.05), "success": 1})
    df = pd.DataFrame(rows)
    plot_speedup(df)

    # Pareto front
    solutions = [
        (0.30, 3500, 0.003, "GPU-heavy",    True),
        (0.55, 2200, 0.001, "Balanced",     True),
        (0.85, 1200, 0.000, "CPU-heavy",    True),
        (0.42, 3000, 0.002, "AI Scheduler", True),
        (0.60, 2800, 0.002, "",             False),
        (0.70, 2600, 0.001, "",             False),
        (0.45, 3200, 0.003, "",             False),
    ]
    plot_pareto_front(solutions)

    # Weight evolution
    from scheduler.adaptive_weights import AdaptiveWeightController
    awc = AdaptiveWeightController()
    history = []
    for i in range(20):
        tc = np.random.uniform(0.5 + i*0.05, 1.5 + i*0.05)
        awc.update(np.random.uniform(0.5, 2), np.random.uniform(0.3, 1), tc)
        history.append(awc.mean_weights())
    plot_weight_evolution(history)

    # Scheduler comparison
    ai_t = [min(c, g) * 0.95 if c/max(g,1e-6) > 1.1 else c for c, g in zip(cpu_t, gpu_t)]
    plot_scheduler_comparison(workloads, cpu_t, gpu_t, ai_t)


if __name__ == "__main__":
    main()
