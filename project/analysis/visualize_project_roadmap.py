"""
visualize_project_roadmap.py
============================
Generates an authoritative, publication-quality architectural roadmap diagram
illustrating all 3 Project Objectives, core objects, data flows, and the
Pre-3rd Objective Mid-Semester Milestone.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIGURES_DIR / "project_roadmap_objectives.png"


def create_roadmap_chart():
    fig, ax = plt.subplots(figsize=(19, 11), dpi=300)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title & Subtitle
    ax.text(
        50, 95.5,
        "CAPSTONE SYSTEM ARCHITECTURE & PROJECT OBJECTIVES ROADMAP",
        ha="center", va="center", fontsize=20, fontweight="bold", color="#f8fafc",
        fontfamily="sans-serif"
    )
    ax.text(
        50, 92.5,
        "AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduling and Resource Optimization for Cloud HPC",
        ha="center", va="center", fontsize=11.5, color="#94a3b8", fontfamily="sans-serif"
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # BOX 1: OBJECTIVE 1 (Workload Profiling & Empirical Datasets)
    # ─────────────────────────────────────────────────────────────────────────────
    b1 = patches.FancyBboxPatch(
        (3, 47), 29, 41,
        boxstyle="round,pad=1.2,rounding_size=2.0",
        facecolor="#111c33", edgecolor="#38bdf8", linewidth=2.5
    )
    ax.add_patch(b1)

    # Header Pill 1
    p1 = patches.FancyBboxPatch(
        (5, 83), 25, 4.2,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor="#0369a1", edgecolor="#38bdf8", linewidth=1.5
    )
    ax.add_patch(p1)
    ax.text(17.5, 85.1, "OBJECTIVE 1: PROFILING & DATA", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#ffffff")

    # Status Badge
    badge1 = patches.FancyBboxPatch(
        (21.5, 79.5), 9.5, 2.8,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="#065f46", edgecolor="#10b981", linewidth=1.2
    )
    ax.add_patch(badge1)
    ax.text(26.25, 80.9, "100% COMPLETE", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#ecfdf5")

    content_obj1 = (
        "• Hardware Testbed:\n"
        "  - Host: Intel Xeon 4 vCPU cores (OpenMP)\n"
        "  - Accelerator: NVIDIA Tesla T4 16GB (CUDA)\n"
        "\n"
        "• Workload Suite (Rodinia 3.1):\n"
        "  - 7 Workloads across 8 configurations:\n"
        "    BFS (large/small), CFD, HotSpot,\n"
        "    K-Means, LUD, NN, SRAD\n"
        "\n"
        "• Layer 1 Raw Execution Telemetry:\n"
        "  - 124 physical runs captured\n"
        "  - Explicit warm-up run isolation\n"
        "  - Exit code & audit failure tracking\n"
        "\n"
        "• Strict Timing Parity Gate:\n"
        "  - CPU wall loop time vs GPU Total Path\n"
        "  - GPU Total Path = H2D + Kernel + D2H\n"
        "  - Eliminates 'kernel-only' academic bias\n"
        "\n"
        "• Layer 2 Aggregated Dataset:\n"
        "  - 8 validated configurations\n"
        "  - Median runtimes, std dev, speedup\n"
        "  - CFD: 68.9x (GPU) | NN: 0.65x (CPU)"
    )
    ax.text(5, 62.5, content_obj1, ha="left", va="center",
            fontsize=8.5, color="#cbd5e1", linespacing=1.25)

    # ─────────────────────────────────────────────────────────────────────────────
    # BOX 2: OBJECTIVE 2 (Pre-Execution Feature Engineering & ML Predictor)
    # ─────────────────────────────────────────────────────────────────────────────
    b2 = patches.FancyBboxPatch(
        (35.5, 47), 29, 41,
        boxstyle="round,pad=1.2,rounding_size=2.0",
        facecolor="#0f2922", edgecolor="#34d399", linewidth=2.5
    )
    ax.add_patch(b2)

    # Header Pill 2
    p2 = patches.FancyBboxPatch(
        (37.5, 83), 25, 4.2,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor="#047857", edgecolor="#34d399", linewidth=1.5
    )
    ax.add_patch(p2)
    ax.text(50.0, 85.1, "OBJECTIVE 2: ML PREDICTION", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#ffffff")

    # Status Badge
    badge2 = patches.FancyBboxPatch(
        (54.0, 79.5), 9.5, 2.8,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="#065f46", edgecolor="#10b981", linewidth=1.2
    )
    ax.add_patch(badge2)
    ax.text(58.75, 80.9, "100% COMPLETE", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#ecfdf5")

    content_obj2 = (
        "• Pre-Execution Feature Schema:\n"
        "  - Frozen Rubric v1.0 source inspection\n"
        "  - Arithmetic intensity, parallelism degree,\n"
        "    memory boundness, transfer ratio\n"
        "\n"
        "• Strict Anti-Leakage Gate:\n"
        "  - 18 post-execution columns banned\n"
        "  - Zero telemetry leakage into ML features\n"
        "\n"
        "• Layer 3 ML Dataset:\n"
        "  - 8 configurations: 2 CPU (25%), 6 GPU (75%)\n"
        "\n"
        "• Random Forest Models:\n"
        "  - Model A: Full pre-execution features\n"
        "  - Model B: Intrinsic algorithmic only\n"
        "\n"
        "• Leave-One-Workload-Out (LOWO) CV:\n"
        "  - Accuracy: 87.50% | Macro F1: 79.49%\n"
        "  - Baseline: 75.00% Acc, 42.86% F1 (+36.6%)\n"
        "  - Zero drop when workload name removed!\n"
        "\n"
        "• Visualizations & Performance Analysis:\n"
        "  - 14 publication figures + 10-Q audit"
    )
    ax.text(37.5, 62.5, content_obj2, ha="left", va="center",
            fontsize=8.5, color="#cbd5e1", linespacing=1.25)

    # ─────────────────────────────────────────────────────────────────────────────
    # BOX 3: OBJECTIVE 3 (Adaptive Multi-Objective Scheduling & DAG Orchestration)
    # ─────────────────────────────────────────────────────────────────────────────
    b3 = patches.FancyBboxPatch(
        (68, 47), 29, 41,
        boxstyle="round,pad=1.2,rounding_size=2.0",
        facecolor="#2d2210", edgecolor="#fbbf24", linewidth=2.5
    )
    ax.add_patch(b3)

    # Header Pill 3
    p3 = patches.FancyBboxPatch(
        (70, 83), 25, 4.2,
        boxstyle="round,pad=0.4,rounding_size=1.0",
        facecolor="#b45309", edgecolor="#fbbf24", linewidth=1.5
    )
    ax.add_patch(p3)
    ax.text(82.5, 85.1, "OBJECTIVE 3: ADAPTIVE SCHEDULING", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#ffffff")

    # Status Badge
    badge3 = patches.FancyBboxPatch(
        (84.0, 79.5), 12.0, 2.8,
        boxstyle="round,pad=0.2,rounding_size=0.8",
        facecolor="#78350f", edgecolor="#fbbf24", linewidth=1.2
    )
    ax.add_patch(badge3)
    ax.text(90.0, 80.9, "CORE BUILT & TESTED", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#fef3c7")

    content_obj3 = (
        "• WorkloadPredictor Inference API:\n"
        "  - Seamless wrapper over Model A & B\n"
        "  - Predicts device, confidence & runtime\n"
        "\n"
        "• WorkloadDAG Dependency Graph:\n"
        "  - Task nodes, dependency edges, cycle check\n"
        "  - Critical path length priority calculation\n"
        "  - Dependency-respecting topological order\n"
        "\n"
        "• Multi-Objective Optimization Engine:\n"
        "  - f1: Makespan (parallel execution slots)\n"
        "  - f2: Energy proxy (Watt-seconds)\n"
        "  - f3: PCIe transfer cost (overhead penalty)\n"
        "  - Weighted Sum & NSGA-II Pareto Solver\n"
        "\n"
        "• Bayesian Weight Adaptation:\n"
        "  - Dirichlet posterior updating\n"
        "  - Thompson Sampling for dynamic weights\n"
        "\n"
        "• Parallel Execution Dispatcher:\n"
        "  - Concurrent CPU (OpenMP) & GPU (CUDA)"
    )
    ax.text(70, 62.5, content_obj3, ha="left", va="center",
            fontsize=8.5, color="#cbd5e1", linespacing=1.25)

    # ─────────────────────────────────────────────────────────────────────────────
    # CONNECTING ARROWS BETWEEN OBJECTIVES
    # ─────────────────────────────────────────────────────────────────────────────
    ax.annotate("", xy=(34.5, 67.5), xytext=(32.5, 67.5),
                arrowprops=dict(arrowstyle="-|>", color="#38bdf8", lw=3.0, mutation_scale=20))
    ax.text(33.5, 70.0, "L2 Data", ha="center", va="bottom", fontsize=8.5, color="#38bdf8", fontweight="bold")

    ax.annotate("", xy=(67.0, 67.5), xytext=(65.0, 67.5),
                arrowprops=dict(arrowstyle="-|>", color="#34d399", lw=3.0, mutation_scale=20))
    ax.text(66.0, 70.0, "ML Models", ha="center", va="bottom", fontsize=8.5, color="#34d399", fontweight="bold")

    # ─────────────────────────────────────────────────────────────────────────────
    # BANNER: PRE-3RD OBJECTIVE MILESTONE & MID-SEM READINESS
    # ─────────────────────────────────────────────────────────────────────────────
    banner = patches.FancyBboxPatch(
        (3, 22), 94, 21,
        boxstyle="round,pad=1.2,rounding_size=2.0",
        facecolor="#1e1b4b", edgecolor="#818cf8", linewidth=2.8
    )
    ax.add_patch(banner)

    # Milestone Tag
    tag = patches.FancyBboxPatch(
        (6, 38.5), 44, 3.8,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#4338ca", edgecolor="#a5b4fc", linewidth=1.5
    )
    ax.add_patch(tag)
    ax.text(28.0, 40.4, "★ CURRENT MILESTONE: PRE-3RD OBJECTIVE BOUNDARY ★",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#ffffff")

    # Mid-Sem Verification Tag
    vtag = patches.FancyBboxPatch(
        (52, 38.5), 42, 3.8,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#065f46", edgecolor="#34d399", linewidth=1.5
    )
    ax.add_patch(vtag)
    ax.text(73.0, 40.4, "✓ 100% READY FOR MID-SEM PRESENTATION",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#ffffff")

    milestone_text_col1 = (
        "• Mid-Semester Project State Assessment:\n"
        "  ✓ Objective 1 (Empirical Profiling): 100% complete across all 7 target Rodinia workloads.\n"
        "  ✓ Objective 2 (ML Device Prediction): 100% complete (87.5% acc, 79.5% F1, 14 publication figures).\n"
        "  ✓ Objective 3 (Adaptive Scheduling): Core algorithms, DAG builder, WorkloadPredictor API,\n"
        "    and Bayesian Thompson sampler built and validated with 18 automated tests passing in 0.64s."
    )
    ax.text(6, 31.5, milestone_text_col1, ha="left", va="center",
            fontsize=9.0, color="#e2e8f0", linespacing=1.35)

    milestone_text_col2 = (
        "• Planned Roadmap for Post-Mid-Sem Execution (Second Half of Semester):\n"
        "  1. Live Cloud Cluster Deployment: Launch multi-task parallel workloads on live multi-core/GPU instances.\n"
        "  2. Dynamic DAG Stress Testing: Execute real dependency pipelines (e.g. BFS → CFD → Pathfinder).\n"
        "  3. Online Bayesian Feedback Loop: Track online makespan and energy reduction vs static FIFO/HEFT baselines.\n"
        "  4. Final Comparative Benchmark: Quantify energy efficiency gains (Joules saved) and makespan speedups."
    )
    ax.text(52, 31.5, milestone_text_col2, ha="left", va="center",
            fontsize=9.0, color="#fde68a", linespacing=1.35)

    # ─────────────────────────────────────────────────────────────────────────────
    # FOOTER: KEY PERFORMANCE SUMMARY CARDS
    # ─────────────────────────────────────────────────────────────────────────────
    stat_boxes = [
        ("Physical Runs", "124 Runs", "Layer 1 Telemetry (7 Workloads)", "#0284c7"),
        ("ML Accuracy", "87.50%", "LOWO CV (Model A & Model B)", "#059669"),
        ("Macro F1", "79.49%", "+36.63% over Baseline (42.86%)", "#10b981"),
        ("Max GPU Speedup", "68.92x", "CFD on Tesla T4 (High AI)", "#d97706"),
        ("CPU Best Speedup", "0.14x / 0.65x", "BFS-small / NN (Transfer Bottleneck)", "#dc2626"),
        ("Figures & Tests", "14 Figs | 18 Tests", "All Validated & Passing Cleanly", "#7c3aed"),
    ]

    box_w = 14.5
    gap = 1.3
    start_x = 3.0

    for idx, (title, val, sub, col) in enumerate(stat_boxes):
        cur_x = start_x + idx * (box_w + gap)
        sbox = patches.FancyBboxPatch(
            (cur_x, 3.5), box_w, 14.5,
            boxstyle="round,pad=0.8,rounding_size=1.2",
            facecolor="#1e293b", edgecolor=col, linewidth=2.0
        )
        ax.add_patch(sbox)
        ax.text(cur_x + box_w / 2.0, 15.2, title.upper(), ha="center", va="center",
                fontsize=7.8, fontweight="bold", color="#94a3b8")
        ax.text(cur_x + box_w / 2.0, 10.8, val, ha="center", va="center",
                fontsize=13.0, fontweight="bold", color="#ffffff")
        ax.text(cur_x + box_w / 2.0, 6.2, sub, ha="center", va="center",
                fontsize=6.8, color="#cbd5e1", wrap=True)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Roadmap chart successfully generated at: {OUT_PATH}")


if __name__ == "__main__":
    create_roadmap_chart()
