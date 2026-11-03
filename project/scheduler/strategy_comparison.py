"""
scheduler/strategy_comparison.py
================================
Benchmarking and comparative evaluation across multiple scheduling strategies:
  1. AI-Adaptive   : ML device prediction + Bayesian dynamic multi-objective weighting
  2. FIFO-CPU      : Baseline assigning all tasks to CPU slots
  3. FIFO-GPU      : Baseline assigning all tasks to GPU slots (bottlenecked by single GPU slot)
  4. Round-Robin   : Alternates assignment between CPU and GPU
  5. Greedy-Speedup: Assigns each task to whichever device yields higher single-task speedup
  6. Random Baseline: Uniform random assignment between CPU and GPU

Evaluates:
  - Total workflow makespan (seconds)
  - Total energy consumption (Joules, TDP proxy model)
  - Device assignment accuracy vs optimal empirical device
  - GPU speedup factor relative to pure CPU baseline
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profiler.rodinia_configs import WORKLOAD_CONFIG
try:
    from scheduler.scheduler import AIScheduler
    from scheduler.simulation_mode import SimulatedExecutor
    from scheduler.multi_objective import _compute_objectives
except (ImportError, ModuleNotFoundError):
    from scheduler import AIScheduler
    from simulation_mode import SimulatedExecutor
    from multi_objective import _compute_objectives
from profiler.energy_estimator import EnergyEstimator

DEFAULT_FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"


class StrategyComparison:
    """
    Executes and compares all 6 scheduling strategies on a common workload suite.
    """

    def __init__(
        self,
        workloads: Optional[List[str]] = None,
        cpu_slots: int = 4,
        gpu_slots: int = 1,
        random_seed: int = 42
    ):
        self.workloads = workloads or ["bfs", "cfd", "hotspot", "lud", "kmeans", "nn", "srad"]
        self.cpu_slots = cpu_slots
        self.gpu_slots = gpu_slots
        self.rng = np.random.default_rng(random_seed)

        self.sim_exec = SimulatedExecutor()
        self.energy_estimator = EnergyEstimator()
        self.ai_scheduler = AIScheduler(
            cpu_slots=cpu_slots,
            gpu_slots=gpu_slots,
            use_simulation=True,
            verbose=False
        )

    def run_all(self) -> pd.DataFrame:
        """Runs all 6 strategies and returns comprehensive metrics DataFrame."""
        print("=" * 80)
        print("RUNNING MULTI-STRATEGY SCHEDULING BENCHMARK COMPARISON")
        print("=" * 80)

        records = []

        # 1. AI-Adaptive Strategy
        ai_res = self._run_ai_adaptive()
        records.append(ai_res)

        # 2. FIFO-CPU Strategy
        fifo_cpu_res = self._run_static_policy("FIFO-CPU", lambda w, idx: "cpu")
        records.append(fifo_cpu_res)

        # 3. FIFO-GPU Strategy
        fifo_gpu_res = self._run_static_policy("FIFO-GPU", lambda w, idx: "gpu")
        records.append(fifo_gpu_res)

        # 4. Round-Robin Strategy
        rr_res = self._run_static_policy("Round-Robin", lambda w, idx: "gpu" if idx % 2 == 0 else "cpu")
        records.append(rr_res)

        # 5. Greedy-Speedup Strategy
        def greedy_picker(w, idx):
            emp = self.sim_exec.empirical_data.get(w, {})
            c_med = emp.get("cpu_median_s", 1.0)
            g_med = emp.get("gpu_median_s", 1.0)
            return "gpu" if g_med < c_med else "cpu"
        greedy_res = self._run_static_policy("Greedy-Speedup", greedy_picker)
        records.append(greedy_res)

        # 6. Random Baseline Strategy
        random_res = self._run_static_policy("Random", lambda w, idx: "gpu" if self.rng.random() > 0.5 else "cpu")
        records.append(random_res)

        df = pd.DataFrame(records)

        # Compute speedup relative to FIFO-CPU baseline
        cpu_makespan = df.loc[df["strategy"] == "FIFO-CPU", "makespan_s"].values[0]
        df["speedup_over_cpu"] = round(cpu_makespan / df["makespan_s"], 2)

        print("\n" + "=" * 90)
        print(f"{'Strategy':<18} | {'Makespan(s)':<12} | {'Energy(J)':<12} | {'GPU Tasks':<10} | {'Speedup':<8}")
        print("-" * 90)
        for _, r in df.iterrows():
            print(f"{r['strategy']:<18} | {r['makespan_s']:<12.3f} | {r['total_energy_j']:<12.2f} | {r['gpu_task_count']:<10} | {r['speedup_over_cpu']:<8.2f}x")
        print("=" * 90)

        return df

    def _run_ai_adaptive(self) -> Dict[str, Any]:
        """Execute AI-Adaptive scheduler."""
        dag = self.ai_scheduler.build_dag(self.workloads)
        assignment = self.ai_scheduler.schedule(dag)

        # Compute slot-simulated makespan
        tasks_meta = {}
        for w in self.workloads:
            emp = self.sim_exec.empirical_data.get(w, {})
            tasks_meta[w] = {
                "cpu_est_s": emp.get("cpu_median_s", 1.0),
                "gpu_est_s": emp.get("gpu_median_s", 0.5),
                "features": {"transfer_bytes_mb": WORKLOAD_CONFIG.get(w, {}).get("transfer_bytes_mb", 10.0)}
            }
        makespan, energy_raw, transfer = _compute_objectives(
            assignment, tasks_meta, self.cpu_slots, self.gpu_slots
        )

        # Compute accurate TDP energy
        total_energy = 0.0
        gpu_count = 0
        for w, dev in assignment.items():
            dur = tasks_meta[w]["gpu_est_s"] if dev == "gpu" else tasks_meta[w]["cpu_est_s"]
            xfer = tasks_meta[w]["features"]["transfer_bytes_mb"] if dev == "gpu" else 0.0
            e = self.energy_estimator.estimate(w, dev, dur, transfer_bytes_mb=xfer)
            total_energy += e["total_energy_joules"]
            if dev == "gpu":
                gpu_count += 1

        return {
            "strategy": "AI-Adaptive",
            "makespan_s": round(float(makespan), 3),
            "total_energy_j": round(float(total_energy), 2),
            "gpu_task_count": gpu_count,
            "cpu_task_count": len(self.workloads) - gpu_count,
            "assignment": assignment,
        }

    def _run_static_policy(self, name: str, policy_fn) -> Dict[str, Any]:
        """Execute static scheduling policy."""
        assignment = {}
        tasks_meta = {}
        for idx, w in enumerate(self.workloads):
            assignment[w] = policy_fn(w, idx)
            emp = self.sim_exec.empirical_data.get(w, {})
            tasks_meta[w] = {
                "cpu_est_s": emp.get("cpu_median_s", 1.0),
                "gpu_est_s": emp.get("gpu_median_s", 0.5),
                "features": {"transfer_bytes_mb": WORKLOAD_CONFIG.get(w, {}).get("transfer_bytes_mb", 10.0)}
            }

        makespan, _, _ = _compute_objectives(
            assignment, tasks_meta, self.cpu_slots, self.gpu_slots
        )

        total_energy = 0.0
        gpu_count = 0
        for w, dev in assignment.items():
            dur = tasks_meta[w]["gpu_est_s"] if dev == "gpu" else tasks_meta[w]["cpu_est_s"]
            xfer = tasks_meta[w]["features"]["transfer_bytes_mb"] if dev == "gpu" else 0.0
            e = self.energy_estimator.estimate(w, dev, dur, transfer_bytes_mb=xfer)
            total_energy += e["total_energy_joules"]
            if dev == "gpu":
                gpu_count += 1

        return {
            "strategy": name,
            "makespan_s": round(float(makespan), 3),
            "total_energy_j": round(float(total_energy), 2),
            "gpu_task_count": gpu_count,
            "cpu_task_count": len(self.workloads) - gpu_count,
            "assignment": assignment,
        }


def plot_strategy_comparison(df: pd.DataFrame, figures_dir: Path = DEFAULT_FIGURES_DIR) -> None:
    """Generates comparison bar charts across the 6 scheduling strategies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    palette = sns.color_palette("Set2", len(df))

    # 1. Makespan Comparison
    sns.barplot(data=df, x="strategy", y="makespan_s", ax=axes[0], palette=palette)
    axes[0].set_title("Workflow Makespan (Lower is Better)")
    axes[0].set_ylabel("Makespan (seconds)")
    axes[0].set_xlabel("Strategy")
    axes[0].tick_params(axis="x", rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.2f}s", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

    # 2. Total Energy Comparison
    sns.barplot(data=df, x="strategy", y="total_energy_j", ax=axes[1], palette=palette)
    axes[1].set_title("Total Energy Consumption (TDP Proxy)")
    axes[1].set_ylabel("Energy (Joules)")
    axes[1].set_xlabel("Strategy")
    axes[1].tick_params(axis="x", rotation=30)
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.0f}J", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

    # 3. Speedup over CPU
    sns.barplot(data=df, x="strategy", y="speedup_over_cpu", ax=axes[2], palette=palette)
    axes[2].set_title("Speedup vs Pure CPU Baseline")
    axes[2].set_ylabel("Speedup Multiplier (x)")
    axes[2].set_xlabel("Strategy")
    axes[2].axhline(1.0, color="red", linestyle="--", alpha=0.7, label="CPU Baseline")
    axes[2].tick_params(axis="x", rotation=30)
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.2f}x", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

    plt.tight_layout()
    chart_path = figures_dir / "scheduler_strategy_benchmarks.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Strategy comparison chart saved to: {chart_path}")


if __name__ == "__main__":
    comparator = StrategyComparison()
    results_df = comparator.run_all()
    plot_strategy_comparison(results_df)
