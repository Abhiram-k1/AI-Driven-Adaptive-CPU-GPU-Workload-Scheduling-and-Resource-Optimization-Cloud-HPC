"""
dashboard/api.py
================
Backend API service for the live Plotly Dash application.
All returned telemetry and metrics are strictly grounded in:
  1. Real-time host telemetry (via psutil ResourceMonitor)
  2. Verified empirical benchmarks (from Layer 2 CSV on Tesla T4)
  3. Trained Random Forest and Gradient Boosting models
  4. Multi-objective scheduler assignments (WeightedSum & NSGA-II)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profiler.resource_monitor import ResourceMonitor
from profiler.energy_estimator import EnergyEstimator
from profiler.workload_configs import WORKLOAD_CONFIGS, WORKLOAD_INPUT_PROFILES
from scheduler.scheduler import AIScheduler
from scheduler.strategy_comparison import StrategyComparison
from scheduler.multi_objective import _compute_objectives
from scheduler.simulation_mode import SimulatedExecutor


class DashboardAPI:
    """
    Data API feeding the live dashboard.
    """

    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.energy_estimator = EnergyEstimator()
        self.sim_executor = SimulatedExecutor()
        self.layer2_path = PROJECT_ROOT / "dataset" / "layer2_aggregated_performance.csv"
        self.expanded_l3_path = PROJECT_ROOT / "dataset" / "layer3_ml_dataset_expanded.csv"
        self.metrics_path = PROJECT_ROOT / "results" / "metrics" / "model_evaluation_metrics.json"
        self.predictions_path = PROJECT_ROOT / "results" / "predictions" / "lowo_predictions.csv"

    def get_live_resources(self) -> Dict[str, Any]:
        """Fetch real-time CPU, RAM, and GPU snapshot."""
        return self.resource_monitor.get_system_snapshot()

    def get_benchmark_summary(self) -> List[Dict[str, Any]]:
        """Fetch empirical benchmark rows from Layer 2 dataset."""
        if not self.layer2_path.exists():
            return []
        df = pd.read_csv(self.layer2_path)
        records = []
        for _, r in df.iterrows():
            c_med = float(r["cpu_median_ms"]) if pd.notna(r["cpu_median_ms"]) else None
            g_med = float(r["gpu_total_path_ms"]) if pd.notna(r["gpu_total_path_ms"]) else None
            speedup = round(c_med / g_med, 2) if (c_med and g_med and g_med > 0) else None
            records.append({
                "workload": r["workload"],
                "input_file": r["input_file"],
                "input_size_mib": round(float(r["input_size_mib"]), 2),
                "cpu_median_ms": round(c_med, 2) if c_med else None,
                "gpu_median_ms": round(g_med, 2) if g_med else None,
                "speedup": speedup,
                "preferred_device": r["preferred_device"],
                "timing_status": r["gpu_total_path_time_status"],
            })
        return records

    def get_ml_metrics(self) -> Dict[str, Any]:
        """Fetch LOWO evaluation metrics."""
        if not self.metrics_path.exists():
            return {}
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_lowo_predictions(self) -> List[Dict[str, Any]]:
        """Fetch per-sample LOWO holdout predictions."""
        if not self.predictions_path.exists():
            return []
        df = pd.read_csv(self.predictions_path)
        return df.to_dict(orient="records")

    def run_scheduling_pipeline(
        self,
        selected_workloads: List[str],
        mode: str = "balanced",
        cpu_slots: int = 4,
        gpu_slots: int = 1
    ) -> Dict[str, Any]:
        """Runs the live AI Scheduler on user-selected workload tasks."""
        scheduler = AIScheduler(
            cpu_slots=cpu_slots,
            gpu_slots=gpu_slots,
            mode=mode,
            use_simulation=True,
            verbose=False
        )
        dag = scheduler.build_dag(selected_workloads)
        assignment = scheduler.schedule(dag)
        results = scheduler.execute(dag, assignment)

        # Calculate summary metrics
        total_makespan = max((r["mean_s"] for r in results), default=0.0)
        total_energy = sum((r.get("energy_joules", 0.0) for r in results))
        match_count = sum(1 for r in results if r.get("correct_assignment"))

        return {
            "assignment": assignment,
            "tasks_executed": len(results),
            "total_energy_j": round(total_energy, 2),
            "workflow_makespan_s": round(total_makespan, 3),
            "ml_match_accuracy": round((match_count / max(1, len(results))) * 100.0, 1),
            "results": results,
            "bayesian_weights": scheduler.weight_ctrl.mean_weights(),
        }

    def get_strategy_comparison(self) -> List[Dict[str, Any]]:
        """Fetch strategy comparison across the 6 scheduling baselines."""
        comp = StrategyComparison()
        df = comp.run_all()
        return df.to_dict(orient="records")

    def get_workload_catalog(self) -> List[Dict[str, Any]]:
        """Fetch all 19 Rodinia workloads with domains and input profiles."""
        catalog = []
        for wl, cfg in WORKLOAD_CONFIGS.items():
            profiles = WORKLOAD_INPUT_PROFILES.get(wl, {})
            catalog.append({
                "workload": wl,
                "domain": cfg.get("domain", "general"),
                "workload_type": cfg.get("workload_type", "compute"),
                "status": cfg.get("status", "configured"),
                "input_profiles": list(profiles.keys()),
            })
        return catalog

