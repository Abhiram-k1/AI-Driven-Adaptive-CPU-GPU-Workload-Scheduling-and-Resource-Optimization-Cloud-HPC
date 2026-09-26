"""
scheduler.py
============
Main AI Scheduler orchestrator.

Pipeline:
  1. Accept a list of workload tasks (with features)
  2. Build dependency DAG
  3. Run ML predictor on each task
  4. Sample adaptive weights from Bayesian controller
  5. Run multi-objective optimizer
  6. Execute tasks respecting DAG order (CPU/GPU in parallel)
  7. Collect actual timings, update weight controller
  8. Report results

Usage:
    python scheduler/scheduler.py --workloads bfs cfd hotspot lud kmeans
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from profiler.rodinia_configs import WORKLOAD_CONFIG, ALL_WORKLOADS
from ml.predict import WorkloadPredictor
from scheduler.dag_builder import WorkloadDAG
from scheduler.multi_objective import WeightedSumScheduler, _compute_objectives
from scheduler.adaptive_weights import AdaptiveWeightController
from executor.cpu_executor import CPUExecutor
from executor.gpu_executor import GPUExecutor


class AIScheduler:
    """
    AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduler.
    """

    def __init__(
        self,
        n_runs: int = 6,
        cpu_slots: int = 4,
        gpu_slots: int = 1,
        mode: str = "balanced",    # 'speed', 'energy', 'balanced'
        verbose: bool = True,
    ):
        self.n_runs    = n_runs
        self.cpu_slots = cpu_slots
        self.gpu_slots = gpu_slots
        self.verbose   = verbose

        self.predictor = WorkloadPredictor()
        self.weight_ctrl = AdaptiveWeightController()
        self.weight_ctrl.set_mode(mode)

        self.cpu_exec = CPUExecutor()
        self.gpu_exec = GPUExecutor()

        self.results   = []     # list of result dicts
        self.completed = set()  # task_ids that finished

    # ── Setup ─────────────────────────────────────────────────────────────

    def build_dag(
        self,
        workloads: list[str],
        dependencies: list[tuple[str, str]] = None,
    ) -> WorkloadDAG:
        """Build and return the workload DAG."""
        dag = WorkloadDAG(predictor=self.predictor)
        for wl in workloads:
            cfg = WORKLOAD_CONFIG[wl]
            dag.add_task(
                task_id=wl,
                workload=wl,
                features={
                    "input_size_mb":      cfg["input_size_mb"],
                    "compute_intensity":  cfg["compute_intensity"],
                    "parallelism_degree": cfg["parallelism_degree"],
                    "data_dep":           int(cfg["data_dep"]),
                    "transfer_bytes_mb":  cfg["transfer_bytes_mb"],
                },
                n_runs=self.n_runs,
            )
        if dependencies:
            for up, down in dependencies:
                dag.add_dependency(up, down)
        dag.compute_priorities()
        return dag

    # ── Scheduling ─────────────────────────────────────────────────────────

    def _make_task_dict(self, dag: WorkloadDAG) -> dict:
        """Build the task dict expected by the optimizer."""
        tasks = {}
        for node in dag.G.nodes:
            d = dag.G.nodes[node]
            tasks[node] = {
                "cpu_est_s": d["cpu_est_s"],
                "gpu_est_s": d["gpu_est_s"],
                "features":  d["features"],
                "priority":  d["priority"],
                "critical":  d["critical"],
            }
        return tasks

    def schedule(self, dag: WorkloadDAG) -> dict[str, str]:
        """Run multi-objective optimizer, return task→device assignment."""
        tasks = self._make_task_dict(dag)
        weights = self.weight_ctrl.sample_weights()
        if self.verbose:
            print(f"\n[Scheduler] Weights: makespan={weights[0]:.3f}  "
                  f"energy={weights[1]:.3f}  transfer={weights[2]:.3f}")

        optimizer = WeightedSumScheduler(tasks, self.cpu_slots, self.gpu_slots)
        assignment = optimizer.solve(weights)

        if self.verbose:
            print("[Scheduler] Assignment:")
            for tid, dev in assignment.items():
                d = dag.G.nodes[tid]
                crit = "* " if d["critical"] else "  "
                print(f"  {crit}{tid:20s} -> {dev.upper():3s}  "
                      f"(est {d['gpu_est_s'] if dev=='gpu' else d['cpu_est_s']:.3f}s)")
        return assignment

    # ── Execution ──────────────────────────────────────────────────────────

    def execute(
        self,
        dag: WorkloadDAG,
        assignment: dict[str, str],
    ) -> list[dict]:
        """
        Execute tasks in DAG order, respecting dependencies.
        Returns list of result dicts.
        """
        all_results = []
        self.completed = set()

        total_start = time.perf_counter()

        while len(self.completed) < dag.G.number_of_nodes():
            ready = dag.get_ready_tasks(self.completed)
            if not ready:
                break

            if self.verbose:
                print(f"\n[Executor] Ready: {ready}")

            for task_id in ready:
                cfg     = WORKLOAD_CONFIG[task_id]
                device  = assignment[task_id]

                if self.verbose:
                    print(f"  Running {task_id} on {device.upper()}...")

                if device == "gpu":
                    result = self.gpu_exec.run(task_id, cfg, self.n_runs)
                else:
                    result = self.cpu_exec.run(task_id, cfg, self.n_runs)

                result["device_assigned"] = device
                result["predicted_device"] = dag.G.nodes[task_id]["device"]
                result["cpu_est_s"]  = dag.G.nodes[task_id]["cpu_est_s"]
                result["gpu_est_s"]  = dag.G.nodes[task_id]["gpu_est_s"]
                result["correct_assignment"] = (
                    result["device_assigned"] == result["predicted_device"]
                )

                all_results.append(result)
                self.completed.add(task_id)

                # Update weight controller with actual costs
                energy_cost  = result["mean_s"] * (
                    250 if device == "gpu" else 65
                )
                transfer_s   = cfg["transfer_bytes_mb"] / 12000.0 if device == "gpu" else 0.0
                self.weight_ctrl.update(
                    makespan_cost=result["mean_s"],
                    energy_cost=energy_cost,
                    transfer_cost=transfer_s,
                )

        total_makespan = time.perf_counter() - total_start
        if self.verbose:
            print(f"\n[Scheduler] Total makespan: {total_makespan:.2f}s")

        return all_results

    # ── Report ─────────────────────────────────────────────────────────────

    def report(self, results: list[dict]) -> None:
        print("\n" + "=" * 70)
        print("  AI SCHEDULER RESULTS")
        print("=" * 70)
        print(f"  {'Workload':20s} {'Device':5s} {'Mean(s)':>8s} {'Speedup':>8s} {'Correct':>8s}")
        print("-" * 70)
        for r in results:
            print(
                f"  {r['workload']:20s} {r['device']:5s} "
                f"{r['mean_s']:8.3f} "
                f"{r.get('speedup', 1.0):8.2f}x "
                f"{'✓' if r.get('correct_assignment') else '✗':>8s}"
            )
        print("=" * 70)
        print(f"  Weight controller: {self.weight_ctrl.summary()}")


def main():
    parser = argparse.ArgumentParser(description="AI CPU-GPU Workload Scheduler")
    parser.add_argument("--workloads", nargs="+", default=["bfs", "cfd", "hotspot", "lud"])
    parser.add_argument("--n_runs",    type=int,   default=6)
    parser.add_argument("--mode",      default="balanced", choices=["speed", "energy", "balanced"])
    parser.add_argument("--cpu_slots", type=int,   default=4)
    parser.add_argument("--gpu_slots", type=int,   default=1)
    args = parser.parse_args()

    scheduler = AIScheduler(
        n_runs=args.n_runs,
        cpu_slots=args.cpu_slots,
        gpu_slots=args.gpu_slots,
        mode=args.mode,
    )

    # Example pipeline dependencies
    deps = [
        ("bfs", "cfd"),          # CFD depends on BFS
        ("hotspot", "srad"),     # SRAD depends on Hotspot (if both present)
    ]
    deps = [(u, v) for u, v in deps
            if u in args.workloads and v in args.workloads]

    dag        = scheduler.build_dag(args.workloads, dependencies=deps)
    dag.summary()

    assignment = scheduler.schedule(dag)
    results    = scheduler.execute(dag, assignment)
    scheduler.report(results)


if __name__ == "__main__":
    main()
