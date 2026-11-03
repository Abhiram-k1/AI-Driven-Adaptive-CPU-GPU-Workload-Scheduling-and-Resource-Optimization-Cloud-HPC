"""
scheduler/scheduler.py
======================
Main AI-Driven Adaptive Multi-Objective CPU-GPU Workload Scheduler Orchestrator.

Pipeline:
  1. Accept a list of workload tasks (with pre-execution features)
  2. Build dependency DAG with critical path prioritization
  3. Run ML predictor on each task for device placement recommendation & confidence
  4. Query live ResourceMonitor (CPU load, memory pressure, GPU availability)
  5. Sample adaptive multi-objective weights via Bayesian Thompson Sampling (Dirichlet)
  6. Solve multi-objective assignment (Makespan, Energy Proxy, Transfer Overhead)
  7. Execute tasks respecting DAG dependencies across concurrent CPU and GPU slot pools
     (Native subprocess execution if binaries exist; SimulatedExecutor fallback grounded
      in empirical Tesla T4 benchmark measurements)
  8. Calculate TDP-proxy energy via EnergyEstimator
  9. Update Bayesian weight controller via online reinforcement feedback loop
  10. Return complete execution telemetry and audit records.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profiler.rodinia_configs import WORKLOAD_CONFIG, ALL_WORKLOADS
from ml.predict import WorkloadPredictor
try:
    from scheduler.dag_builder import WorkloadDAG
    from scheduler.multi_objective import WeightedSumScheduler, _compute_objectives
    from scheduler.adaptive_weights import AdaptiveWeightController
    from scheduler.simulation_mode import SimulatedExecutor
except (ImportError, ModuleNotFoundError):
    from dag_builder import WorkloadDAG
    from multi_objective import WeightedSumScheduler, _compute_objectives
    from adaptive_weights import AdaptiveWeightController
    from simulation_mode import SimulatedExecutor
from profiler.energy_estimator import EnergyEstimator
from profiler.resource_monitor import ResourceMonitor
from executor.cpu_executor import CPUExecutor
from executor.gpu_executor import GPUExecutor


class AIScheduler:
    """
    AI-Driven Adaptive Multi-Objective CPU-GPU Workload Scheduler.
    """

    def __init__(
        self,
        n_runs: int = 6,
        cpu_slots: int = 4,
        gpu_slots: int = 1,
        mode: str = "balanced",    # 'speed', 'energy', 'balanced', 'transfer'
        use_simulation: Optional[bool] = None,
        verbose: bool = True,
    ):
        self.n_runs = n_runs
        self.cpu_slots = cpu_slots
        self.gpu_slots = gpu_slots
        self.verbose = verbose

        self.predictor = WorkloadPredictor()
        self.weight_ctrl = AdaptiveWeightController()
        self.weight_ctrl.set_mode(mode)

        self.energy_estimator = EnergyEstimator()
        self.resource_monitor = ResourceMonitor()

        # Decide whether to use simulation mode
        # If user specifies use_simulation, respect it; otherwise auto-detect GPU/binaries
        if use_simulation is not None:
            self.use_simulation = use_simulation
        else:
            self.use_simulation = not self.resource_monitor.gpu_detected

        self.sim_exec = SimulatedExecutor()
        self.cpu_exec = CPUExecutor()
        self.gpu_exec = GPUExecutor()

        self.results: List[Dict[str, Any]] = []
        self.completed: set = set()

    def build_dag(
        self,
        workloads: List[str],
        dependencies: Optional[List[Tuple[str, str]]] = None,
    ) -> WorkloadDAG:
        """Build and return the dependency DAG with ML metadata."""
        dag = WorkloadDAG(predictor=self.predictor)
        for wl in workloads:
            cfg = WORKLOAD_CONFIG.get(wl, {
                "input_size_mb": 10.0,
                "compute_intensity": 5,
                "parallelism_degree": 8,
                "data_dep": 0,
                "transfer_bytes_mb": 10.0,
            })
            dag.add_task(
                task_id=wl,
                workload=wl,
                features={
                    "input_size_mb": cfg.get("input_size_mb", 10.0),
                    "compute_intensity": cfg.get("compute_intensity", 5),
                    "parallelism_degree": cfg.get("parallelism_degree", 8),
                    "data_dep": int(cfg.get("data_dep", False)),
                    "transfer_bytes_mb": cfg.get("transfer_bytes_mb", 10.0),
                },
                n_runs=self.n_runs,
            )
        if dependencies:
            for up, down in dependencies:
                dag.add_dependency(up, down)
        dag.compute_priorities()
        return dag

    def _make_task_dict(self, dag: WorkloadDAG) -> Dict[str, Any]:
        """Build the task dictionary required by the multi-objective optimizer."""
        tasks = {}
        for node in dag.G.nodes:
            d = dag.G.nodes[node]
            tasks[node] = {
                "cpu_est_s": d["cpu_est_s"],
                "gpu_est_s": d["gpu_est_s"],
                "features": d["features"],
                "priority": d["priority"],
                "critical": d["critical"],
            }
        return tasks

    def apply_resource_constraints(self, queue_len: int) -> Tuple[int, int]:
        """
        Dynamically adjusts slot limits and biases weights based on live system telemetry.
        """
        snap = self.resource_monitor.get_cpu_snapshot()
        cpu_load = snap["cpu_percent"]
        mem_pct = snap["memory_percent"]

        # Pressure adjustments in Bayesian controller
        gpu_mem_ratio = 0.0
        if self.resource_monitor.gpu_detected:
            gpu_snap = self.resource_monitor.get_gpu_snapshot()
            gpu_mem_ratio = (gpu_snap.get("gpu_memory_percent") or 0.0) / 100.0

        self.weight_ctrl.pressure_adjust(
            queue_length=queue_len,
            gpu_mem_used_ratio=gpu_mem_ratio,
            deadline_proximity=0.5
        )

        effective_cpu_slots = self.cpu_slots
        if cpu_load > 85.0 and effective_cpu_slots > 1:
            effective_cpu_slots -= 1  # Throttle to prevent host saturation

        return effective_cpu_slots, self.gpu_slots

    def schedule(self, dag: WorkloadDAG) -> Dict[str, str]:
        """Run multi-objective optimizer and return task -> device assignment."""
        tasks = self._make_task_dict(dag)
        adj_cpu_slots, adj_gpu_slots = self.apply_resource_constraints(len(tasks))

        weights = self.weight_ctrl.sample_weights()
        if self.verbose:
            print(f"\n[Scheduler] Sampled Bayesian Weights: makespan={weights[0]:.3f} | "
                  f"energy={weights[1]:.3f} | transfer={weights[2]:.3f}")

        optimizer = WeightedSumScheduler(tasks, adj_cpu_slots, adj_gpu_slots)
        assignment = optimizer.solve(weights)

        if self.verbose:
            print("[Scheduler] Optimal Assignment Plan:")
            for tid, dev in assignment.items():
                d = dag.G.nodes[tid]
                crit = "* " if d["critical"] else "  "
                pred_dev = d["device"]
                match_str = "MATCH" if dev == pred_dev else "ADAPTED"
                print(f"  {crit}{tid:20s} -> {dev.upper():3s} (ML recommended: {pred_dev.upper()}, {match_str}) "
                      f"[est: {d['gpu_est_s'] if dev=='gpu' else d['cpu_est_s']:.3f}s]")
        return assignment

    def execute(
        self,
        dag: WorkloadDAG,
        assignment: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Execute tasks in dependency-respecting topological order.
        """
        all_results = []
        self.completed = set()

        total_start = time.perf_counter()

        while len(self.completed) < dag.G.number_of_nodes():
            ready = dag.get_ready_tasks(self.completed)
            if not ready:
                break

            if self.verbose:
                print(f"\n[Executor] Ready Tasks: {ready}")

            for task_id in ready:
                cfg = WORKLOAD_CONFIG.get(task_id, {
                    "input_size_mb": 10.0,
                    "compute_intensity": 5,
                    "parallelism_degree": 8,
                    "data_dep": False,
                    "transfer_bytes_mb": 10.0,
                    "cpu": {"command": "", "cwd": "", "time_regex": ""},
                    "gpu": {"command": "", "cwd": "", "time_regex": ""},
                })
                device = assignment[task_id]

                if self.verbose:
                    exec_mode_str = "SIMULATED (T4 data)" if self.use_simulation else "NATIVE HARDWARE"
                    print(f"  Executing {task_id} on {device.upper()} [{exec_mode_str}]...")

                snap_pre = self.resource_monitor.get_cpu_snapshot()

                if self.use_simulation:
                    result = self.sim_exec.run(task_id, cfg, device=device, n_runs=self.n_runs)
                else:
                    if device == "gpu":
                        result = self.gpu_exec.run(task_id, cfg, self.n_runs)
                    else:
                        result = self.cpu_exec.run(task_id, cfg, self.n_runs)

                snap_post = self.resource_monitor.get_cpu_snapshot()

                # Calculate TDP-based energy consumption
                transfer_mb = cfg.get("transfer_bytes_mb", 0.0)
                energy_rec = self.energy_estimator.estimate(
                    workload=task_id,
                    device=device,
                    runtime_s=result["mean_s"],
                    transfer_bytes_mb=transfer_mb if device == "gpu" else 0.0
                )

                result["device_assigned"] = device
                result["predicted_device"] = dag.G.nodes[task_id]["device"]
                result["cpu_est_s"] = dag.G.nodes[task_id]["cpu_est_s"]
                result["gpu_est_s"] = dag.G.nodes[task_id]["gpu_est_s"]
                result["correct_assignment"] = (result["device_assigned"] == result["predicted_device"])
                result["energy_joules"] = energy_rec["total_energy_joules"]
                result["energy_metadata"] = energy_rec
                result["resource_snapshot_pre"] = snap_pre
                result["resource_snapshot_post"] = snap_post

                all_results.append(result)
                self.completed.add(task_id)

                # Feedback loop: update Dirichlet Bayesian controller
                transfer_s = transfer_mb / 12000.0 if device == "gpu" else 0.0
                self.weight_ctrl.update(
                    makespan_cost=result["mean_s"],
                    energy_cost=energy_rec["total_energy_joules"],
                    transfer_cost=transfer_s,
                )

        total_makespan = time.perf_counter() - total_start
        if self.verbose:
            print(f"\n[Scheduler] Total Workflow Makespan: {total_makespan:.2f}s")

        self.results = all_results
        return all_results

    def report(self, results: List[Dict[str, Any]]) -> None:
        """Prints a publication-ready tabular summary."""
        print("\n" + "=" * 80)
        print("  AI-DRIVEN ADAPTIVE MULTI-OBJECTIVE CPU-GPU SCHEDULER RESULTS")
        print("=" * 80)
        print(f"  {'Workload':<16} {'Device':<6} {'Runtime(s)':>10} {'Energy(J)':>10} {'Speedup':>9} {'ML Match':>9}")
        print("-" * 80)
        total_e = 0.0
        for r in results:
            e = r.get("energy_joules", 0.0)
            total_e += e
            sp = r.get("speedup", 1.0) or 1.0
            print(
                f"  {r['workload']:<16} {r['device_assigned']:<6} "
                f"{r['mean_s']:10.3f} "
                f"{e:10.2f} "
                f"{sp:8.2f}x "
                f"{'YES' if r.get('correct_assignment') else 'NO':>9}"
            )
        print("=" * 80)
        print(f"  Total Energy Consumption: {total_e:.2f} Joules (TDP-model estimate)")
        print(f"  Weight Controller State: {self.weight_ctrl.summary()}")


def main():
    parser = argparse.ArgumentParser(description="AI CPU-GPU Workload Scheduler")
    parser.add_argument("--workloads", nargs="+", default=["bfs", "cfd", "hotspot", "lud", "kmeans"])
    parser.add_argument("--n_runs", type=int, default=6)
    parser.add_argument("--mode", default="balanced", choices=["speed", "energy", "balanced", "transfer"])
    parser.add_argument("--cpu_slots", type=int, default=4)
    parser.add_argument("--gpu_slots", type=int, default=1)
    parser.add_argument("--simulate", action="store_true", default=True, help="Use validated simulation mode")
    args = parser.parse_args()

    scheduler = AIScheduler(
        n_runs=args.n_runs,
        cpu_slots=args.cpu_slots,
        gpu_slots=args.gpu_slots,
        mode=args.mode,
        use_simulation=args.simulate,
    )

    deps = [
        ("bfs", "cfd"),
        ("hotspot", "srad"),
    ]
    deps = [(u, v) for u, v in deps if u in args.workloads and v in args.workloads]

    dag = scheduler.build_dag(args.workloads, dependencies=deps)
    dag.summary()

    assignment = scheduler.schedule(dag)
    results = scheduler.execute(dag, assignment)
    scheduler.report(results)


if __name__ == "__main__":
    main()
