"""
multi_objective.py
==================
Multi-objective optimizer for workload-to-device assignment.

Implements two strategies:
  1. Weighted Sum Scalarisation (fast, single solution)
  2. NSGA-II via DEAP (slower, returns full Pareto front)

Objectives (all to be minimised):
  f1 = Makespan        : max completion time across all tasks
  f2 = Energy proxy    : sum(runtime * device_power_factor)
  f3 = Transfer cost   : sum(transfer_time for GPU-assigned tasks)

Usage:
    from scheduler.multi_objective import WeightedSumScheduler, NSGAScheduler
    from scheduler.adaptive_weights import AdaptiveWeightController

    awc = AdaptiveWeightController()
    w = awc.sample_weights()

    scheduler = WeightedSumScheduler(tasks, cpu_count=8, gpu_count=1)
    assignment = scheduler.solve(w)
    # → {"bfs": "cpu", "lud": "gpu", ...}
"""

import sys
import random
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Device power factors (Watts, approximate) ────────────────────────────────
CPU_POWER_W = 65.0    # typical server CPU TDP
GPU_POWER_W = 250.0   # typical NVIDIA V100 TDP

# PCIe bandwidth (MB/s)
PCIE_MB_S = 12_000.0


def _compute_objectives(
    assignment: dict[str, str],    # task_id -> "cpu" | "gpu"
    tasks: dict,
    cpu_slots: int = 8,
    gpu_slots: int = 1,
) -> tuple[float, float, float]:
    """
    Compute (makespan, energy, transfer_cost) for a given assignment.
    Uses a simplified slot-based parallel execution model.
    """
    cpu_queue, gpu_queue = [], []

    for tid, device in assignment.items():
        t = tasks[tid]
        if device == "cpu":
            cpu_queue.append(t["cpu_est_s"])
        else:
            gpu_queue.append(t["gpu_est_s"])

    # Makespan: simulate parallel execution across slots
    def _makespan(queue, slots):
        if not queue:
            return 0.0
        queue_s = sorted(queue, reverse=True)
        slot_end = [0.0] * slots
        for dur in queue_s:
            i = np.argmin(slot_end)
            slot_end[i] += dur
        return max(slot_end)

    makespan = max(_makespan(cpu_queue, cpu_slots), _makespan(gpu_queue, gpu_slots))

    # Energy: runtime * power
    energy = sum(
        t["cpu_est_s"] * CPU_POWER_W if assignment[tid] == "cpu"
        else t["gpu_est_s"] * GPU_POWER_W
        for tid, t in tasks.items()
    )

    # Transfer cost: PCIe time for GPU-assigned tasks
    transfer_cost = sum(
        t["features"].get("transfer_bytes_mb", 0) / PCIE_MB_S
        for tid, t in tasks.items()
        if assignment[tid] == "gpu"
    )

    return makespan, energy, transfer_cost


class WeightedSumScheduler:
    """
    Fast weighted-sum multi-objective optimizer.
    Explores all 2^N device assignments (feasible for N <= ~20 tasks).
    Falls back to greedy for larger task sets.
    """

    def __init__(self, tasks: dict, cpu_slots: int = 8, gpu_slots: int = 1):
        """
        Parameters
        ----------
        tasks : dict  {task_id: {cpu_est_s, gpu_est_s, features, ...}}
        """
        self.tasks = tasks
        self.cpu_slots = cpu_slots
        self.gpu_slots = gpu_slots
        self.task_ids = list(tasks.keys())

    def solve(
        self,
        weights: tuple[float, float, float],
        max_enumerate: int = 20,
    ) -> dict[str, str]:
        """
        Find device assignment minimising weighted sum of objectives.

        For <= max_enumerate tasks: exhaustive search.
        For larger sets: greedy per-task assignment.
        """
        w1, w2, w3 = weights

        def score(assignment):
            f1, f2, f3 = _compute_objectives(
                assignment, self.tasks, self.cpu_slots, self.gpu_slots
            )
            # Normalise roughly (energy is in Watt-seconds, much larger)
            return w1 * f1 + w2 * (f2 / 10000.0) + w3 * f3

        n = len(self.task_ids)

        if n <= max_enumerate:
            # Exhaustive search over all 2^n assignments
            best_score = float("inf")
            best_asgn  = {}
            for mask in range(1 << n):
                asgn = {
                    self.task_ids[i]: ("gpu" if (mask >> i) & 1 else "cpu")
                    for i in range(n)
                }
                s = score(asgn)
                if s < best_score:
                    best_score = s
                    best_asgn = asgn
            return best_asgn

        else:
            # Greedy: assign each task to the device with lower weighted cost
            asgn = {}
            for tid in self.task_ids:
                t = self.tasks[tid]
                # Per-task marginal cost
                cpu_cost = w1 * t["cpu_est_s"] + w2 * (t["cpu_est_s"] * CPU_POWER_W / 10000.0)
                gpu_transfer = t["features"].get("transfer_bytes_mb", 0) / PCIE_MB_S
                gpu_cost = (
                    w1 * t["gpu_est_s"]
                    + w2 * (t["gpu_est_s"] * GPU_POWER_W / 10000.0)
                    + w3 * gpu_transfer
                )
                asgn[tid] = "gpu" if gpu_cost < cpu_cost else "cpu"
            return asgn


class NSGAScheduler:
    """
    NSGA-II multi-objective scheduler using DEAP.
    Returns a Pareto front of non-dominated device assignments.
    """

    def __init__(self, tasks: dict, cpu_slots: int = 8, gpu_slots: int = 1):
        self.tasks = tasks
        self.cpu_slots = cpu_slots
        self.gpu_slots = gpu_slots
        self.task_ids = list(tasks.keys())
        self._import_deap()

    def _import_deap(self):
        try:
            from deap import base, creator, tools, algorithms
            self._deap = (base, creator, tools, algorithms)
        except ImportError:
            self._deap = None
            print("[NSGAScheduler] WARNING: deap not installed. Install with: pip install deap")

    def _individual_to_assignment(self, individual: list) -> dict:
        return {
            tid: ("gpu" if bit else "cpu")
            for tid, bit in zip(self.task_ids, individual)
        }

    def _evaluate(self, individual: list) -> tuple:
        asgn = self._individual_to_assignment(individual)
        f1, f2, f3 = _compute_objectives(asgn, self.tasks, self.cpu_slots, self.gpu_slots)
        return (f1, f2 / 10000.0, f3)   # normalise energy

    def solve(
        self,
        pop_size: int = 50,
        n_gen: int = 100,
        cx_prob: float = 0.7,
        mut_prob: float = 0.2,
    ) -> list[dict]:
        """
        Run NSGA-II. Returns list of Pareto-optimal assignments.
        """
        if self._deap is None:
            print("[NSGAScheduler] Falling back to WeightedSumScheduler")
            ws = WeightedSumScheduler(self.tasks, self.cpu_slots, self.gpu_slots)
            return [ws.solve((0.33, 0.33, 0.34))]

        base, creator, tools, algorithms = self._deap
        n = len(self.task_ids)

        # Reset creator (avoid re-definition errors)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate)
        toolbox.register("mate",    tools.cxUniform, indpb=0.5)
        toolbox.register("mutate",  tools.mutFlipBit, indpb=1.0 / n)
        toolbox.register("select",  tools.selNSGA2)

        pop = toolbox.population(n=pop_size)
        pop, logbook = algorithms.eaMuPlusLambda(
            pop, toolbox,
            mu=pop_size, lambda_=pop_size,
            cxpb=cx_prob, mutpb=mut_prob,
            ngen=n_gen, verbose=False,
        )

        # Extract Pareto front
        pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        return [self._individual_to_assignment(ind) for ind in pareto]


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Fake task data
    test_tasks = {
        "lud":       {"cpu_est_s": 2.1, "gpu_est_s": 0.3, "features": {"transfer_bytes_mb": 32}},
        "bfs":       {"cpu_est_s": 0.8, "gpu_est_s": 0.9, "features": {"transfer_bytes_mb": 28}},
        "hotspot":   {"cpu_est_s": 1.5, "gpu_est_s": 0.4, "features": {"transfer_bytes_mb": 2}},
        "kmeans":    {"cpu_est_s": 3.2, "gpu_est_s": 1.1, "features": {"transfer_bytes_mb": 73}},
        "backprop":  {"cpu_est_s": 1.0, "gpu_est_s": 0.5, "features": {"transfer_bytes_mb": 8}},
    }

    ws = WeightedSumScheduler(test_tasks)
    print("=== Weighted Sum Scheduler ===")
    for w in [(0.8, 0.1, 0.1), (0.33, 0.33, 0.34), (0.1, 0.1, 0.8)]:
        asgn = ws.solve(w)
        f1, f2, f3 = _compute_objectives(asgn, test_tasks)
        print(f"  weights={w}  ->  makespan={f1:.2f}s  energy={f2:.0f}J  transfer={f3:.3f}s")
        print(f"  assignment: {asgn}")
