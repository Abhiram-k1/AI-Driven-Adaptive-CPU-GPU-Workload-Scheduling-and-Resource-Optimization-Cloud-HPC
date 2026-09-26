"""
dag_builder.py
==============
Constructs a Directed Acyclic Graph (DAG) of workload dependencies.

In a real cloud HPC pipeline, tasks may depend on each other's outputs.
We model two types of dependencies:

  1. Explicit dependencies — user declares "task B needs output of task A"
  2. Data-flow dependencies — detected from config: tasks sharing data files

Each node carries:
  - workload name
  - task features (from profiler)
  - predicted device & runtime (from ML predictor)
  - priority (critical-path length)

Usage:
    from scheduler.dag_builder import WorkloadDAG
    dag = WorkloadDAG()
    dag.add_task("lud",       features={...})
    dag.add_task("hotspot",   features={...})
    dag.add_dependency("lud", "hotspot")   # hotspot depends on lud
    dag.compute_priorities()
    order = dag.topological_order()
"""

import networkx as nx
from typing import Optional
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.predict import WorkloadPredictor


class WorkloadDAG:
    """
    DAG where each node is a workload task.
    Edges represent execution dependencies (u → v means u must finish before v).
    """

    def __init__(self, predictor: Optional[WorkloadPredictor] = None):
        self.G = nx.DiGraph()
        self.predictor = predictor or WorkloadPredictor()

    # ── Graph construction ────────────────────────────────────────────────

    def add_task(
        self,
        task_id: str,
        workload: str,
        features: dict,
        n_runs: int = 6,
    ) -> None:
        """Add a task node with ML-predicted scheduling metadata."""
        prediction = self.predictor.predict(features)
        self.G.add_node(task_id, **{
            "workload":    workload,
            "features":    features,
            "device":      prediction["device"],
            "confidence":  prediction["confidence"],
            "cpu_est_s":   prediction["cpu_est_s"],
            "gpu_est_s":   prediction["gpu_est_s"],
            "speedup":     prediction["speedup_pred"],
            "transfer_overhead": prediction["transfer_overhead_ratio"],
            "n_runs":      n_runs,
            "priority":    0,           # set by compute_priorities()
            "critical":    False,       # set by compute_priorities()
        })

    def add_dependency(self, upstream: str, downstream: str) -> None:
        """downstream must wait for upstream to complete."""
        assert upstream in self.G.nodes, f"Unknown task: {upstream}"
        assert downstream in self.G.nodes, f"Unknown task: {downstream}"
        assert not nx.has_path(self.G, downstream, upstream), (
            f"Adding {upstream}→{downstream} would create a cycle!"
        )
        self.G.add_edge(upstream, downstream)

    # ── Critical path analysis ────────────────────────────────────────────

    def _node_duration(self, node: str) -> float:
        """Expected duration of a node based on its predicted device."""
        data = self.G.nodes[node]
        return data["gpu_est_s"] if data["device"] == "gpu" else data["cpu_est_s"]

    def compute_priorities(self) -> None:
        """
        Compute bottom-up critical path length for each node.
        Priority = sum of durations on the longest remaining path.
        Nodes on the critical path are marked critical=True.
        """
        # Reverse topological order (bottom-up)
        rev_topo = list(reversed(list(nx.topological_sort(self.G))))
        cp = {}   # critical path length from this node to end

        for node in rev_topo:
            my_dur = self._node_duration(node)
            successors = list(self.G.successors(node))
            if successors:
                cp[node] = my_dur + max(cp[s] for s in successors)
            else:
                cp[node] = my_dur

        # Assign priorities and mark critical path
        max_cp = max(cp.values()) if cp else 1.0
        critical_path = set()
        for node in self.G.nodes:
            self.G.nodes[node]["priority"] = cp[node]

        # Trace the critical path (greedy from sources)
        for source in [n for n in self.G.nodes if self.G.in_degree(n) == 0]:
            path = [source]
            cur = source
            while True:
                succ = list(self.G.successors(cur))
                if not succ:
                    break
                cur = max(succ, key=lambda s: cp[s])
                path.append(cur)
            for n in path:
                critical_path.add(n)

        for node in self.G.nodes:
            self.G.nodes[node]["critical"] = node in critical_path

    # ── Traversal helpers ─────────────────────────────────────────────────

    def topological_order(self) -> list[str]:
        """Return tasks in dependency-respecting execution order."""
        return list(nx.topological_sort(self.G))

    def priority_order(self) -> list[str]:
        """Return tasks sorted by priority (critical path length), descending."""
        return sorted(
            self.G.nodes,
            key=lambda n: self.G.nodes[n]["priority"],
            reverse=True
        )

    def get_ready_tasks(self, completed: set[str]) -> list[str]:
        """Return tasks whose all predecessors are completed."""
        ready = []
        for node in self.G.nodes:
            if node not in completed:
                preds = set(self.G.predecessors(node))
                if preds.issubset(completed):
                    ready.append(node)
        # Sort by priority (highest first)
        ready.sort(key=lambda n: self.G.nodes[n]["priority"], reverse=True)
        return ready

    def summary(self) -> None:
        """Print a summary of the DAG."""
        print(f"\n{'='*60}")
        print(f"  Workload DAG Summary")
        print(f"  Tasks: {self.G.number_of_nodes()} | Edges: {self.G.number_of_edges()}")
        print(f"{'='*60}")
        for node in self.priority_order():
            d = self.G.nodes[node]
            crit = "★" if d["critical"] else " "
            print(
                f"  {crit} {node:20s} device={d['device']:3s} "
                f"priority={d['priority']:.3f}s "
                f"conf={d['confidence']:.2f}"
            )
        print()


# ── Example / test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from profiler.rodinia_configs import WORKLOAD_CONFIG

    dag = WorkloadDAG()

    # Add tasks with their static feature sets
    for wl_name, cfg in WORKLOAD_CONFIG.items():
        dag.add_task(
            task_id=wl_name,
            workload=wl_name,
            features={
                "input_size_mb":      cfg["input_size_mb"],
                "compute_intensity":  cfg["compute_intensity"],
                "parallelism_degree": cfg["parallelism_degree"],
                "data_dep":           int(cfg["data_dep"]),
                "transfer_bytes_mb":  cfg["transfer_bytes_mb"],
            },
        )

    # Example pipeline dependencies
    # BFS result feeds CFD; CFD result feeds Pathfinder
    dag.add_dependency("bfs",     "cfd")
    dag.add_dependency("cfd",     "pathfinder")
    dag.add_dependency("hotspot", "srad")

    dag.compute_priorities()
    dag.summary()

    print("Execution order (topological):", dag.topological_order())
    print("\nReady tasks (none completed):", dag.get_ready_tasks(completed=set()))
