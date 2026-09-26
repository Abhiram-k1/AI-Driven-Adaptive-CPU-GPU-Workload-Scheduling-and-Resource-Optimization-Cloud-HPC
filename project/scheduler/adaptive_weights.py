"""
adaptive_weights.py
===================
Bayesian Online Learning for multi-objective scheduler weight adaptation.

Three objectives:
  w1 = Makespan weight         (minimize total completion time)
  w2 = Energy weight           (minimize energy proxy = runtime * utilization)
  w3 = Transfer cost weight    (minimize PCIe transfer overhead)

Weights are maintained as a Dirichlet distribution and updated using
Thompson Sampling: after each task completes, we observe a "reward"
(lower is better for each objective), and update the posterior.

Usage:
    from scheduler.adaptive_weights import AdaptiveWeightController
    awc = AdaptiveWeightController()

    # After a task completes:
    awc.update(
        makespan_cost=1.2,       # actual runtime (s)
        energy_cost=0.8,         # GPU utilization * runtime
        transfer_cost=0.05,      # actual transfer time (s)
    )
    w1, w2, w3 = awc.sample_weights()
"""

import numpy as np
from collections import deque
from typing import Tuple


class AdaptiveWeightController:
    """
    Maintains adaptive objective weights using Bayesian Online Learning.

    Internally keeps a Dirichlet concentration vector [α1, α2, α3].
    Each α increases when the corresponding objective is dominant (costly),
    giving it a higher weight in the next scheduling round.
    """

    def __init__(
        self,
        initial_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        window: int = 20,
        learning_rate: float = 0.3,
    ):
        self.alpha = np.array(list(initial_weights), dtype=float)
        self.history = deque(maxlen=window)
        self.lr = learning_rate
        self._round = 0

    def update(
        self,
        makespan_cost: float,
        energy_cost: float,
        transfer_cost: float,
    ) -> None:
        """
        Observe actual costs after task completion and update Dirichlet params.
        Higher actual cost → increase α for that objective → higher weight.
        """
        costs = np.array([makespan_cost, energy_cost, transfer_cost], dtype=float)
        costs = np.clip(costs, 1e-6, None)

        # Normalise costs to [0, 1] relative to each other
        norm_costs = costs / costs.sum()

        # Bayesian update: α increases proportional to observed cost
        self.alpha += self.lr * norm_costs * 3   # *3 keeps alpha scale reasonable

        # Clip to avoid degenerate distributions
        self.alpha = np.clip(self.alpha, 0.5, 50.0)

        self.history.append(norm_costs)
        self._round += 1

    def sample_weights(self) -> Tuple[float, float, float]:
        """
        Thompson Sampling: draw weights from the current Dirichlet posterior.
        Returns (w_makespan, w_energy, w_transfer) summing to 1.
        """
        weights = np.random.dirichlet(self.alpha)
        return tuple(float(w) for w in weights)

    def mean_weights(self) -> Tuple[float, float, float]:
        """Return the expected (mean) weights from the Dirichlet distribution."""
        mean = self.alpha / self.alpha.sum()
        return tuple(float(w) for w in mean)

    def set_mode(self, mode: str) -> None:
        """
        Manually bias weights toward a scheduling priority mode.
        mode options: 'speed', 'energy', 'balanced'
        """
        if mode == "speed":
            self.alpha = np.array([5.0, 1.0, 1.0])
        elif mode == "energy":
            self.alpha = np.array([1.0, 5.0, 1.0])
        elif mode == "transfer":
            self.alpha = np.array([1.0, 1.0, 5.0])
        elif mode == "balanced":
            self.alpha = np.array([1.0, 1.0, 1.0])

    def pressure_adjust(
        self,
        queue_length: int,
        gpu_mem_used_ratio: float,
        deadline_proximity: float,   # 0=far, 1=imminent
    ) -> None:
        """
        Auto-adjust α based on system pressure signals:
        - Long queue        → increase makespan weight
        - GPU memory full   → increase transfer weight (avoid GPU)
        - Near deadline     → increase makespan weight aggressively
        """
        if queue_length > 10:
            self.alpha[0] *= (1.0 + 0.05 * min(queue_length, 20))
        if gpu_mem_used_ratio > 0.85:
            self.alpha[2] *= 1.5
        if deadline_proximity > 0.8:
            self.alpha[0] *= 2.0
        self.alpha = np.clip(self.alpha, 0.5, 50.0)

    def summary(self) -> str:
        w = self.mean_weights()
        return (
            f"Round {self._round} | α={self.alpha.round(2)} | "
            f"mean_w=(makespan={w[0]:.3f}, energy={w[1]:.3f}, transfer={w[2]:.3f})"
        )


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    awc = AdaptiveWeightController()
    print("Initial:", awc.summary())

    # Simulate a batch of tasks with varying costs
    np.random.seed(42)
    for i in range(15):
        # Simulate high transfer cost scenario
        awc.update(
            makespan_cost=np.random.uniform(0.5, 2.0),
            energy_cost=np.random.uniform(0.3, 1.0),
            transfer_cost=np.random.uniform(0.8, 2.5),   # transfer is dominant
        )

    print("After 15 updates (high transfer cost):", awc.summary())

    # Thompson sample weights 5 times
    print("\nSampled weights:")
    for _ in range(5):
        w = awc.sample_weights()
        print(f"  makespan={w[0]:.3f}  energy={w[1]:.3f}  transfer={w[2]:.3f}")

    # Simulate deadline pressure
    awc.pressure_adjust(queue_length=15, gpu_mem_used_ratio=0.9, deadline_proximity=0.9)
    print("\nAfter deadline pressure:", awc.summary())
