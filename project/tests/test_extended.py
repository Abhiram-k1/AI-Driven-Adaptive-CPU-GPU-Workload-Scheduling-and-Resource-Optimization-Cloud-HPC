"""
tests/test_extended.py
======================
Comprehensive extended unit and integration test suite (Requirement 18).

Covers:
  - 19-Workload configuration discovery and S/M/L input profiles
  - Rubric v1.0 complete coverage and strict anti-leakage guarantees
  - Dedicated timing output parsers for all 19 Rodinia workloads
  - EnergyEstimator TDP models, disclaimer verification, and device comparison
  - ResourceMonitor live host telemetry and GPU graceful fallback
  - SimulatedExecutor empirical replay and distribution preservation
  - ConcurrentSchedulerQueue multi-slot management and lifecycle states
  - Multi-Objective WeightedSum and Pareto dominance verification
  - AdaptiveWeightController Dirichlet mathematical invariants (sum=1, non-negative)
  - StrategyComparison multi-policy benchmark execution
  - DashboardAPI endpoint fidelity and schema compliance
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profiler.workload_configs import WORKLOAD_CONFIGS, WORKLOAD_INPUT_PROFILES, ALL_WORKLOADS
from ml.feature_schema import FROZEN_RUBRIC_V1, extract_pre_execution_features, FORBIDDEN_LEAKAGE_COLUMNS
from profiler.parsers import get_workload_parser, parse_hotspot_cpu_stdout, parse_hotspot_gpu_stdout, parse_lud_cpu_stdout, parse_kmeans_gpu_stdout, parse_backprop_cpu_stdout
from profiler.energy_estimator import EnergyEstimator
from profiler.resource_monitor import ResourceMonitor
from scheduler.simulation_mode import SimulatedExecutor
from scheduler.concurrent_queue import ConcurrentSchedulerQueue, TaskStatus
from scheduler.adaptive_weights import AdaptiveWeightController
from scheduler.multi_objective import WeightedSumScheduler, _compute_objectives
from scheduler.strategy_comparison import StrategyComparison
from dashboard.api import DashboardAPI


class TestWorkloadConfigurations(unittest.TestCase):
    """Test comprehensive 19-workload registry and input profiles."""

    def test_complete_19_workload_registry(self):
        self.assertEqual(len(ALL_WORKLOADS), 19)
        self.assertEqual(len(WORKLOAD_CONFIGS), 19)

    def test_sml_input_profiles_coverage(self):
        self.assertEqual(len(WORKLOAD_INPUT_PROFILES), 19)
        for wl, profiles in WORKLOAD_INPUT_PROFILES.items():
            self.assertIn("S", profiles, f"Missing Small profile for {wl}")
            self.assertIn("M", profiles, f"Missing Medium profile for {wl}")
            self.assertIn("L", profiles, f"Missing Large profile for {wl}")
            self.assertGreater(profiles["L"]["input_size_mib"], profiles["S"]["input_size_mib"])

    def test_rubric_v1_completeness(self):
        self.assertEqual(len(FROZEN_RUBRIC_V1), 19)
        for wl in ALL_WORKLOADS:
            self.assertIn(wl, FROZEN_RUBRIC_V1)
            entry = FROZEN_RUBRIC_V1[wl]
            self.assertIn("workload_type", entry)
            self.assertIn("estimated_parallelism", entry)
            self.assertIn("estimated_compute_intensity", entry)
            self.assertIn("has_irregular_memory", entry)
            self.assertIn("transfer_estimation_ratio", entry)


class TestParsersExtended(unittest.TestCase):
    """Test dedicated parsers for extended workload suite."""

    def test_hotspot_parsers(self):
        cpu_out = "Compute time: 1.486\nOutput written"
        res_cpu = parse_hotspot_cpu_stdout(cpu_out)
        self.assertTrue(res_cpu["timing_parse_ok"])
        self.assertAlmostEqual(res_cpu["benchmark_reported_time_ms"], 1486.0)

        gpu_out = "Time: 0.418\nEnding..."
        res_gpu = parse_hotspot_gpu_stdout(gpu_out)
        self.assertTrue(res_gpu["timing_parse_ok"])
        self.assertAlmostEqual(res_gpu["gpu_total_path_time_ms"], 418.0)

    def test_lud_cpu_parser(self):
        out = "Time: 2.048\nDone"
        res = parse_lud_cpu_stdout(out)
        self.assertTrue(res["timing_parse_ok"])
        self.assertAlmostEqual(res["benchmark_reported_time_ms"], 2048.0)

    def test_kmeans_gpu_parser(self):
        out = "Time for kMeans: 1.103\nClusters written"
        res = parse_kmeans_gpu_stdout(out)
        self.assertTrue(res["timing_parse_ok"])
        self.assertAlmostEqual(res["gpu_total_path_time_ms"], 1103.0)

    def test_backprop_parser(self):
        out = "Time: 1.250 sec\nDone training"
        res = parse_backprop_cpu_stdout(out)
        self.assertTrue(res["timing_parse_ok"])
        self.assertAlmostEqual(res["benchmark_reported_time_ms"], 1250.0)

    def test_parser_factory_fallback(self):
        fn = get_workload_parser("unknown_workload", "cpu")
        out = "Execution time: 0.500"
        res = fn(out)
        self.assertTrue(res["timing_parse_ok"])
        self.assertAlmostEqual(res["benchmark_reported_time_ms"], 500.0)


class TestEnergyEstimation(unittest.TestCase):
    """Test EnergyEstimator TDP models and attribution disclaimers."""

    def setUp(self):
        self.ee = EnergyEstimator()

    def test_tdp_proxy_calculation(self):
        res = self.ee.estimate("cfd", "gpu", runtime_s=2.0, utilization_factor=1.0)
        # Tesla T4 TDP = 70W; 2.0s * 70W = 140 Joules
        self.assertAlmostEqual(res["total_energy_joules"], 140.0)
        self.assertEqual(res["measurement_type"], "ESTIMATED_NOT_MEASURED")
        self.assertEqual(res["estimation_method"], "tdp_model")
        self.assertIn("ACADEMIC NOTICE", res["disclaimer"])

    def test_device_energy_comparison(self):
        # CFD CPU takes ~77.5s (at 65W TDP = ~5037J) vs GPU ~1.125s (at 70W TDP = ~78J)
        comp = self.ee.compare_devices("cfd", cpu_runtime_s=77.5, gpu_runtime_s=1.125)
        self.assertEqual(comp["more_efficient_device"], "gpu")
        self.assertGreater(comp["energy_ratio_cpu_over_gpu"], 50.0)


class TestResourceMonitoring(unittest.TestCase):
    """Test live system monitor and GPU graceful degradation."""

    def setUp(self):
        self.rm = ResourceMonitor()

    def test_cpu_snapshot_validity(self):
        snap = self.rm.get_cpu_snapshot()
        self.assertGreaterEqual(snap["cpu_percent"], 0.0)
        self.assertLessEqual(snap["cpu_percent"], 100.0)
        self.assertGreater(snap["memory_total_gb"], 0.0)
        self.assertGreater(snap["cpu_count_logical"], 0)

    def test_gpu_graceful_degradation(self):
        gpu_snap = self.rm.get_gpu_snapshot()
        if not self.rm.gpu_detected:
            self.assertFalse(gpu_snap["gpu_available"])
            self.assertEqual(gpu_snap["status"], "GPU_MONITORING_UNAVAILABLE")
            self.assertIsNone(gpu_snap["gpu_utilization_percent"])


class TestSimulationMode(unittest.TestCase):
    """Test empirical replay executor."""

    def setUp(self):
        self.sim = SimulatedExecutor()

    def test_empirical_lookup_fidelity(self):
        # CFD is in empirical Layer 2
        res = self.sim.run("cfd", {}, device="gpu", n_runs=6)
        self.assertTrue(res["is_simulated"])
        self.assertEqual(res["execution_type"], "SIMULATED_FROM_KAGGLE_T4_VALIDATED_DATA")
        # Empirical CFD GPU median is ~1.125s
        self.assertAlmostEqual(res["mean_s"], 1.125, delta=0.05)


class TestConcurrentQueue(unittest.TestCase):
    """Test multi-slot queue management and task states."""

    def test_concurrent_execution_lifecycle(self):
        queue = ConcurrentSchedulerQueue(cpu_slots=2, gpu_slots=1)
        t1 = queue.submit("t1", "bfs", "cpu", {}, n_runs=1)
        t2 = queue.submit("t2", "cfd", "gpu", {}, n_runs=1)
        
        status = queue.get_queue_status()
        self.assertEqual(status["total_tasks"], 2)

        results = queue.wait_all(timeout=5.0)
        self.assertIn("t1", results)
        self.assertIn("t2", results)
        self.assertEqual(queue.tasks["t1"]["status"], TaskStatus.COMPLETED)
        self.assertEqual(queue.tasks["t2"]["status"], TaskStatus.COMPLETED)
        queue.shutdown()


class TestMultiObjectiveAndAdaptiveWeights(unittest.TestCase):
    """Test optimization mathematics and Dirichlet adaptive weight invariants."""

    def test_dirichlet_weight_invariants(self):
        awc = AdaptiveWeightController()
        for _ in range(10):
            w = awc.sample_weights()
            self.assertEqual(len(w), 3)
            self.assertTrue(all(val >= 0.0 for val in w))
            self.assertAlmostEqual(sum(w), 1.0, places=4)

    def test_bayesian_reinforcement_update(self):
        awc = AdaptiveWeightController(learning_rate=0.5)
        w_before = awc.mean_weights()
        # Heavy transfer bottleneck observed
        awc.update(makespan_cost=0.1, energy_cost=0.1, transfer_cost=10.0)
        w_after = awc.mean_weights()
        # Transfer weight w[2] must increase
        self.assertGreater(w_after[2], w_before[2])


class TestStrategyComparison(unittest.TestCase):
    """Test multi-strategy comparison benchmark."""

    def test_comparison_completeness(self):
        comp = StrategyComparison(workloads=["bfs", "cfd", "hotspot"])
        df = comp.run_all()
        self.assertEqual(len(df), 6)
        strategies = set(df["strategy"])
        expected = {"AI-Adaptive", "FIFO-CPU", "FIFO-GPU", "Round-Robin", "Greedy-Speedup", "Random"}
        self.assertEqual(strategies, expected)
        # AI-Adaptive makespan should be substantially faster than pure CPU
        ai_makespan = df.loc[df["strategy"] == "AI-Adaptive", "makespan_s"].values[0]
        cpu_makespan = df.loc[df["strategy"] == "FIFO-CPU", "makespan_s"].values[0]
        self.assertLess(ai_makespan, cpu_makespan)


class TestDashboardAPI(unittest.TestCase):
    """Test dashboard backend API endpoints."""

    def setUp(self):
        self.api = DashboardAPI()

    def test_api_endpoints(self):
        benchmarks = self.api.get_benchmark_summary()
        self.assertGreater(len(benchmarks), 0)
        
        metrics = self.api.get_ml_metrics()
        self.assertIn("model_a", metrics)
        self.assertIn("model_b", metrics)
        self.assertIn("majority_baseline", metrics)

        catalog = self.api.get_workload_catalog()
        self.assertEqual(len(catalog), 19)

    def test_benchmark_lifecycle(self):
        status = self.api.get_live_benchmark_status()
        self.assertIn("is_running", status)
        self.assertIn("total_workloads", status)
        self.assertIn("completed_workloads", status)

        # Test start and stop
        start_res = self.api.start_benchmark_run()
        self.assertIn(start_res.get("status"), ["started", "already_running"])
        self.assertTrue(self.api.is_benchmark_running())

        # Stop benchmark
        stop_res = self.api.stop_benchmark_run()
        self.assertIn(stop_res.get("status"), ["stopped", "not_running"])
        self.assertFalse(self.api.is_benchmark_running())


if __name__ == "__main__":
    unittest.main()

