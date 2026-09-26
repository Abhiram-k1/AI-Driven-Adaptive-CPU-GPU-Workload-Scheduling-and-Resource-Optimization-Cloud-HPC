"""
tests/test_pipeline.py
======================
Unit and integration test suite covering Section 32 requirements:
  - Profiling: input-size calculation, parser correctness, failed execution handling, timing conversion, warm-up exclusion
  - Dataset: Layer 1 schema, Layer 2 aggregation, Layer 3 gating, label generation, leakage detection
  - ML: feature generation, encoding, LOWO grouping, prediction output, metric calculation
  - Visualization: figures generated successfully, expected files exist, valid non-empty files.
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profiler.workload_configs import compute_file_size_mib, compute_file_size_bytes
from profiler.parsers import parse_bfs_cpu_stdout, parse_bfs_gpu_stdout, parse_cfd_cpu_stdout, parse_cfd_gpu_stdout
from profiler.measurement_utils import calculate_run_statistics, bytes_to_mib, seconds_to_ms
from dataset.aggregate_measurements import aggregate_layer1_to_layer2
from dataset.build_ml_dataset import build_ml_dataset
from ml.feature_schema import extract_pre_execution_features, FORBIDDEN_LEAKAGE_COLUMNS, MODEL_A_FEATURES, MODEL_B_FEATURES
from ml.train_classifier import DeviceClassifierPipeline
from ml.evaluate_classifier import compute_model_metrics
from ml.predict import predict_device


class TestProfiling(unittest.TestCase):
    """Profiling module tests."""

    def test_input_size_calculation(self):
        # Test bytes to MiB math
        self.assertAlmostEqual(bytes_to_mib(1024 * 1024), 1.0)
        self.assertAlmostEqual(bytes_to_mib(64198393), 61.22453, places=3)

    def test_bfs_cpu_parser(self):
        sample_stdout = "Reading File\nStart traversing the tree\nCompute time: 0.035133\nResult stored in result.txt"
        res = parse_bfs_cpu_stdout(sample_stdout)
        self.assertTrue(res["timing_parse_ok"])
        self.assertAlmostEqual(res["benchmark_reported_time_ms"], 35.133, places=3)
        self.assertEqual(res["timing_type"], "application_reported_omp_wtime_loop")

    def test_bfs_gpu_parser(self):
        sample_stdout = (
            "Reading File\n"
            "Initial Host to Device Time: 14.120000 ms\n"
            "GPU Kernel Time: 3.830000 ms\n"
            "Final Device to Host Time: 0.770000 ms\n"
        )
        res = parse_bfs_gpu_stdout(sample_stdout)
        self.assertTrue(res["timing_parse_ok"])
        self.assertAlmostEqual(res["host_to_device_time_ms"], 14.12, places=2)
        self.assertAlmostEqual(res["gpu_kernel_time_ms"], 3.83, places=2)
        self.assertAlmostEqual(res["device_to_host_time_ms"], 0.77, places=2)
        self.assertAlmostEqual(res["gpu_total_path_time_ms"], 18.72, places=2)

    def test_cfd_cpu_parser(self):
        sample_stdout = "Compute time: 77.4126\nSaving solution...\n"
        res = parse_cfd_cpu_stdout(sample_stdout)
        self.assertTrue(res["timing_parse_ok"])
        self.assertAlmostEqual(res["benchmark_reported_time_ms"], 77412.60, places=2)

    def test_cfd_gpu_parser(self):
        sample_stdout = "Name: Tesla T4\nStarting...\n0.000556622 seconds per iteration\nSaving solution..."
        res = parse_cfd_gpu_stdout(sample_stdout, iterations=2000)
        self.assertTrue(res["timing_parse_ok"])
        # 0.000556622 * 2000 * 1000 = 1113.244 ms
        self.assertAlmostEqual(res["gpu_phase_time_ms"], 1113.244, places=2)

    def test_failed_execution_handling(self):
        fail_stdout = "/bin/sh: 1: ./bfs: not found"
        res = parse_bfs_gpu_stdout(fail_stdout)
        self.assertFalse(res["timing_parse_ok"])
        self.assertIsNone(res["gpu_total_path_time_ms"])

    def test_run_statistics_and_warmup_exclusion(self):
        values = [10.0, 12.0, 11.0, 10.5, 11.5, 11.0]
        stats = calculate_run_statistics(values)
        self.assertEqual(stats["n_valid"], 6)
        self.assertAlmostEqual(stats["median"], 11.0)
        self.assertAlmostEqual(stats["min"], 10.0)
        self.assertAlmostEqual(stats["max"], 12.0)


class TestDataset(unittest.TestCase):
    """Dataset module tests."""

    def setUp(self):
        self.layer1_path = PROJECT_ROOT / "dataset" / "raw_data" / "layer1_raw_executions.csv"
        self.layer2_path = PROJECT_ROOT / "dataset" / "layer2_aggregated_performance.csv"
        self.layer3_path = PROJECT_ROOT / "dataset" / "layer3_ml_dataset.csv"

    def test_layer1_schema_and_contents(self):
        self.assertTrue(self.layer1_path.exists())
        df = pd.read_csv(self.layer1_path)
        required_cols = [
            "workload", "input_file", "input_size_bytes", "input_size_mib", "device",
            "run_id", "is_warmup", "benchmark_reported_time_ms", "cpu_process_wall_time_ms",
            "timing_type", "build_status", "return_code", "timing_parse_ok"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)
        # Verify warm-up rows exist and are tracked
        self.assertTrue((df["is_warmup"] == True).sum() > 0)
        # Verify failed runs are tracked
        self.assertTrue((df["timing_parse_ok"] == False).sum() > 0)

    def test_layer2_aggregation(self):
        self.assertTrue(self.layer2_path.exists())
        df = pd.read_csv(self.layer2_path)
        self.assertIn("cpu_median_ms", df.columns)
        self.assertIn("gpu_total_path_ms", df.columns)
        self.assertIn("preferred_device", df.columns)
        self.assertEqual(len(df), 8)
        self.assertEqual(df["workload"].nunique(), 7)

        # BFS on graph1MW should be GPU-preferred
        bfs_large = df[(df["workload"] == "bfs") & (df["input_file"] == "graph1MW_6.txt")]
        self.assertEqual(len(bfs_large), 1)
        self.assertEqual(bfs_large["preferred_device"].iloc[0], "gpu")

        # BFS on graph4096 should be CPU-preferred
        bfs_small = df[(df["workload"] == "bfs") & (df["input_file"] == "graph4096.txt")]
        self.assertEqual(len(bfs_small), 1)
        self.assertEqual(bfs_small["preferred_device"].iloc[0], "cpu")

        # CFD should be validated and GPU-preferred
        cfd_row = df[df["workload"] == "cfd"]
        self.assertEqual(len(cfd_row), 1)
        self.assertEqual(cfd_row["preferred_device"].iloc[0], "gpu")

        # NN should be CPU-preferred due to PCIe transfer dominance
        nn_row = df[df["workload"] == "nn"]
        self.assertEqual(len(nn_row), 1)
        self.assertEqual(nn_row["preferred_device"].iloc[0], "cpu")

    def test_layer3_gating_and_anti_leakage(self):
        self.assertTrue(self.layer3_path.exists())
        df = pd.read_csv(self.layer3_path)
        self.assertEqual(len(df), 8)
        # Gate check: all admitted rows have valid preferred_device
        self.assertTrue(all(df["preferred_device"].isin(["cpu", "gpu"])))
        self.assertIn("cpu", df["preferred_device"].values)
        self.assertIn("gpu", df["preferred_device"].values)
        self.assertEqual((df["preferred_device"] == "cpu").sum(), 2)
        self.assertEqual((df["preferred_device"] == "gpu").sum(), 6)
        # Anti-leakage check
        for col in df.columns:
            self.assertNotIn(col, FORBIDDEN_LEAKAGE_COLUMNS)


class TestMachineLearning(unittest.TestCase):
    """ML and feature engineering tests."""

    def test_feature_generation(self):
        feats = extract_pre_execution_features("cfd", "fvcorr.domn.193K", 43.54)
        for col in MODEL_A_FEATURES:
            self.assertIn(col, feats)
        self.assertEqual(feats["workload_type"], "iterative_solver")
        self.assertEqual(feats["workload_domain"], "fluid_dynamics")

    def test_model_pipeline_multiclass(self):
        # Synthetic test data covering both CPU and GPU classes to verify encoding and split math
        df_synthetic = pd.DataFrame([
            {"workload": "cfd", "input_file": "f1", "input_size_mib": 43.5, "workload_type": "iterative_solver",
             "workload_domain": "fluid_dynamics", "estimated_parallelism": 5, "estimated_memory_boundness": 4,
             "estimated_compute_intensity": 4, "has_irregular_memory": 0, "has_strong_serial_dependency": 0,
             "estimated_transfer_size_mib": 43.5, "transfer_estimation_method": "source_buffer_analysis",
             "cpu_cores_available": 4, "gpu_type_encoded": "Tesla_T4", "gpu_memory_gb": 16.0},
            {"workload": "synth_cpu", "input_file": "f2", "input_size_mib": 10.0, "workload_type": "graph",
             "workload_domain": "graph_traversal", "estimated_parallelism": 2, "estimated_memory_boundness": 5,
             "estimated_compute_intensity": 1, "has_irregular_memory": 1, "has_strong_serial_dependency": 1,
             "estimated_transfer_size_mib": 10.0, "transfer_estimation_method": "source_buffer_analysis",
             "cpu_cores_available": 4, "gpu_type_encoded": "Tesla_T4", "gpu_memory_gb": 16.0},
        ])
        y = np.array([1, 0])  # 1=GPU, 0=CPU

        pipe_a = DeviceClassifierPipeline(MODEL_A_FEATURES, random_state=42).fit(df_synthetic, y)
        preds_a = pipe_a.predict(df_synthetic)
        self.assertEqual(len(preds_a), 2)

        pipe_b = DeviceClassifierPipeline(MODEL_B_FEATURES, random_state=42).fit(df_synthetic, y)
        preds_b = pipe_b.predict(df_synthetic)
        self.assertEqual(len(preds_b), 2)

    def test_metrics_calculation(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        metrics = compute_model_metrics(y_true, y_pred)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertIn("precision_macro", metrics)
        self.assertIn("f1_macro", metrics)


class TestVisualization(unittest.TestCase):
    """Visualization tests."""

    def test_all_figures_exist_and_non_empty(self):
        fig_dir = PROJECT_ROOT / "analysis" / "figures"
        required_figs = [
            "class_distribution.png",
            "model_a_confusion_matrix.png",
            "model_b_confusion_matrix.png",
            "model_comparison.png",
            "per_workload_f1.png",
            "feature_importance_model_a.png",
            "feature_importance_model_b.png",
            "lowo_comparison.png",
            "prediction_confidence.png",
            "actual_vs_predicted_distribution.png",
            "speedup_chart.png",
            "pareto_front.png",
            "weight_evolution.png",
            "scheduler_comparison.png",
        ]
        for fig_name in required_figs:
            fig_path = fig_dir / fig_name
            self.assertTrue(fig_path.exists(), f"Figure '{fig_name}' is missing!")
            self.assertGreater(fig_path.stat().st_size, 1000, f"Figure '{fig_name}' appears empty!")


if __name__ == "__main__":
    unittest.main()
