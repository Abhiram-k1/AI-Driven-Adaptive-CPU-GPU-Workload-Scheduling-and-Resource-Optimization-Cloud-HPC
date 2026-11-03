"""
dataset/build_expanded_dataset.py
=================================
Builds the expanded Layer 3 ML dataset covering all 19 Rodinia workloads and
Small / Medium / Large input profiles (Requirement 2 & Requirement 5).

Guarantees:
1. Strict distinction between empirical measurements from Kaggle Tesla T4 and scaled estimates.
2. Anti-leakage: No post-execution timing metrics in the feature set.
3. Every row clearly tracks 'execution_environment' and 'data_source'.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from profiler.workload_configs import WORKLOAD_INPUT_PROFILES
from ml.feature_schema import (
    FROZEN_RUBRIC_V1,
    extract_pre_execution_features,
    FORBIDDEN_LEAKAGE_COLUMNS,
    MODEL_A_FEATURES
)

DEFAULT_L2_PATH = PROJECT_ROOT / "dataset" / "layer2_aggregated_performance.csv"
DEFAULT_L3_PATH = PROJECT_ROOT / "dataset" / "layer3_ml_dataset.csv"
EXPANDED_L3_PATH = PROJECT_ROOT / "dataset" / "layer3_ml_dataset_expanded.csv"


def build_expanded_dataset(
    l2_path: Path = DEFAULT_L2_PATH,
    l3_path: Path = DEFAULT_L3_PATH,
    out_path: Path = EXPANDED_L3_PATH
) -> pd.DataFrame:
    """Generates the multi-scale, multi-workload expanded ML dataset."""
    print("=" * 80)
    print("BUILDING EXPANDED LAYER 3 ML DATASET (19 WORKLOADS, S/M/L SCALING)")
    print("=" * 80)

    rows = []

    # 1. Base empirical rows from Layer 3
    if l3_path.exists():
        df_l3_base = pd.read_csv(l3_path)
        for _, r in df_l3_base.iterrows():
            item = r.to_dict()
            item["execution_environment"] = "kaggle_tesla_t4"
            item["data_source"] = "empirical_measurement"
            rows.append(item)
        print(f"Loaded {len(rows)} verified empirical rows from {l3_path.name}")

    # 2. Add input-scaled configurations for existing 7 profiled workloads
    existing_workloads = {"bfs", "cfd", "hotspot", "lud", "kmeans", "srad", "nn"}
    
    # Track existing (workload, input_file) pairs to avoid duplication
    existing_pairs = {(r["workload"], r["input_file"]) for r in rows}

    for wl in existing_workloads:
        profiles = WORKLOAD_INPUT_PROFILES.get(wl, {})
        for size_label, p in profiles.items():
            input_file = p["input_file"]
            if (wl, input_file) in existing_pairs:
                continue

            size_mib = float(p["input_size_mib"])
            feats = extract_pre_execution_features(
                workload=wl,
                input_file=input_file,
                input_size_mib=size_mib,
                cpu_cores=4,
                gpu_type="Tesla_T4",
                gpu_memory_gb=16.0
            )

            # Assign preferred device:
            # Small inputs with low compute intensity flip to CPU due to PCIe transfer dominance
            rubric = FROZEN_RUBRIC_V1[wl]
            comp_intensity = rubric["estimated_compute_intensity"]
            transfer_mib = feats["estimated_transfer_size_mib"]

            if size_mib < 1.0 and comp_intensity <= 2:
                pref_dev = "cpu"
            elif wl == "nn":
                pref_dev = "cpu"  # empirically validated as CPU-preferred on T4
            elif wl == "bfs" and size_label == "S":
                pref_dev = "cpu"  # graph4096 is empirically CPU-preferred
            else:
                pref_dev = "gpu"

            feats["preferred_device"] = pref_dev
            feats["target_label"] = 1 if pref_dev == "gpu" else 0
            feats["execution_environment"] = "kaggle_tesla_t4_scaled"
            feats["data_source"] = "input_scaled_estimate"

            rows.append(feats)
            existing_pairs.add((wl, input_file))

    # 3. Add remaining 12 Rodinia workloads across S/M/L scales
    extended_workloads = [
        "nw", "backprop", "pathfinder", "particlefilter", "heartwall",
        "b_tree", "leukocyte", "streamcluster", "hybridsort", "lavaMD",
        "gaussian", "mummergpu"
    ]

    for wl in extended_workloads:
        profiles = WORKLOAD_INPUT_PROFILES.get(wl, {})
        rubric = FROZEN_RUBRIC_V1[wl]
        
        # Determine algorithmic device preference
        comp = rubric["estimated_compute_intensity"]
        par = rubric["estimated_parallelism"]
        mem_bound = rubric["estimated_memory_boundness"]
        irreg = rubric["has_irregular_memory"]
        serial = rubric["has_strong_serial_dependency"]

        # High parallelism and compute-intensity favors GPU; heavy irregular pointer-chasing favors CPU
        gpu_suitability = (comp * 1.5) + (par * 1.5) - (mem_bound * 1.0) - (irreg * 1.2) - (serial * 1.2)
        base_pref = "gpu" if gpu_suitability >= 4.0 else "cpu"

        for size_label, p in profiles.items():
            input_file = p["input_file"]
            size_mib = float(p["input_size_mib"])

            feats = extract_pre_execution_features(
                workload=wl,
                input_file=input_file,
                input_size_mib=size_mib,
                cpu_cores=4,
                gpu_type="Tesla_T4",
                gpu_memory_gb=16.0
            )

            # Small sizes may flip to CPU
            if size_label == "S" and size_mib < 5.0 and comp <= 3:
                pref_dev = "cpu"
            else:
                pref_dev = base_pref

            feats["preferred_device"] = pref_dev
            feats["target_label"] = 1 if pref_dev == "gpu" else 0
            feats["execution_environment"] = "unprofiled_host_blocked_cuda"
            feats["data_source"] = "rubric_heuristic__no_timing_data"

            rows.append(feats)

    df_expanded = pd.DataFrame(rows)

    # Validate zero leakage
    for col in df_expanded.columns:
        assert col not in FORBIDDEN_LEAKAGE_COLUMNS, f"DATA LEAKAGE DETECTED: {col}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_expanded.to_csv(out_path, index=False)
    print(f"Expanded dataset saved to: {out_path}")
    print(f"Total configurations: {len(df_expanded)}")
    print(f"Workload count: {df_expanded['workload'].nunique()}")
    print("Class distribution:\n", df_expanded["preferred_device"].value_counts())
    print("Data source breakdown:\n", df_expanded["data_source"].value_counts())

    return df_expanded


if __name__ == "__main__":
    build_expanded_dataset()

