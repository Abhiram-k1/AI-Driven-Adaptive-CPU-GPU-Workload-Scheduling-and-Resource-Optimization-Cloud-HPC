"""
feature_extractor.py
====================
Transforms raw profiling CSV into ML-ready features.

Derived features per workload:
  - cpu_mean_s, gpu_mean_s          : mean runtime per device
  - cpu_std_s, gpu_std_s            : std deviation
  - speedup_ratio                   : cpu_mean / gpu_mean (>1 means GPU faster)
  - transfer_overhead_ratio         : transfer_time / gpu_mean (penalises GPU)
  - compute_transfer_ratio          : compute_intensity / transfer_bytes_mb
  - best_device                     : label: 'gpu' if speedup_ratio > threshold, else 'cpu'
  - best_device_binary              : 1=gpu, 0=cpu (target for classifier)

Usage:
    python profiler/feature_extractor.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_CSV  = Path(__file__).parent.parent / "data" / "profiles" / "rodinia_benchmark_data.csv"
FEAT_CSV = Path(__file__).parent.parent / "data" / "datasets" / "features.csv"

# PCIe 3.0 x16 bandwidth in MB/s
PCIE_MB_PER_S = 12_000.0

# GPU is chosen if it is at least this much faster than CPU
GPU_SPEEDUP_THRESHOLD = 1.10  # 10% faster


def extract_features(raw_path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # Keep only successful runs
    df = df[df["success"] == 1].copy()

    # Aggregate per (workload, device)
    agg = (
        df.groupby(["workload", "device"])["elapsed_s"]
        .agg(mean="mean", std="std", min="min", max="max", median="median")
        .reset_index()
    )

    cpu_agg = agg[agg["device"] == "cpu"].set_index("workload").add_prefix("cpu_")
    gpu_agg = agg[agg["device"] == "gpu"].set_index("workload").add_prefix("gpu_")

    # Merge cpu + gpu stats
    feat = cpu_agg.join(gpu_agg, how="inner").reset_index()
    feat = feat.rename(columns={"workload": "workload"})

    # Pull static features from one device's rows
    static_cols = [
        "workload", "input_size_mb", "compute_intensity",
        "parallelism_degree", "data_dep", "transfer_bytes_mb", "domain",
    ]
    static = df.drop_duplicates("workload")[static_cols].set_index("workload")
    feat = feat.set_index("workload").join(static).reset_index()

    # ── Derived features ────────────────────────────────────────────────────
    # Speedup ratio: >1 means GPU is faster
    feat["speedup_ratio"] = feat["cpu_mean"] / feat["gpu_mean"]

    # Estimated PCIe transfer time in seconds
    feat["transfer_time_s"] = feat["transfer_bytes_mb"] / PCIE_MB_PER_S

    # Transfer overhead ratio: what fraction of GPU time is spent in transfer
    feat["transfer_overhead_ratio"] = feat["transfer_time_s"] / feat["gpu_mean"]

    # Compute-to-transfer ratio: high = transfer cost is small relative to work
    feat["compute_transfer_ratio"] = feat["compute_intensity"] / (feat["transfer_bytes_mb"] + 1e-6)

    # Variance stability: low std => predictable behaviour
    feat["cpu_cv"] = feat["cpu_std"] / (feat["cpu_mean"] + 1e-9)  # coefficient of variation
    feat["gpu_cv"] = feat["gpu_std"] / (feat["gpu_mean"] + 1e-9)

    # Log-transformed sizes (more numerically stable for trees/SVMs)
    feat["log_input_mb"]  = np.log1p(feat["input_size_mb"])
    feat["log_speedup"]   = np.log(feat["speedup_ratio"])

    # ── Labels ──────────────────────────────────────────────────────────────
    feat["best_device"] = feat["speedup_ratio"].apply(
        lambda r: "gpu" if r >= GPU_SPEEDUP_THRESHOLD else "cpu"
    )
    feat["best_device_binary"] = (feat["best_device"] == "gpu").astype(int)

    return feat


def save_features(feat: pd.DataFrame, out_path: Path = FEAT_CSV) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_path, index=False)
    print(f"Features saved → {out_path}")
    print(f"  Shape: {feat.shape}")
    print(f"  GPU-best: {feat['best_device_binary'].sum()} / {len(feat)}")
    print(feat[["workload", "cpu_mean", "gpu_mean", "speedup_ratio", "best_device"]].to_string(index=False))


if __name__ == "__main__":
    feat = extract_features()
    save_features(feat)
