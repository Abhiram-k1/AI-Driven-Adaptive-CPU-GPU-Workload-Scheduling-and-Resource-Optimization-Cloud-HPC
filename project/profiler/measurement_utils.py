"""
profiler/measurement_utils.py
=============================
Statistical and measurement helper utilities for the Rodinia profiling pipeline.
Strictly implements repeated-runs protocol from PROJECT_PLAN.md Section 9.7:
  - 1 warm-up run (excluded from statistics)
  - >= 6 timed runs
  - median as primary statistic
  - mean, std, min, max, n_valid as supporting statistics.
"""

import math
import statistics
from typing import List, Dict, Any, Optional


def calculate_run_statistics(values: List[Optional[float]]) -> Dict[str, Any]:
    """
    Computes robust summary statistics from a list of measurements.
    Filters out None, NaN, and non-finite values.
    Returns:
      median, mean, std, min, max, n_valid.
    """
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "median": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "n_valid": 0,
        }

    med = statistics.median(valid)
    mean = statistics.mean(valid)
    std = statistics.stdev(valid) if n_valid > 1 else 0.0
    mn = min(valid)
    mx = max(valid)

    return {
        "median": float(med),
        "mean": float(mean),
        "std": float(std),
        "min": float(mn),
        "max": float(mx),
        "n_valid": n_valid,
    }


def bytes_to_mib(byte_count: int) -> float:
    """Convert bytes to Mebibytes (MiB)."""
    return float(byte_count) / (1024.0 * 1024.0)


def seconds_to_ms(seconds: float) -> float:
    """Convert seconds to milliseconds."""
    return float(seconds) * 1000.0
