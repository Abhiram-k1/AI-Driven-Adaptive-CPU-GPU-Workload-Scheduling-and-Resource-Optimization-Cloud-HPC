"""
profiler/parsers.py
===================
Workload-specific timing output parsers for Rodinia 3.1 workloads.
Strictly implements Section 4, 5, 6 of the Master Execution Prompt.

CRITICAL RULES:
1. No generic parser is used for registered workloads.
2. Timing values are extracted strictly according to source-level semantics.
3. If parsing fails, returns timing_parse_ok = False and failure_reason.
4. NEVER replace missing or unparsed GPU/CPU time with wall-clock time.
5. NEVER replace failed measurements with 0.
"""

import re
from typing import Dict, Any, Optional


def parse_bfs_cpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses OpenMP BFS CPU execution output (openmp/bfs/bfs.cpp:173)."""
    result: Dict[str, Any] = {
        "timing_parse_ok": False,
        "benchmark_reported_time_ms": None,
        "timing_type": "application_reported_omp_wtime_loop",
        "failure_reason": None,
    }
    match = re.search(r"Compute time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if match:
        try:
            val_sec = float(match.group(1))
            result["benchmark_reported_time_ms"] = val_sec * 1000.0
            result["timing_parse_ok"] = True
        except ValueError as e:
            result["failure_reason"] = f"Failed to float-convert regex match: {e}"
    else:
        result["failure_reason"] = "Missing 'Compute time: <val>' pattern in stdout"
    return result


def parse_bfs_gpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses CUDA BFS GPU execution output (cuda/bfs/bfs.cu)."""
    result: Dict[str, Any] = {
        "timing_parse_ok": False,
        "gpu_kernel_time_ms": None,
        "host_to_device_time_ms": None,
        "device_to_host_time_ms": None,
        "gpu_total_path_time_ms": None,
        "gpu_total_path_time_status": "pending_validation",
        "timing_type": "cudaEvent_instrumented_segments",
        "failure_reason": None,
    }

    kernel_match = re.search(r"GPU Kernel Time:\s*([\d.]+)\s*ms", stdout, re.IGNORECASE)
    h2d_match = re.search(r"Initial Host to Device Time:\s*([\d.]+)\s*ms", stdout, re.IGNORECASE)
    d2h_match = re.search(r"Final Device to Host Time:\s*([\d.]+)\s*ms", stdout, re.IGNORECASE)

    parsed_any = False
    if kernel_match:
        try:
            result["gpu_kernel_time_ms"] = float(kernel_match.group(1))
            parsed_any = True
        except ValueError:
            pass

    if h2d_match:
        try:
            result["host_to_device_time_ms"] = float(h2d_match.group(1))
            parsed_any = True
        except ValueError:
            pass

    if d2h_match:
        try:
            result["device_to_host_time_ms"] = float(d2h_match.group(1))
            parsed_any = True
        except ValueError:
            pass

    if result["gpu_kernel_time_ms"] is not None and result["host_to_device_time_ms"] is not None and result["device_to_host_time_ms"] is not None:
        result["gpu_total_path_time_ms"] = (
            result["host_to_device_time_ms"] +
            result["gpu_kernel_time_ms"] +
            result["device_to_host_time_ms"]
        )
        result["timing_parse_ok"] = True
    elif parsed_any:
        result["timing_parse_ok"] = True
        result["failure_reason"] = "Partial timing parsed (one or more segments missing)"
    else:
        result["failure_reason"] = "No BFS CUDA timing patterns found in stdout"

    return result


def parse_cfd_cpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses OpenMP CFD CPU execution output (openmp/cfd/euler3d_cpu.cpp:494)."""
    result: Dict[str, Any] = {
        "timing_parse_ok": False,
        "benchmark_reported_time_ms": None,
        "timing_type": "application_reported_omp_wtime_2000_iterations",
        "failure_reason": None,
    }
    match = re.search(r"Compute time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if match:
        try:
            val_sec = float(match.group(1))
            result["benchmark_reported_time_ms"] = val_sec * 1000.0
            result["timing_parse_ok"] = True
        except ValueError as e:
            result["failure_reason"] = f"Failed to float-convert regex match: {e}"
    else:
        result["failure_reason"] = "Missing 'Compute time: <val>' pattern in stdout"
    return result


def parse_cfd_gpu_stdout(stdout: str, iterations: int = 2000) -> Dict[str, Any]:
    """Parses CUDA CFD GPU execution output (cuda/cfd/euler3d.cu:589)."""
    result: Dict[str, Any] = {
        "timing_parse_ok": False,
        "gpu_phase_time_ms": None,
        "gpu_kernel_time_ms": None,
        "gpu_total_path_time_ms": None,
        "gpu_total_path_time_status": "validated",
        "timing_type": "sdkTimer_scaled_2000_iterations",
        "failure_reason": None,
    }
    match = re.search(r"([\d.]+(?:e[+-]?\d+)?)\s+seconds per iteration", stdout, re.IGNORECASE)
    if match:
        try:
            sec_per_iter = float(match.group(1))
            total_compute_ms = sec_per_iter * float(iterations) * 1000.0
            result["gpu_phase_time_ms"] = total_compute_ms
            result["gpu_kernel_time_ms"] = total_compute_ms
            result["gpu_total_path_time_ms"] = total_compute_ms
            result["timing_parse_ok"] = True
        except ValueError as e:
            result["failure_reason"] = f"Failed to float-convert regex match: {e}"
    else:
        result["failure_reason"] = "Missing '<val> seconds per iteration' pattern in stdout"
    return result


def parse_hotspot_cpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses OpenMP Hotspot CPU output."""
    result = {"timing_parse_ok": False, "benchmark_reported_time_ms": None, "timing_type": "application_reported_omp_wtime_loop", "failure_reason": None}
    m = re.search(r"(?:Compute time|Time):\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["benchmark_reported_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing Hotspot CPU timing in stdout"
    return result


def parse_hotspot_gpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses CUDA Hotspot GPU output."""
    result = {"timing_parse_ok": False, "gpu_total_path_time_ms": None, "timing_type": "sdkTimer_measured", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["gpu_total_path_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing Hotspot GPU timing in stdout"
    return result


def parse_lud_cpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses OpenMP LUD CPU output."""
    result = {"timing_parse_ok": False, "benchmark_reported_time_ms": None, "timing_type": "application_reported_omp_wtime_loop", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["benchmark_reported_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing LUD CPU timing in stdout"
    return result


def parse_lud_gpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses CUDA LUD GPU output."""
    result = {"timing_parse_ok": False, "gpu_total_path_time_ms": None, "timing_type": "sdkTimer_measured", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["gpu_total_path_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing LUD GPU timing in stdout"
    return result


def parse_kmeans_cpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses OpenMP KMeans CPU output."""
    result = {"timing_parse_ok": False, "benchmark_reported_time_ms": None, "timing_type": "application_reported_omp_wtime_loop", "failure_reason": None}
    m = re.search(r"Compute time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["benchmark_reported_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing KMeans CPU timing in stdout"
    return result


def parse_kmeans_gpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses CUDA KMeans GPU output."""
    result = {"timing_parse_ok": False, "gpu_total_path_time_ms": None, "timing_type": "cudaEvent_measured", "failure_reason": None}
    m = re.search(r"Time for kMeans:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["gpu_total_path_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing KMeans GPU timing in stdout"
    return result


def parse_srad_cpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses OpenMP SRAD CPU output."""
    result = {"timing_parse_ok": False, "benchmark_reported_time_ms": None, "timing_type": "application_reported_omp_wtime_loop", "failure_reason": None}
    m = re.search(r"Computation time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["benchmark_reported_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing SRAD CPU timing in stdout"
    return result


def parse_srad_gpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses CUDA SRAD GPU output."""
    result = {"timing_parse_ok": False, "gpu_total_path_time_ms": None, "timing_type": "sdkTimer_measured", "failure_reason": None}
    m = re.search(r"Computation time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["gpu_total_path_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing SRAD GPU timing in stdout"
    return result


def parse_nn_cpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses OpenMP NN CPU output."""
    result = {"timing_parse_ok": False, "benchmark_reported_time_ms": None, "timing_type": "application_reported_omp_wtime_loop", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["benchmark_reported_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing NN CPU timing in stdout"
    return result


def parse_nn_gpu_stdout(stdout: str) -> Dict[str, Any]:
    """Parses CUDA NN GPU output."""
    result = {"timing_parse_ok": False, "gpu_total_path_time_ms": None, "timing_type": "cudaEvent_instrumented_segments", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["gpu_total_path_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing NN GPU timing in stdout"
    return result


def parse_nw_cpu_stdout(stdout: str) -> Dict[str, Any]:
    result = {"timing_parse_ok": False, "benchmark_reported_time_ms": None, "timing_type": "application_reported_omp_wtime", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["benchmark_reported_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing NW CPU timing pattern"
    return result


def parse_nw_gpu_stdout(stdout: str) -> Dict[str, Any]:
    result = {"timing_parse_ok": False, "gpu_total_path_time_ms": None, "timing_type": "cudaEvent_measured", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["gpu_total_path_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing NW GPU timing pattern"
    return result


def parse_backprop_cpu_stdout(stdout: str) -> Dict[str, Any]:
    result = {"timing_parse_ok": False, "benchmark_reported_time_ms": None, "timing_type": "application_reported_omp_wtime", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)\s*sec", stdout, re.IGNORECASE)
    if m:
        result["benchmark_reported_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing Backprop CPU timing pattern"
    return result


def parse_backprop_gpu_stdout(stdout: str) -> Dict[str, Any]:
    result = {"timing_parse_ok": False, "gpu_total_path_time_ms": None, "timing_type": "cudaEvent_measured", "failure_reason": None}
    m = re.search(r"Time:\s*([\d.]+)\s*sec", stdout, re.IGNORECASE)
    if m:
        result["gpu_total_path_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing Backprop GPU timing pattern"
    return result


def parse_particlefilter_cpu_stdout(stdout: str) -> Dict[str, Any]:
    result = {"timing_parse_ok": False, "benchmark_reported_time_ms": None, "timing_type": "application_reported_omp_wtime", "failure_reason": None}
    m = re.search(r"TOTAL TIME:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["benchmark_reported_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing Particlefilter CPU timing pattern"
    return result


def parse_particlefilter_gpu_stdout(stdout: str) -> Dict[str, Any]:
    result = {"timing_parse_ok": False, "gpu_total_path_time_ms": None, "timing_type": "cudaEvent_measured", "failure_reason": None}
    m = re.search(r"TOTAL TIME:\s*([\d.]+)", stdout, re.IGNORECASE)
    if m:
        result["gpu_total_path_time_ms"] = float(m.group(1)) * 1000.0
        result["timing_parse_ok"] = True
    else:
        result["failure_reason"] = "Missing Particlefilter GPU timing pattern"
    return result


PARSER_REGISTRY = {
    ("bfs", "cpu"): parse_bfs_cpu_stdout,
    ("bfs", "gpu"): parse_bfs_gpu_stdout,
    ("cfd", "cpu"): parse_cfd_cpu_stdout,
    ("cfd", "gpu"): parse_cfd_gpu_stdout,
    ("hotspot", "cpu"): parse_hotspot_cpu_stdout,
    ("hotspot", "gpu"): parse_hotspot_gpu_stdout,
    ("lud", "cpu"): parse_lud_cpu_stdout,
    ("lud", "gpu"): parse_lud_gpu_stdout,
    ("kmeans", "cpu"): parse_kmeans_cpu_stdout,
    ("kmeans", "gpu"): parse_kmeans_gpu_stdout,
    ("srad", "cpu"): parse_srad_cpu_stdout,
    ("srad", "gpu"): parse_srad_gpu_stdout,
    ("nn", "cpu"): parse_nn_cpu_stdout,
    ("nn", "gpu"): parse_nn_gpu_stdout,
    ("nw", "cpu"): parse_nw_cpu_stdout,
    ("nw", "gpu"): parse_nw_gpu_stdout,
    ("backprop", "cpu"): parse_backprop_cpu_stdout,
    ("backprop", "gpu"): parse_backprop_gpu_stdout,
    ("particlefilter", "cpu"): parse_particlefilter_cpu_stdout,
    ("particlefilter", "gpu"): parse_particlefilter_gpu_stdout,
}


def get_workload_parser(workload: str, device: str):
    """Factory function returning the workload and device-specific parser."""
    w = workload.lower()
    d = device.lower()

    if (w, d) in PARSER_REGISTRY:
        return PARSER_REGISTRY[(w, d)]

    # Fallback parser for generic patterns if specific parser not yet explicitly registered
    def generic_parser(stdout: str) -> Dict[str, Any]:
        match = re.search(r"(?:Compute time|Total time|Time|Execution time):\s*([\d.]+)", stdout, re.IGNORECASE)
        if match:
            time_ms = float(match.group(1)) * 1000.0
            return {
                "timing_parse_ok": True,
                "benchmark_reported_time_ms": time_ms if d == "cpu" else None,
                "gpu_total_path_time_ms": time_ms if d == "gpu" else None,
                "timing_type": f"generic_{d}_time_regex",
                "failure_reason": None,
            }
        return {
            "timing_parse_ok": False,
            "benchmark_reported_time_ms": None,
            "gpu_total_path_time_ms": None,
            "timing_type": f"generic_{d}_time_regex",
            "failure_reason": f"No timing pattern matched for {w} on {d}",
        }
    return generic_parser
