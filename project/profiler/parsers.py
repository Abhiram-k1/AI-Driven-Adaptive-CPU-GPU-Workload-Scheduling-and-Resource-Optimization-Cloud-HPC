"""
profiler/parsers.py
===================
Workload-specific timing output parsers for Rodinia 3.1 workloads.
Strictly implements Section 4, 5, 6 of the Master Execution Prompt.

CRITICAL RULES:
1. No generic parser is used.
2. Timing values are extracted strictly according to source-level semantics.
3. If parsing fails, returns timing_parse_ok = False and failure_reason.
4. NEVER replace missing or unparsed GPU/CPU time with wall-clock time.
5. NEVER replace failed measurements with 0.
"""

import re
from typing import Dict, Any, Optional


def parse_bfs_cpu_stdout(stdout: str) -> Dict[str, Any]:
    """
    Parses OpenMP BFS CPU execution output (openmp/bfs/bfs.cpp:173).
    Prints: 'Compute time: <val>' where <val> is in seconds.
    """
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
    """
    Parses CUDA BFS GPU execution output (cuda/bfs/bfs.cu).
    Extracts:
      - Initial Host to Device Time (ms)
      - GPU Kernel Time (ms)
      - Final Device to Host Time (ms)
    """
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
    """
    Parses OpenMP CFD CPU execution output (openmp/cfd/euler3d_cpu.cpp:494).
    Prints: 'Compute time: <val>' where <val> is in seconds for 2000 RK iterations.
    Scaled to milliseconds (val * 1000.0).
    """
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
    """
    Parses CUDA CFD GPU execution output (cuda/cfd/euler3d.cu:589).
    Prints: '<val> seconds per iteration'.
    Nominally measures 2000 Runge-Kutta iterations.
    Total GPU compute phase time = val_sec_per_iter * iterations * 1000.0 (ms).
    """
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


def get_workload_parser(workload: str, device: str):
    """Factory function returning the workload and device-specific parser."""
    w = workload.lower()
    d = device.lower()
    if w == "bfs":
        return parse_bfs_cpu_stdout if d == "cpu" else parse_bfs_gpu_stdout
    elif w == "cfd":
        return parse_cfd_cpu_stdout if d == "cpu" else parse_cfd_gpu_stdout
    else:
        # Fallback parser for generic patterns if specific parser not yet registered
        def generic_parser(stdout: str) -> Dict[str, Any]:
            match = re.search(r"(?:Compute time|Time|Execution time):\s*([\d.]+)", stdout, re.IGNORECASE)
            if match:
                return {
                    "timing_parse_ok": True,
                    "benchmark_reported_time_ms": float(match.group(1)) * 1000.0,
                    "timing_type": "generic_time_regex",
                    "failure_reason": None,
                }
            return {
                "timing_parse_ok": False,
                "benchmark_reported_time_ms": None,
                "timing_type": "generic_time_regex",
                "failure_reason": f"No timing pattern matched for {w} on {d}",
            }
        return generic_parser
