"""
profiler/workload_profiler.py
=============================
Workload-specific execution and profiling harness.
Strictly implements:
  - 1 warm-up run + >= 6 timed runs
  - Preserves stdout, stderr, return codes, and build status
  - Dispatches to workload-specific parsers
  - Appends every run as an immutable record in Layer 1
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from profiler.workload_configs import WORKLOAD_CONFIGS, compute_file_size_bytes, compute_file_size_mib
from profiler.parsers import get_workload_parser
from profiler.measurement_utils import calculate_run_statistics

DEFAULT_RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "dataset" / "raw_data" / "layer1_raw_executions.csv"


def execute_single_run(command: str, cwd: str, timeout_sec: int = 300) -> Dict[str, Any]:
    """
    Executes a shell command and captures complete wall time, returncode, stdout, and stderr.
    """
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec
        )
        wall_time_ms = (time.perf_counter() - start) * 1000.0
        return {
            "return_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "cpu_process_wall_time_ms": wall_time_ms,
            "execution_ok": (proc.returncode == 0)
        }
    except subprocess.TimeoutExpired as e:
        wall_time_ms = (time.perf_counter() - start) * 1000.0
        return {
            "return_code": -1,
            "stdout": e.stdout if e.stdout else "",
            "stderr": f"Timeout expired after {timeout_sec}s",
            "cpu_process_wall_time_ms": wall_time_ms,
            "execution_ok": False
        }
    except Exception as e:
        wall_time_ms = (time.perf_counter() - start) * 1000.0
        return {
            "return_code": -2,
            "stdout": "",
            "stderr": str(e),
            "cpu_process_wall_time_ms": wall_time_ms,
            "execution_ok": False
        }


def profile_workload_device(
    workload: str,
    device: str,
    n_timed_runs: int = 6,
    input_file_name: Optional[str] = None,
    output_csv: Path = DEFAULT_RAW_DATA_PATH
) -> List[Dict[str, Any]]:
    """
    Executes the standard repeated-run protocol for a given workload and device:
      1 warm-up run + n_timed_runs (>= 6) timed runs.
    Saves each execution immediately into Layer 1 raw dataset.
    """
    if workload not in WORKLOAD_CONFIGS:
        raise ValueError(f"Workload '{workload}' is not configured in WORKLOAD_CONFIGS.")

    cfg = WORKLOAD_CONFIGS[workload]
    inp = input_file_name or cfg.get("primary_input", "")
    inp_path = cfg["input_path_resolver"](inp) if "input_path_resolver" in cfg else inp

    size_bytes = compute_file_size_bytes(inp_path)
    size_mib = compute_file_size_mib(inp_path)

    cmd_template = cfg[f"{device}_command"]
    cmd = cmd_template.format(input_file=inp_path, temp_file=inp_path, power_file=inp_path, matrix_size=256)
    cwd = cfg[f"{device}_cwd"]

    parser_fn = get_workload_parser(workload, device)
    records: List[Dict[str, Any]] = []

    print(f"\n--- Profiling {workload.upper()} on {device.upper()} (1 warm-up + {n_timed_runs} timed runs) ---")

    # 1. Warm-up Run
    print(f"[{workload} {device}] Executing Warm-Up Run...")
    warm_res = execute_single_run(cmd, cwd)
    warm_parsed = parser_fn(warm_res["stdout"]) if warm_res["execution_ok"] else {
        "timing_parse_ok": False,
        "failure_reason": f"Execution failed (rc={warm_res['return_code']})"
    }

    warm_record: Dict[str, Any] = {
        "workload": workload,
        "input_file": inp,
        "input_size_bytes": size_bytes,
        "input_size_mib": size_mib,
        "device": device,
        "run_id": 0,
        "is_warmup": True,
        "benchmark_reported_time_ms": warm_parsed.get("benchmark_reported_time_ms"),
        "cpu_process_wall_time_ms": warm_res["cpu_process_wall_time_ms"],
        "gpu_phase_time_ms": warm_parsed.get("gpu_phase_time_ms"),
        "gpu_kernel_time_ms": warm_parsed.get("gpu_kernel_time_ms"),
        "host_to_device_time_ms": warm_parsed.get("host_to_device_time_ms"),
        "device_to_host_time_ms": warm_parsed.get("device_to_host_time_ms"),
        "gpu_total_path_time_ms": warm_parsed.get("gpu_total_path_time_ms"),
        "gpu_total_path_time_status": cfg.get("gpu_total_path_time_status", "unverified"),
        "timing_type": warm_parsed.get("timing_type", "unknown"),
        "build_status": "executed",
        "return_code": warm_res["return_code"],
        "timing_parse_ok": warm_parsed.get("timing_parse_ok", False),
        "failure_reason": warm_parsed.get("failure_reason") or (warm_res["stderr"][:200] if not warm_res["execution_ok"] else None),
        "stdout": warm_res["stdout"],
        "stderr": warm_res["stderr"],
        "gpu_device_index": 0 if device == "gpu" else None,
        "gpu_type": "Tesla T4" if device == "gpu" else None,
        "cpu_cores_available": 4,
    }
    records.append(warm_record)

    # 2. Timed Runs (1 to n_timed_runs)
    for run_id in range(1, n_timed_runs + 1):
        print(f"[{workload} {device}] Executing Run {run_id}/{n_timed_runs}...", end=" ", flush=True)
        res = execute_single_run(cmd, cwd)
        parsed = parser_fn(res["stdout"]) if res["execution_ok"] else {
            "timing_parse_ok": False,
            "failure_reason": f"Execution failed (rc={res['return_code']})"
        }

        rec: Dict[str, Any] = {
            "workload": workload,
            "input_file": inp,
            "input_size_bytes": size_bytes,
            "input_size_mib": size_mib,
            "device": device,
            "run_id": run_id,
            "is_warmup": False,
            "benchmark_reported_time_ms": parsed.get("benchmark_reported_time_ms"),
            "cpu_process_wall_time_ms": res["cpu_process_wall_time_ms"],
            "gpu_phase_time_ms": parsed.get("gpu_phase_time_ms"),
            "gpu_kernel_time_ms": parsed.get("gpu_kernel_time_ms"),
            "host_to_device_time_ms": parsed.get("host_to_device_time_ms"),
            "device_to_host_time_ms": parsed.get("device_to_host_time_ms"),
            "gpu_total_path_time_ms": parsed.get("gpu_total_path_time_ms"),
            "gpu_total_path_time_status": cfg.get("gpu_total_path_time_status", "unverified"),
            "timing_type": parsed.get("timing_type", "unknown"),
            "build_status": "executed",
            "return_code": res["return_code"],
            "timing_parse_ok": parsed.get("timing_parse_ok", False),
            "failure_reason": parsed.get("failure_reason") or (res["stderr"][:200] if not res["execution_ok"] else None),
            "stdout": res["stdout"],
            "stderr": res["stderr"],
            "gpu_device_index": 0 if device == "gpu" else None,
            "gpu_type": "Tesla T4" if device == "gpu" else None,
            "cpu_cores_available": 4,
        }
        records.append(rec)
        status_str = "OK" if rec["timing_parse_ok"] else f"FAILED ({rec['failure_reason']})"
        print(f"{status_str}")

    # Append to Layer 1 CSV
    append_records_to_layer1(records, output_csv)
    return records


def append_records_to_layer1(records: List[Dict[str, Any]], output_path: Path):
    """Appends records to the Layer 1 CSV dataset preserving all columns."""
    df_new = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(output_path, index=False)
    print(f"Layer 1 updated: {len(df_combined)} total execution records saved at {output_path}")
