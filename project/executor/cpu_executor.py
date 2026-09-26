"""
cpu_executor.py
===============
Executes Rodinia OpenMP workloads on CPU.
Runs each workload n_runs times, parses timing, returns statistics.
"""

import re
import subprocess
import time
import numpy as np
from typing import Optional


class CPUExecutor:

    def run(self, task_id: str, cfg: dict, n_runs: int = 6, timeout: int = 300) -> dict:
        """
        Run the CPU (OpenMP) variant of a workload n_runs times.

        Parameters
        ----------
        task_id : str   — workload identifier
        cfg     : dict  — from WORKLOAD_CONFIG[task_id]
        n_runs  : int   — number of repeated runs

        Returns
        -------
        dict with keys: workload, device, mean_s, std_s, min_s, max_s,
                        runs, speedup (cpu/cpu = 1.0)
        """
        dcfg    = cfg["cpu"]
        times   = []
        success = 0

        for run_id in range(1, n_runs + 1):
            t0 = time.perf_counter()
            try:
                result = subprocess.run(
                    dcfg["command"],
                    shell=True,
                    cwd=dcfg["cwd"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                )
                wall_time = time.perf_counter() - t0
                elapsed = self._parse_time(result.stdout, dcfg["time_regex"]) or wall_time
                if result.returncode == 0:
                    success += 1
            except (subprocess.TimeoutExpired, Exception) as e:
                elapsed = timeout
                wall_time = time.perf_counter() - t0

            times.append(elapsed)

        arr = np.array(times)
        return {
            "workload":   task_id,
            "device":     "cpu",
            "mean_s":     float(arr.mean()),
            "std_s":      float(arr.std()),
            "min_s":      float(arr.min()),
            "max_s":      float(arr.max()),
            "median_s":   float(np.median(arr)),
            "runs":       times,
            "n_success":  success,
            "speedup":    1.0,   # baseline = cpu
        }

    @staticmethod
    def _parse_time(stdout: str, pattern: str) -> Optional[float]:
        m = re.search(pattern, stdout)
        return float(m.group(1)) if m else None
