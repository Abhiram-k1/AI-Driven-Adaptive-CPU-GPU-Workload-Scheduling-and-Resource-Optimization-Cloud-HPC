"""
profiler/resource_monitor.py
============================
Live system resource monitoring using psutil and NVIDIA Management Library (NVML / nvidia-smi).

Features:
  - CPU utilization (overall and per-core)
  - System memory (total, used, available, percentage)
  - GPU telemetry probe (utilization, memory used, total memory) via nvidia-smi
  - Graceful fallback: when no NVIDIA GPU or CUDA driver is detected,
    returns status='GPU_MONITORING_UNAVAILABLE' and never fabricates mock numbers.
  - Background asynchronous sampling thread with summary statistics.
"""

import time
import subprocess
import threading
from typing import Dict, Any, List, Optional
import psutil


class ResourceMonitor:
    """
    Live Hardware Resource Monitor for Cloud HPC scheduling.
    Collects real-time CPU, RAM, and GPU telemetry.
    """

    def __init__(self, sample_interval_s: float = 1.0):
        self.sample_interval_s = sample_interval_s
        self.history: List[Dict[str, Any]] = []
        self._is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Probe GPU capability once at init
        self.gpu_detected = self._probe_gpu_hardware()

    def _probe_gpu_hardware(self) -> bool:
        """Check if an NVIDIA GPU with working nvidia-smi is accessible."""
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2
            )
            return res.returncode == 0 and len(res.stdout.strip()) > 0
        except (FileNotFoundError, subprocess.SubprocessError, Exception):
            return False

    def get_cpu_snapshot(self) -> Dict[str, Any]:
        """Collect current CPU and host memory metrics."""
        cpu_percent = psutil.cpu_percent(interval=None)
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        cpu_freq = psutil.cpu_freq()

        return {
            "timestamp": time.time(),
            "cpu_percent": float(cpu_percent),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "per_cpu_percent": [float(p) for p in per_cpu],
            "cpu_freq_current_mhz": round(cpu_freq.current, 1) if cpu_freq else None,
            "memory_total_gb": round(mem.total / (1024.0 ** 3), 2),
            "memory_used_gb": round(mem.used / (1024.0 ** 3), 2),
            "memory_available_gb": round(mem.available / (1024.0 ** 3), 2),
            "memory_percent": float(mem.percent),
        }

    def get_gpu_snapshot(self) -> Dict[str, Any]:
        """
        Query NVIDIA GPU utilization and memory via nvidia-smi.
        If GPU is unavailable, returns explicit UNAVAILABLE status.
        Never fabricates telemetry.
        """
        if not self.gpu_detected:
            return {
                "gpu_available": False,
                "status": "GPU_MONITORING_UNAVAILABLE",
                "message": "No NVIDIA GPU detected or nvidia-smi not in PATH on this host.",
                "gpu_utilization_percent": None,
                "gpu_memory_used_mb": None,
                "gpu_memory_total_mb": None,
                "gpu_memory_percent": None,
                "gpu_temperature_c": None,
            }

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                gpu_util = float(parts[0])
                mem_used = float(parts[1])
                mem_total = float(parts[2])
                temp = float(parts[3])
                mem_pct = round((mem_used / max(mem_total, 1.0)) * 100.0, 2)

                return {
                    "gpu_available": True,
                    "status": "ONLINE",
                    "gpu_utilization_percent": gpu_util,
                    "gpu_memory_used_mb": mem_used,
                    "gpu_memory_total_mb": mem_total,
                    "gpu_memory_percent": mem_pct,
                    "gpu_temperature_c": temp,
                }
            else:
                return {
                    "gpu_available": False,
                    "status": "GPU_QUERY_FAILED",
                    "error": res.stderr.strip() or "Empty stdout",
                }
        except Exception as e:
            return {
                "gpu_available": False,
                "status": "GPU_MONITORING_ERROR",
                "error": str(e),
            }

    def get_system_snapshot(self) -> Dict[str, Any]:
        """Get combined CPU, RAM, and GPU status snapshot."""
        cpu_data = self.get_cpu_snapshot()
        gpu_data = self.get_gpu_snapshot()
        return {
            "timestamp": cpu_data["timestamp"],
            "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cpu_data["timestamp"])),
            "cpu": cpu_data,
            "gpu": gpu_data,
        }

    def start_background_monitor(self) -> None:
        """Start non-blocking background telemetry sampling thread."""
        with self._lock:
            if self._is_monitoring:
                return
            self._is_monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while self._is_monitoring:
            snap = self.get_system_snapshot()
            with self._lock:
                self.history.append(snap)
                # Keep rolling window of last 500 samples
                if len(self.history) > 500:
                    self.history.pop(0)
            time.sleep(self.sample_interval_s)

    def stop_background_monitor(self) -> None:
        """Stop background sampling thread."""
        with self._lock:
            self._is_monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return captured historical snapshots."""
        with self._lock:
            return list(self.history)

    def summarize_monitoring_period(self) -> Dict[str, Any]:
        """Compute statistical summary over collected history."""
        with self._lock:
            history_copy = list(self.history)

        if not history_copy:
            current = self.get_system_snapshot()
            return {
                "sample_count": 0,
                "cpu_mean_percent": current["cpu"]["cpu_percent"],
                "cpu_max_percent": current["cpu"]["cpu_percent"],
                "memory_mean_percent": current["cpu"]["memory_percent"],
                "gpu_available": current["gpu"]["gpu_available"],
            }

        cpu_percents = [s["cpu"]["cpu_percent"] for s in history_copy]
        mem_percents = [s["cpu"]["memory_percent"] for s in history_copy]
        
        gpu_avail = history_copy[-1]["gpu"]["gpu_available"]
        gpu_mean_util = None
        if gpu_avail:
            gpu_utils = [s["gpu"]["gpu_utilization_percent"] for s in history_copy if s["gpu"].get("gpu_utilization_percent") is not None]
            if gpu_utils:
                gpu_mean_util = round(sum(gpu_utils) / len(gpu_utils), 2)

        return {
            "sample_count": len(history_copy),
            "cpu_mean_percent": round(sum(cpu_percents) / len(cpu_percents), 2),
            "cpu_max_percent": round(max(cpu_percents), 2),
            "cpu_min_percent": round(min(cpu_percents), 2),
            "memory_mean_percent": round(sum(mem_percents) / len(mem_percents), 2),
            "gpu_available": gpu_avail,
            "gpu_mean_utilization": gpu_mean_util,
        }

