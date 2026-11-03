"""
profiler/energy_estimator.py
============================
Energy estimation for CPU and GPU workload executions in Cloud HPC.

CRITICAL ARCHITECTURAL CONSTRAINTS (Requirement 14):
1. All energy values are ESTIMATES based on Thermal Design Power (TDP) models,
   NOT direct physical power meter measurements.
2. In production HPC clusters, IPMI, Intel RAPL, or NVIDIA NVML/nvidia-smi power sensors
   provide physical telemetry. On execution environments without direct sensor access,
   or where GPUs are simulated or unavailable, the TDP proxy model is utilized.
3. Every returned dictionary explicitly includes:
   - measurement_type: "ESTIMATED_NOT_MEASURED"
   - estimation_method: "tdp_model"
   - clear academic disclaimer.
"""

from typing import Dict, Any, Optional


class EnergyEstimator:
    """
    TDP-based Energy Estimator for Heterogeneous Computing Nodes.
    
    Formula:
      Energy (Joules) = Runtime (seconds) * Effective_Power (Watts)
      where Effective_Power = TDP (Watts) * Utilization_Factor
    """

    # Authoritative TDP Profiles (Thermal Design Power in Watts)
    TDP_PROFILES = {
        # CPU Profiles
        "cpu_generic_server": 65.0,    # Standard server CPU (e.g., Xeon/EPYC mid-range)
        "cpu_high_perf": 125.0,        # High-power HPC CPU socket
        "cpu_cloud_vm_4core": 45.0,    # 4 vCPU Cloud instance allocation
        # GPU Profiles
        "gpu_tesla_t4": 70.0,          # NVIDIA Tesla T4 (Cloud Inference & HPC, 70W max)
        "gpu_v100": 250.0,             # NVIDIA Tesla V100 PCIe (250W)
        "gpu_a100": 400.0,             # NVIDIA A100 SXM4 (400W)
        "gpu_rtx_3090": 350.0,         # High-end workstation GPU
    }

    # PCIe 3.0 / 4.0 Host-Device interconnect power and bandwidth proxies
    PCIE_TRANSFER_POWER_W = 15.0       # Average interconnect controller active power (Watts)
    PCIE_BANDWIDTH_MB_S = 12000.0      # Practical PCIe 3.0 x16 throughput (MB/s)

    def __init__(
        self,
        default_cpu_profile: str = "cpu_generic_server",
        default_gpu_profile: str = "gpu_tesla_t4"
    ):
        self.default_cpu_profile = default_cpu_profile
        self.default_gpu_profile = default_gpu_profile

    def get_tdp(self, device: str, hw_profile: Optional[str] = None) -> float:
        """Return Thermal Design Power in Watts for given device and hardware profile."""
        if hw_profile and hw_profile in self.TDP_PROFILES:
            return self.TDP_PROFILES[hw_profile]
        
        dev = device.lower()
        if "cpu" in dev:
            return self.TDP_PROFILES.get(self.default_cpu_profile, 65.0)
        elif "gpu" in dev:
            return self.TDP_PROFILES.get(self.default_gpu_profile, 70.0)
        return 65.0

    def estimate(
        self,
        workload: str,
        device: str,
        runtime_s: float,
        utilization_factor: float = 0.85,
        transfer_bytes_mb: float = 0.0,
        hw_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates TDP-proxy energy consumption for a workload task execution.
        
        Parameters:
        -----------
        workload : str - Workload identifier (e.g. 'cfd', 'bfs')
        device : str - 'cpu' or 'gpu'
        runtime_s : float - Execution duration in seconds
        utilization_factor : float - Estimated active load fraction (0.0 to 1.0)
        transfer_bytes_mb : float - Data transferred across host-device interconnect (for GPU)
        hw_profile : str, optional - Custom TDP profile key
        
        Returns:
        --------
        dict containing energy metrics and explicit attribution metadata.
        """
        device = device.lower()
        base_tdp = self.get_tdp(device, hw_profile)
        effective_power_w = base_tdp * max(0.1, min(1.0, utilization_factor))
        
        compute_energy_j = runtime_s * effective_power_w

        transfer_time_s = 0.0
        transfer_energy_j = 0.0
        if device == "gpu" and transfer_bytes_mb > 0:
            transfer_time_s = transfer_bytes_mb / self.PCIE_BANDWIDTH_MB_S
            transfer_energy_j = transfer_time_s * self.PCIE_TRANSFER_POWER_W

        total_energy_j = compute_energy_j + transfer_energy_j

        return {
            "workload": workload,
            "device": device,
            "runtime_s": round(float(runtime_s), 5),
            "tdp_watts": round(float(base_tdp), 2),
            "effective_power_watts": round(float(effective_power_w), 2),
            "compute_energy_joules": round(float(compute_energy_j), 4),
            "transfer_energy_joules": round(float(transfer_energy_j), 4),
            "total_energy_joules": round(float(total_energy_j), 4),
            "energy_kwh": round(float(total_energy_j / 3600000.0), 8),
            "hw_profile": hw_profile or (self.default_gpu_profile if device == "gpu" else self.default_cpu_profile),
            "measurement_type": "ESTIMATED_NOT_MEASURED",
            "estimation_method": "tdp_model",
            "is_proxy": True,
            "disclaimer": (
                "ACADEMIC NOTICE: Energy metrics are computed via calibrated Thermal Design Power (TDP) "
                "models and do not represent direct physical shunt resistor or RAPL/NVML power telemetry."
            ),
        }

    def compare_devices(
        self,
        workload: str,
        cpu_runtime_s: float,
        gpu_runtime_s: float,
        transfer_bytes_mb: float = 0.0,
        cpu_util: float = 0.90,
        gpu_util: float = 0.85
    ) -> Dict[str, Any]:
        """Compares energy consumption between CPU and GPU for the same workload execution."""
        cpu_est = self.estimate(workload, "cpu", cpu_runtime_s, utilization_factor=cpu_util)
        gpu_est = self.estimate(workload, "gpu", gpu_runtime_s, utilization_factor=gpu_util, transfer_bytes_mb=transfer_bytes_mb)

        cpu_j = cpu_est["total_energy_joules"]
        gpu_j = gpu_est["total_energy_joules"]

        energy_savings_pct = round(((cpu_j - gpu_j) / max(cpu_j, 1e-6)) * 100.0, 2)
        more_efficient = "gpu" if gpu_j < cpu_j else "cpu"
        energy_ratio = round(cpu_j / max(gpu_j, 1e-6), 3)

        return {
            "workload": workload,
            "cpu_estimate": cpu_est,
            "gpu_estimate": gpu_est,
            "more_efficient_device": more_efficient,
            "energy_ratio_cpu_over_gpu": energy_ratio,
            "energy_savings_percent_with_gpu": energy_savings_pct,
            "tradeoff_analysis": (
                f"GPU uses {energy_ratio}x less energy ({energy_savings_pct}% savings)"
                if more_efficient == "gpu"
                else f"CPU uses {round(gpu_j / max(cpu_j, 1e-6), 2)}x less energy than GPU"
            )
        }

