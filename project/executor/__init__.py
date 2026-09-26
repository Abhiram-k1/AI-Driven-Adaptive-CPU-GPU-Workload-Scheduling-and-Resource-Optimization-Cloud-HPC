"""
executor package
================
CPU and GPU workload execution wrappers.
"""
from executor.cpu_executor import CPUExecutor
from executor.gpu_executor import GPUExecutor

__all__ = ["CPUExecutor", "GPUExecutor"]
