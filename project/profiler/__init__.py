"""
profiler module
"""
from profiler.workload_configs import WORKLOAD_CONFIGS, compute_file_size_mib, compute_file_size_bytes
from profiler.parsers import get_workload_parser, parse_bfs_cpu_stdout, parse_bfs_gpu_stdout, parse_cfd_cpu_stdout, parse_cfd_gpu_stdout
from profiler.measurement_utils import calculate_run_statistics, bytes_to_mib, seconds_to_ms
from profiler.workload_profiler import profile_workload_device, execute_single_run
