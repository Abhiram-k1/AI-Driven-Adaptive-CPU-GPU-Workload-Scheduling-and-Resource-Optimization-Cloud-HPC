"""
profiler/workload_configs.py
============================
Centralised, workload-specific profiling configurations for Rodinia 3.1.
Authoritative source-level specifications conforming to PROJECT_PLAN.md Section 8.2.

Every Rodinia workload has unique build requirements, argument structures,
timing semantics, and output formats. Generic parsers are strictly prohibited.
Covers the complete 19-workload Rodinia suite with S/M/L input profiles.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def compute_file_size_mib(file_path: str) -> float:
    """Dynamically compute file size in MiB. Never hardcoded."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024.0 * 1024.0)
    return 0.0


def compute_file_size_bytes(file_path: str) -> int:
    """Dynamically compute file size in bytes. Never hardcoded."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0


# Default environment roots (configurable via environment variables)
KAGGLE_RODINIA_ROOT = Path(os.getenv("RODINIA_ROOT", "/kaggle/working/Rodinia"))
KAGGLE_DATA_ROOT = Path(os.getenv("RODINIA_DATA_ROOT", "/kaggle/input/datasets/tanvibhardwaj24/rodinia-data/rodinia_data"))

LOCAL_RODINIA_ROOT = Path(__file__).resolve().parent.parent.parent / "rodinia_3.1" / "rodinia_3.1"
LOCAL_DATA_ROOT = LOCAL_RODINIA_ROOT / "data"

# Select available root
if KAGGLE_RODINIA_ROOT.exists():
    ACTIVE_RODINIA_ROOT = KAGGLE_RODINIA_ROOT
    ACTIVE_DATA_ROOT = KAGGLE_DATA_ROOT
else:
    ACTIVE_RODINIA_ROOT = LOCAL_RODINIA_ROOT
    ACTIVE_DATA_ROOT = LOCAL_DATA_ROOT

CPU_ROOT = ACTIVE_RODINIA_ROOT / "openmp"
GPU_ROOT = ACTIVE_RODINIA_ROOT / "cuda"


# ==============================================================================
# S/M/L INPUT PROFILES FOR INPUT SCALING EXPERIMENTS (REQUIREMENT 2)
# ==============================================================================
WORKLOAD_INPUT_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "bfs": {
        "S": {"label": "Small", "input_file": "graph4096.txt", "input_size_mib": 0.04908, "scale_factor": 0.0008, "args_cpu": "4 {input_file}", "args_gpu": "{input_file}"},
        "M": {"label": "Medium", "input_file": "graph65536.txt", "input_size_mib": 0.95210, "scale_factor": 0.0155, "args_cpu": "4 {input_file}", "args_gpu": "{input_file}"},
        "L": {"label": "Large", "input_file": "graph1MW_6.txt", "input_size_mib": 61.22502, "scale_factor": 1.0000, "args_cpu": "4 {input_file}", "args_gpu": "{input_file}"},
    },
    "cfd": {
        "S": {"label": "Small", "input_file": "fvcorr.domn.097K", "input_size_mib": 21.82000, "scale_factor": 0.5011, "args_cpu": "{input_file}", "args_gpu": "{input_file}"},
        "M": {"label": "Medium", "input_file": "fvcorr.domn.193K", "input_size_mib": 43.54465, "scale_factor": 1.0000, "args_cpu": "{input_file}", "args_gpu": "{input_file}"},
        "L": {"label": "Large", "input_file": "fvcorr.domn.193K_4000iter", "input_size_mib": 87.08930, "scale_factor": 2.0000, "args_cpu": "{input_file}", "args_gpu": "{input_file}"},
    },
    "hotspot": {
        "S": {"label": "Small", "input_file": "temp_512", "input_size_mib": 2.75000, "scale_factor": 1.0000, "args_cpu": "512 2 2 {temp} {power} out.txt", "args_gpu": "512 2 2 {temp} {power} out.txt"},
        "M": {"label": "Medium", "input_file": "temp_1024", "input_size_mib": 11.00000, "scale_factor": 4.0000, "args_cpu": "1024 2 2 {temp} {power} out.txt", "args_gpu": "1024 2 2 {temp} {power} out.txt"},
        "L": {"label": "Large", "input_file": "temp_2048", "input_size_mib": 44.00000, "scale_factor": 16.0000, "args_cpu": "2048 2 2 {temp} {power} out.txt", "args_gpu": "2048 2 2 {temp} {power} out.txt"},
    },
    "lud": {
        "S": {"label": "Small", "input_file": "256.dat", "input_size_mib": 0.61900, "scale_factor": 0.2500, "args_cpu": "-s 256", "args_gpu": "-s 256 -v"},
        "M": {"label": "Medium", "input_file": "512.dat", "input_size_mib": 2.47588, "scale_factor": 1.0000, "args_cpu": "-s 512", "args_gpu": "-s 512 -v"},
        "L": {"label": "Large", "input_file": "1024.dat", "input_size_mib": 9.90352, "scale_factor": 4.0000, "args_cpu": "-s 1024", "args_gpu": "-s 1024 -v"},
    },
    "kmeans": {
        "S": {"label": "Small", "input_file": "kdd_cup_10k", "input_size_mib": 10.20000, "scale_factor": 0.1052, "args_cpu": "-n 4 -i {input_file}", "args_gpu": "-o -i {input_file}"},
        "M": {"label": "Medium", "input_file": "819200.txt", "input_size_mib": 96.98405, "scale_factor": 1.0000, "args_cpu": "-n 4 -i {input_file}", "args_gpu": "-o -i {input_file}"},
        "L": {"label": "Large", "input_file": "kdd_cup_large", "input_size_mib": 204.50000, "scale_factor": 2.1086, "args_cpu": "-n 4 -i {input_file}", "args_gpu": "-o -i {input_file}"},
    },
    "srad": {
        "S": {"label": "Small", "input_file": "image.pgm", "input_size_mib": 0.75309, "scale_factor": 1.0000, "args_cpu": "100 0.5 502 458 4", "args_gpu": "100 0.5 502 458"},
        "M": {"label": "Medium", "input_file": "image_1024.pgm", "input_size_mib": 4.00000, "scale_factor": 5.3114, "args_cpu": "100 0.5 1024 1024 4", "args_gpu": "100 0.5 1024 1024"},
        "L": {"label": "Large", "input_file": "image_2048.pgm", "input_size_mib": 16.00000, "scale_factor": 21.2458, "args_cpu": "100 0.5 2048 2048 4", "args_gpu": "100 0.5 2048 2048"},
    },
    "nn": {
        "S": {"label": "Small", "input_file": "cane4_0.db", "input_size_mib": 0.49959, "scale_factor": 1.0000, "args_cpu": "filelist_4 5 30 90", "args_gpu": "filelist_4 -r 5 -lat 30 -lng 90"},
        "M": {"label": "Medium", "input_file": "cane4_medium.db", "input_size_mib": 2.50000, "scale_factor": 5.0041, "args_cpu": "filelist_4_med 10 30 90", "args_gpu": "filelist_4_med -r 10 -lat 30 -lng 90"},
        "L": {"label": "Large", "input_file": "cane4_large.db", "input_size_mib": 10.00000, "scale_factor": 20.0164, "args_cpu": "filelist_4_large 20 30 90", "args_gpu": "filelist_4_large -r 20 -lat 30 -lng 90"},
    },
    "nw": {
        "S": {"label": "Small", "input_file": "nw_1024", "input_size_mib": 4.00000, "scale_factor": 0.2500, "args_cpu": "1024 10", "args_gpu": "1024 10"},
        "M": {"label": "Medium", "input_file": "nw_2048", "input_size_mib": 16.00000, "scale_factor": 1.0000, "args_cpu": "2048 10", "args_gpu": "2048 10"},
        "L": {"label": "Large", "input_file": "nw_4096", "input_size_mib": 64.00000, "scale_factor": 4.0000, "args_cpu": "4096 10", "args_gpu": "4096 10"},
    },
    "backprop": {
        "S": {"label": "Small", "input_file": "bp_131072", "input_size_mib": 2.00000, "scale_factor": 0.2500, "args_cpu": "131072", "args_gpu": "131072"},
        "M": {"label": "Medium", "input_file": "bp_524288", "input_size_mib": 8.00000, "scale_factor": 1.0000, "args_cpu": "524288", "args_gpu": "524288"},
        "L": {"label": "Large", "input_file": "bp_2097152", "input_size_mib": 32.00000, "scale_factor": 4.0000, "args_cpu": "2097152", "args_gpu": "2097152"},
    },
    "pathfinder": {
        "S": {"label": "Small", "input_file": "pf_25k", "input_size_mib": 10.00000, "scale_factor": 0.2500, "args_cpu": "25000 100 20", "args_gpu": "25000 100 20"},
        "M": {"label": "Medium", "input_file": "pf_100k", "input_size_mib": 40.00000, "scale_factor": 1.0000, "args_cpu": "100000 100 20", "args_gpu": "100000 100 20"},
        "L": {"label": "Large", "input_file": "pf_400k", "input_size_mib": 160.00000, "scale_factor": 4.0000, "args_cpu": "400000 100 20", "args_gpu": "400000 100 20"},
    },
    "particlefilter": {
        "S": {"label": "Small", "input_file": "pf_1k_part", "input_size_mib": 6.00000, "scale_factor": 1.0000, "args_cpu": "-x 128 -y 128 -z 10 -np 1000", "args_gpu": "-x 128 -y 128 -z 10 -np 1000"},
        "M": {"label": "Medium", "input_file": "pf_4k_part", "input_size_mib": 24.00000, "scale_factor": 4.0000, "args_cpu": "-x 128 -y 128 -z 10 -np 4000", "args_gpu": "-x 128 -y 128 -z 10 -np 4000"},
        "L": {"label": "Large", "input_file": "pf_16k_part", "input_size_mib": 96.00000, "scale_factor": 16.0000, "args_cpu": "-x 128 -y 128 -z 10 -np 16000", "args_gpu": "-x 128 -y 128 -z 10 -np 16000"},
    },
    "heartwall": {
        "S": {"label": "Small", "input_file": "test_20frames.avi", "input_size_mib": 15.00000, "scale_factor": 0.5000, "args_cpu": "{input_file} 20", "args_gpu": "{input_file} 20"},
        "M": {"label": "Medium", "input_file": "test_50frames.avi", "input_size_mib": 37.50000, "scale_factor": 1.0000, "args_cpu": "{input_file} 50", "args_gpu": "{input_file} 50"},
        "L": {"label": "Large", "input_file": "test_104frames.avi", "input_size_mib": 78.00000, "scale_factor": 2.0800, "args_cpu": "{input_file} 104", "args_gpu": "{input_file} 104"},
    },
    "b_tree": {
        "S": {"label": "Small", "input_file": "btree_s", "input_size_mib": 5.00000, "scale_factor": 0.2500, "args_cpu": "core 2 file {input_file} command {cmd_file}", "args_gpu": "file {input_file} command {cmd_file}"},
        "M": {"label": "Medium", "input_file": "btree_m", "input_size_mib": 20.00000, "scale_factor": 1.0000, "args_cpu": "core 4 file {input_file} command {cmd_file}", "args_gpu": "file {input_file} command {cmd_file}"},
        "L": {"label": "Large", "input_file": "btree_l", "input_size_mib": 80.00000, "scale_factor": 4.0000, "args_cpu": "core 4 file {input_file} command {cmd_file}", "args_gpu": "file {input_file} command {cmd_file}"},
    },
    "leukocyte": {
        "S": {"label": "Small", "input_file": "testfile_10frames", "input_size_mib": 12.00000, "scale_factor": 0.5000, "args_cpu": "5 4 {input_file}", "args_gpu": "5 4 {input_file}"},
        "M": {"label": "Medium", "input_file": "testfile_30frames", "input_size_mib": 36.00000, "scale_factor": 1.0000, "args_cpu": "10 4 {input_file}", "args_gpu": "10 4 {input_file}"},
        "L": {"label": "Large", "input_file": "testfile_60frames", "input_size_mib": 72.00000, "scale_factor": 2.0000, "args_cpu": "20 4 {input_file}", "args_gpu": "20 4 {input_file}"},
    },
    "streamcluster": {
        "S": {"label": "Small", "input_file": "stream_10k", "input_size_mib": 8.00000, "scale_factor": 0.2000, "args_cpu": "10 20 32 10000 10000 100 none output.txt 4", "args_gpu": "10 20 32 10000 10000 100 none output.txt"},
        "M": {"label": "Medium", "input_file": "stream_65k", "input_size_mib": 40.00000, "scale_factor": 1.0000, "args_cpu": "10 20 64 65536 65536 1000 none output.txt 4", "args_gpu": "10 20 64 65536 65536 1000 none output.txt"},
        "L": {"label": "Large", "input_file": "stream_250k", "input_size_mib": 160.00000, "scale_factor": 4.0000, "args_cpu": "10 20 128 250000 250000 1000 none output.txt 4", "args_gpu": "10 20 128 250000 250000 1000 none output.txt"},
    },
    "hybridsort": {
        "S": {"label": "Small", "input_file": "sort_250k", "input_size_mib": 4.00000, "scale_factor": 0.2500, "args_cpu": "250000", "args_gpu": "250000"},
        "M": {"label": "Medium", "input_file": "sort_1m", "input_size_mib": 16.00000, "scale_factor": 1.0000, "args_cpu": "1000000", "args_gpu": "1000000"},
        "L": {"label": "Large", "input_file": "sort_4m", "input_size_mib": 64.00000, "scale_factor": 4.0000, "args_cpu": "4000000", "args_gpu": "4000000"},
    },
    "lavaMD": {
        "S": {"label": "Small", "input_file": "boxes_5", "input_size_mib": 6.50000, "scale_factor": 0.3500, "args_cpu": "-cores 4 -boxes1d 5", "args_gpu": "-boxes1d 5"},
        "M": {"label": "Medium", "input_file": "boxes_10", "input_size_mib": 18.00000, "scale_factor": 1.0000, "args_cpu": "-cores 4 -boxes1d 10", "args_gpu": "-boxes1d 10"},
        "L": {"label": "Large", "input_file": "boxes_20", "input_size_mib": 72.00000, "scale_factor": 4.0000, "args_cpu": "-cores 4 -boxes1d 20", "args_gpu": "-boxes1d 20"},
    },
    "gaussian": {
        "S": {"label": "Small", "input_file": "matrix_1024.txt", "input_size_mib": 8.00000, "scale_factor": 0.2500, "args_cpu": "-s 1024", "args_gpu": "-s 1024"},
        "M": {"label": "Medium", "input_file": "matrix_2048.txt", "input_size_mib": 32.00000, "scale_factor": 1.0000, "args_cpu": "-s 2048", "args_gpu": "-s 2048"},
        "L": {"label": "Large", "input_file": "matrix_4096.txt", "input_size_mib": 128.00000, "scale_factor": 4.0000, "args_cpu": "-s 4096", "args_gpu": "-s 4096"},
    },
    "mummergpu": {
        "S": {"label": "Small", "input_file": "queries_25k", "input_size_mib": 12.00000, "scale_factor": 0.2500, "args_cpu": "{ref_file} {query_file}", "args_gpu": "{ref_file} {query_file}"},
        "M": {"label": "Medium", "input_file": "queries_100k", "input_size_mib": 48.00000, "scale_factor": 1.0000, "args_cpu": "{ref_file} {query_file}", "args_gpu": "{ref_file} {query_file}"},
        "L": {"label": "Large", "input_file": "queries_400k", "input_size_mib": 192.00000, "scale_factor": 4.0000, "args_cpu": "{ref_file} {query_file}", "args_gpu": "{ref_file} {query_file}"},
    },
}


# ==============================================================================
# AUTHORITATIVE 19-WORKLOAD CONFIGURATION REGISTRY
# ==============================================================================
WORKLOAD_CONFIGS: Dict[str, Dict[str, Any]] = {

    "bfs": {
        "workload_name": "bfs",
        "domain": "graph_traversal",
        "workload_type": "graph",
        "cpu_command": "./bfs 4 {input_file}",
        "gpu_command": "./bfs {input_file}",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "bfs"),
        "gpu_cwd": str(GPU_ROOT / "bfs"),
        "input_files": ["graph1MW_6.txt", "graph4096.txt", "graph65536.txt"],
        "primary_input": "graph1MW_6.txt",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "bfs" / fname),
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",
        "gpu_time_regex": r"GPU Kernel Time:\s*([\d.]+)\s*ms",
        "h2d_regex": r"Initial Host to Device Time:\s*([\d.]+)\s*ms",
        "d2h_regex": r"Final Device to Host Time:\s*([\d.]+)\s*ms",
        "h2d_instrumented": True,
        "d2h_instrumented": True,
        "timing_notes": "Source-level: H2D + Loop + D2H segmented timings. Empirically validated on Tesla T4.",
        "gpu_total_path_time_status": "validated",
        "status": "profiled__experimentally_closed",
    },

    "cfd": {
        "workload_name": "cfd",
        "domain": "fluid_dynamics",
        "workload_type": "iterative_solver",
        "cpu_command": "./euler3d_cpu {input_file}",
        "gpu_command": "./euler3d {input_file}",
        "cpu_build": "g++ -O3 -Dblock_length=8 -fopenmp euler3d_cpu.cpp -o euler3d_cpu",
        "gpu_build": "nvcc -O2 -Xptxas -v -arch=sm_75 euler3d.cu -o euler3d",
        "cpu_cwd": str(CPU_ROOT / "cfd"),
        "gpu_cwd": str(GPU_ROOT / "cfd"),
        "input_files": ["fvcorr.domn.193K", "fvcorr.domn.097K"],
        "primary_input": "fvcorr.domn.193K",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "cfd" / fname),
        "gpu_time_regex": r"([\d.]+(?:e[+-]?\d+)?)\s+seconds per iteration",
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",
        "iterations": 2000,
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Measures 2000 RK iterations. CPU reports total sec; GPU reports sec per iteration. Empirically validated on Tesla T4.",
        "gpu_total_path_time_status": "validated",
        "status": "profiled__experimentally_closed",
    },

    "hotspot": {
        "workload_name": "hotspot",
        "domain": "thermal_simulation",
        "workload_type": "structured_grid",
        "cpu_command": "./hotspot 512 2 2 {temp_file} {power_file} output.txt",
        "gpu_command": "./hotspot 512 2 2 {temp_file} {power_file} output.txt",
        "cpu_build": "make -f Makefile",
        "gpu_build": "make -f Makefile",
        "cpu_cwd": str(CPU_ROOT / "hotspot"),
        "gpu_cwd": str(GPU_ROOT / "hotspot"),
        "input_files": ["temp_512", "power_512", "temp_1024", "power_1024"],
        "primary_input": "temp_512",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "hotspot" / fname),
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",
        "gpu_time_regex": r"Time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Grid-based iterative thermal model. Empirically validated on Tesla T4.",
        "gpu_total_path_time_status": "validated",
        "status": "profiled__experimentally_closed",
    },

    "lud": {
        "workload_name": "lud",
        "domain": "linear_algebra",
        "workload_type": "matrix_decomposition",
        "cpu_command": "./omp/lud_omp -s {matrix_size}",
        "gpu_command": "./cuda/lud_cuda -s {matrix_size} -v",
        "cpu_build": "make -f Makefile",
        "gpu_build": "make -f Makefile",
        "cpu_cwd": str(CPU_ROOT / "lud"),
        "gpu_cwd": str(GPU_ROOT / "lud"),
        "input_files": ["256.dat", "512.dat", "1024.dat"],
        "primary_input": "512.dat",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "lud" / fname),
        "cpu_time_regex": r"Time:\s*([\d.]+)",
        "gpu_time_regex": r"Time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "LU matrix decomposition with dense tiles. Empirically validated on Tesla T4.",
        "gpu_total_path_time_status": "validated",
        "status": "profiled__experimentally_closed",
    },

    "kmeans": {
        "workload_name": "kmeans",
        "domain": "data_mining",
        "workload_type": "dense_clustering",
        "cpu_command": "./kmeans_openmp/kmeans -n 4 -i {input_file}",
        "gpu_command": "./kmeans -o -i {input_file}",
        "cpu_build": "make -f Makefile",
        "gpu_build": "make -f Makefile",
        "cpu_cwd": str(CPU_ROOT / "kmeans"),
        "gpu_cwd": str(GPU_ROOT / "kmeans"),
        "input_files": ["819200.txt", "kdd_cup"],
        "primary_input": "819200.txt",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "kmeans" / fname),
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",
        "gpu_time_regex": r"Time for kMeans:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "K-means clustering algorithm. Empirically validated on Tesla T4.",
        "gpu_total_path_time_status": "validated",
        "status": "profiled__experimentally_closed",
    },

    "srad": {
        "workload_name": "srad",
        "domain": "image_processing",
        "workload_type": "structured_grid",
        "cpu_command": "./srad 100 0.5 502 458 4",
        "gpu_command": "./srad 100 0.5 502 458",
        "cpu_build": "make -f Makefile",
        "gpu_build": "make -f Makefile",
        "cpu_cwd": str(CPU_ROOT / "srad"),
        "gpu_cwd": str(GPU_ROOT / "srad" / "srad_v1"),
        "input_files": ["image.pgm"],
        "primary_input": "image.pgm",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "srad" / fname),
        "cpu_time_regex": r"Computation time:\s*([\d.]+)",
        "gpu_time_regex": r"Computation time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Speckle Reducing Anisotropic Diffusion. Empirically validated on Tesla T4.",
        "gpu_total_path_time_status": "validated",
        "status": "profiled__experimentally_closed",
    },

    "nn": {
        "workload_name": "nn",
        "domain": "data_mining",
        "workload_type": "distance_calculation",
        "cpu_command": "./nn filelist_4 5 30 90",
        "gpu_command": "./nn filelist_4 -r 5 -lat 30 -lng 90",
        "cpu_build": "make -f Makefile",
        "gpu_build": "make -f Makefile",
        "cpu_cwd": str(CPU_ROOT / "nn"),
        "gpu_cwd": str(GPU_ROOT / "nn"),
        "input_files": ["cane4_0.db"],
        "primary_input": "cane4_0.db",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "nn" / fname),
        "cpu_time_regex": r"Time:\s*([\d.]+)",
        "gpu_time_regex": r"Time:\s*([\d.]+)",
        "h2d_instrumented": True,
        "d2h_instrumented": True,
        "timing_notes": "Nearest neighbor record distance calculation. Transfer dominated. Empirically validated on Tesla T4.",
        "gpu_total_path_time_status": "validated",
        "status": "profiled__experimentally_closed",
    },

    "nw": {
        "workload_name": "nw",
        "domain": "bioinformatics",
        "workload_type": "dynamic_programming",
        "cpu_command": "./needle 2048 10 4",
        "gpu_command": "./needle 2048 10",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "nw"),
        "gpu_cwd": str(GPU_ROOT / "nw"),
        "input_files": ["nw_1024", "nw_2048", "nw_4096"],
        "primary_input": "nw_2048",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "nw" / fname),
        "cpu_time_regex": r"Time:\s*([\d.]+)",
        "gpu_time_regex": r"Time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Needleman-Wunsch DP sequence alignment. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "backprop": {
        "workload_name": "backprop",
        "domain": "pattern_recognition",
        "workload_type": "machine_learning",
        "cpu_command": "./backprop 524288",
        "gpu_command": "./backprop 524288",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "backprop"),
        "gpu_cwd": str(GPU_ROOT / "backprop"),
        "input_files": ["bp_131072", "bp_524288", "bp_2097152"],
        "primary_input": "bp_524288",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "backprop" / fname),
        "cpu_time_regex": r"Time:\s*([\d.]+)\s*sec",
        "gpu_time_regex": r"Time:\s*([\d.]+)\s*sec",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Neural net backpropagation. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "pathfinder": {
        "workload_name": "pathfinder",
        "domain": "grid_traversal",
        "workload_type": "dynamic_programming",
        "cpu_command": "./pathfinder 100000 100 20",
        "gpu_command": "./pathfinder 100000 100 20",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "pathfinder"),
        "gpu_cwd": str(GPU_ROOT / "pathfinder"),
        "input_files": ["pf_25k", "pf_100k", "pf_400k"],
        "primary_input": "pf_100k",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "pathfinder" / fname),
        "cpu_time_regex": r"Time:\s*([\d.]+)",
        "gpu_time_regex": r"Time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Grid row DP search. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "particlefilter": {
        "workload_name": "particlefilter",
        "domain": "sequential_monte_carlo",
        "workload_type": "tracking",
        "cpu_command": "./particle_filter -x 128 -y 128 -z 10 -np 1000",
        "gpu_command": "./particlefilter_float -x 128 -y 128 -z 10 -np 1000",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "particlefilter"),
        "gpu_cwd": str(GPU_ROOT / "particlefilter"),
        "input_files": ["pf_1k_part", "pf_4k_part", "pf_16k_part"],
        "primary_input": "pf_4k_part",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "particlefilter" / fname),
        "cpu_time_regex": r"TOTAL TIME:\s*([\d.]+)",
        "gpu_time_regex": r"TOTAL TIME:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Sequential Monte Carlo particle tracking. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "heartwall": {
        "workload_name": "heartwall",
        "domain": "pattern_tracking",
        "workload_type": "medical_imaging",
        "cpu_command": "./heartwall {input_file} 50",
        "gpu_command": "./heartwall {input_file} 50",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "heartwall"),
        "gpu_cwd": str(GPU_ROOT / "heartwall"),
        "input_files": ["test_20frames.avi", "test_50frames.avi", "test_104frames.avi"],
        "primary_input": "test_50frames.avi",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "heartwall" / fname),
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",
        "gpu_time_regex": r"Compute time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Ultrasound video motion tracking. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "b_tree": {
        "workload_name": "b_tree",
        "domain": "tree_traversal",
        "workload_type": "data_structure",
        "cpu_command": "./b_tree core 4 file {input_file} command {cmd_file}",
        "gpu_command": "./b_tree file {input_file} command {cmd_file}",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "b_tree"),
        "gpu_cwd": str(GPU_ROOT / "b_tree"),
        "input_files": ["btree_s", "btree_m", "btree_l"],
        "primary_input": "btree_m",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "b_tree" / fname),
        "cpu_time_regex": r"Total time:\s*([\d.]+)",
        "gpu_time_regex": r"Total time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "B-Tree node traversal and queries. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "leukocyte": {
        "workload_name": "leukocyte",
        "domain": "cell_detection",
        "workload_type": "medical_imaging",
        "cpu_command": "./leukocyte 10 4 {input_file}",
        "gpu_command": "./leukocyte 10 4 {input_file}",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "leukocyte"),
        "gpu_cwd": str(GPU_ROOT / "leukocyte"),
        "input_files": ["testfile_10frames", "testfile_30frames", "testfile_60frames"],
        "primary_input": "testfile_30frames",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "leukocyte" / fname),
        "cpu_time_regex": r"Total time:\s*([\d.]+)",
        "gpu_time_regex": r"Total time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Leukocyte blood cell motion tracking. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "streamcluster": {
        "workload_name": "streamcluster",
        "domain": "data_mining",
        "workload_type": "dense_clustering",
        "cpu_command": "./sc_cpu 10 20 64 65536 65536 1000 none output.txt 4",
        "gpu_command": "./sc_gpu 10 20 64 65536 65536 1000 none output.txt",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "streamcluster"),
        "gpu_cwd": str(GPU_ROOT / "streamcluster"),
        "input_files": ["stream_10k", "stream_65k", "stream_250k"],
        "primary_input": "stream_65k",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "streamcluster" / fname),
        "cpu_time_regex": r"Total time:\s*([\d.]+)",
        "gpu_time_regex": r"Total time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Streaming cluster allocation. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "hybridsort": {
        "workload_name": "hybridsort",
        "domain": "sorting",
        "workload_type": "parallel_sorting",
        "cpu_command": "./hybridsort_cpu 1000000",
        "gpu_command": "./hybridsort_gpu 1000000",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "hybridsort"),
        "gpu_cwd": str(GPU_ROOT / "hybridsort"),
        "input_files": ["sort_250k", "sort_1m", "sort_4m"],
        "primary_input": "sort_1m",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "hybridsort" / fname),
        "cpu_time_regex": r"Total time:\s*([\d.]+)",
        "gpu_time_regex": r"Total time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Bucket and merge sort. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "lavaMD": {
        "workload_name": "lavaMD",
        "domain": "molecular_dynamics",
        "workload_type": "n_body",
        "cpu_command": "./lavaMD -cores 4 -boxes1d 10",
        "gpu_command": "./lavaMD -boxes1d 10",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "lavaMD"),
        "gpu_cwd": str(GPU_ROOT / "lavaMD"),
        "input_files": ["boxes_5", "boxes_10", "boxes_20"],
        "primary_input": "boxes_10",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "lavaMD" / fname),
        "cpu_time_regex": r"Total time:\s*([\d.]+)",
        "gpu_time_regex": r"Total time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Short-range 3D molecular potential calculation. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "gaussian": {
        "workload_name": "gaussian",
        "domain": "dense_linear_system",
        "workload_type": "linear_algebra",
        "cpu_command": "./gaussian_cpu -s 2048",
        "gpu_command": "./gaussian_gpu -s 2048",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "gaussian"),
        "gpu_cwd": str(GPU_ROOT / "gaussian"),
        "input_files": ["matrix_1024.txt", "matrix_2048.txt", "matrix_4096.txt"],
        "primary_input": "matrix_2048.txt",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "gaussian" / fname),
        "cpu_time_regex": r"Total time:\s*([\d.]+)",
        "gpu_time_regex": r"Total time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Dense matrix elimination. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },

    "mummergpu": {
        "workload_name": "mummergpu",
        "domain": "bioinformatics",
        "workload_type": "sequence_alignment",
        "cpu_command": "./mummer_cpu {ref_file} {query_file}",
        "gpu_command": "./mummergpu {ref_file} {query_file}",
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "mummergpu"),
        "gpu_cwd": str(GPU_ROOT / "mummergpu"),
        "input_files": ["queries_25k", "queries_100k", "queries_400k"],
        "primary_input": "queries_100k",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "mummergpu" / fname),
        "cpu_time_regex": r"Total time:\s*([\d.]+)",
        "gpu_time_regex": r"Total time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Suffix tree sequence matching. Local compilation blocked: CUDA unavailable on Windows host.",
        "gpu_total_path_time_status": "blocked_no_cuda",
        "status": "configured__blocked_no_cuda",
    },
}

ALL_WORKLOADS: List[str] = list(WORKLOAD_CONFIGS.keys())


def get_workload_config(name: str) -> Optional[Dict[str, Any]]:
    """Retrieve workload configuration by name."""
    return WORKLOAD_CONFIGS.get(name.lower())


def get_input_profiles(name: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Retrieve S/M/L input profiles for a specific workload."""
    return WORKLOAD_INPUT_PROFILES.get(name.lower())
