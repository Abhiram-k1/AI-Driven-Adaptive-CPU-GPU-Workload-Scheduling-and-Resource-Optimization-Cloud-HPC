"""
profiler/workload_configs.py
============================
Centralised, workload-specific profiling configurations for Rodinia 3.1.
Authoritative source-level specifications conforming to PROJECT_PLAN.md Section 8.2.

Every Rodinia workload has unique build requirements, argument structures,
timing semantics, and output formats. Generic parsers are strictly prohibited.
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


WORKLOAD_CONFIGS: Dict[str, Dict[str, Any]] = {

    "bfs": {
        "workload_name": "bfs",
        "domain": "graph_traversal",
        "workload_type": "graph",
        "cpu_command": "./bfs 4 {input_file}",
        "gpu_command": "./bfs {input_file}",
        # CPU takes thread count (4); GPU does not.
        "cpu_build": "make -f Makefile openmp",
        "gpu_build": "make -f Makefile cuda",
        "cpu_cwd": str(CPU_ROOT / "bfs"),
        "gpu_cwd": str(GPU_ROOT / "bfs"),
        "input_files": ["graph1MW_6.txt", "graph4096.txt", "graph65536.txt"],
        "primary_input": "graph1MW_6.txt",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "bfs" / fname),
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",          # seconds (openmp/bfs/bfs.cpp:173)
        "gpu_time_regex": r"GPU Kernel Time:\s*([\d.]+)\s*ms",  # ms (cuda/bfs/bfs.cu loop)
        "h2d_regex": r"Initial Host to Device Time:\s*([\d.]+)\s*ms",
        "d2h_regex": r"Final Device to Host Time:\s*([\d.]+)\s*ms",
        "h2d_instrumented": True,
        "d2h_instrumented": True,
        "timing_notes": (
            "[PENDING end-to-end validation] Source-level segment structure: "
            "(1) Initial H2D (~14.12 ms candidate) transfers graph arrays before traversal loop. "
            "(2) GPU Traversal Loop phase (~3.83 ms candidate) executes 12 iterations of Kernel/Kernel2; "
            "internal 1-byte control copies (d_over) are intrinsic to loop control. "
            "(3) Final D2H (~0.77 ms candidate) copies node costs to host after loop. "
            "Candidate total GPU offload path = H2D + Loop + D2H = 14.12 + 3.83 + 0.77 ≈ 18.72 ms. "
            "Gated: gpu_total_path_time_status = 'pending_validation'. preferred_device = null."
        ),
        "gpu_total_path_time_status": "pending_validation",
        "status": "source_inspection_complete__end_to_end_validation_pending",
    },

    "cfd": {
        "workload_name": "cfd",
        "domain": "fluid_dynamics",
        "workload_type": "iterative_solver",
        "cpu_command": "./euler3d_cpu {input_file}",
        "gpu_command": "./euler3d {input_file}",
        "cpu_build": "g++ -O3 -Dblock_length=8 -fopenmp euler3d_cpu.cpp -o euler3d_cpu",
        "gpu_build": (
            "nvcc -O2 -Xptxas -v -arch=sm_75 "
            "-I/kaggle/working/Rodinia/cuda/hybridsort "
            "euler3d.cu -o euler3d"
        ),
        "cpu_cwd": str(CPU_ROOT / "cfd"),
        "gpu_cwd": str(GPU_ROOT / "cfd"),
        "input_files": ["fvcorr.domn.193K", "fvcorr.domn.097K"],
        "primary_input": "fvcorr.domn.193K",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "cfd" / fname),
        # Source inspection:
        # GPU (cuda/cfd/euler3d.cu:589): outputs "<val> seconds per iteration"
        "gpu_time_regex": r"([\d.]+(?:e[+-]?\d+)?)\s+seconds per iteration",
        # CPU (openmp/cfd/euler3d_cpu.cpp:494): outputs "Compute time: <val>" (total seconds for all 2000 iter)
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",
        "iterations": 2000,
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": (
            "[EXPERIMENTALLY CLOSED] Source inspection + Kaggle Tesla T4 verification: "
            "Both CPU and GPU measure 2000 Runge-Kutta iterations, excluding file I/O, domain setup, "
            "and solution dump. GPU reports seconds per iteration; CPU reports total seconds for 2000 iter. "
            "Scaling rule: gpu_total_compute_ms = sec_per_iter * 2000.0 * 1000.0; cpu_compute_ms = sec * 1000.0. "
            "Comparability confirmed: both measure identical 2000 RK iteration compute phase. "
            "gpu_total_path_time_status = 'validated', comparable_timing_ok = True, preferred_device = 'gpu'."
        ),
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
        "timing_notes": "Grid-based iterative thermal model. Configured; pending cluster validation.",
        "gpu_total_path_time_status": "pending_validation",
        "status": "configured__unprofiled",
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
        "input_files": ["256.dat", "512.dat"],
        "primary_input": "256.dat",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "lud" / fname),
        "cpu_time_regex": r"Time:\s*([\d.]+)",
        "gpu_time_regex": r"Time:\s*([\d.]+)",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "LU matrix decomposition with dense tiles. Configured; pending cluster validation.",
        "gpu_total_path_time_status": "pending_validation",
        "status": "configured__unprofiled",
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
        "input_files": ["kdd_cup"],
        "primary_input": "kdd_cup",
        "input_path_resolver": lambda fname: str(ACTIVE_DATA_ROOT / "kmeans" / fname),
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",
        "gpu_time_regex": r"GPU Kernel Time:\s*([\d.]+)\s*ms",
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "K-means clustering algorithm. Configured; pending cluster validation.",
        "gpu_total_path_time_status": "pending_validation",
        "status": "configured__unprofiled",
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
        "timing_notes": "Speckle Reducing Anisotropic Diffusion. Configured; pending cluster validation.",
        "gpu_total_path_time_status": "pending_validation",
        "status": "configured__unprofiled",
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
        "h2d_instrumented": False,
        "d2h_instrumented": False,
        "timing_notes": "Nearest neighbor record distance calculation. Configured; pending cluster validation.",
        "gpu_total_path_time_status": "pending_validation",
        "status": "configured__unprofiled",
    },
}
