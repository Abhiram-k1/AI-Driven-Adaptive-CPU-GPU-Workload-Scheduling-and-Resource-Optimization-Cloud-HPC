"""
rodinia_configs.py — Centralised workload configs for Rodinia 3.1
=================================================================
Provides WORKLOAD_CONFIG and ALL_WORKLOADS for all 19 workloads.
Used by scheduler and DAG components with environment-aware path resolution.
"""

import os
from pathlib import Path

# Environment-aware path resolution
RODINIA_ROOT = Path(os.getenv("RODINIA_ROOT", "/kaggle/working/Rodinia"))
DATA_ROOT = Path(os.getenv("RODINIA_DATA_ROOT", "/kaggle/input/datasets/tanvibhardwaj24/rodinia-data/rodinia_data"))

LOCAL_RODINIA_ROOT = Path(__file__).resolve().parent.parent.parent / "rodinia_3.1" / "rodinia_3.1"
LOCAL_DATA_ROOT = LOCAL_RODINIA_ROOT / "data"

if not RODINIA_ROOT.exists() and LOCAL_RODINIA_ROOT.exists():
    RODINIA_ROOT = LOCAL_RODINIA_ROOT
    DATA_ROOT = LOCAL_DATA_ROOT

CPU_ROOT = RODINIA_ROOT / "openmp"
GPU_ROOT = RODINIA_ROOT / "cuda"
D = str(DATA_ROOT)

WORKLOAD_CONFIG = {

    "bfs": {
        "cpu": {"cwd": str(CPU_ROOT / "bfs"), "command": f"./bfs 4 {D}/bfs/graph1MW_6.txt", "time_regex": r"Compute time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "bfs"), "command": f"./bfs {D}/bfs/graph1MW_6.txt", "time_regex": r"GPU Kernel Time:\s*([\d.]+)\s*ms"},
        "input_size_mb": 61.2, "compute_intensity": 2, "parallelism_degree": 8, "data_dep": False, "transfer_bytes_mb": 61.2, "domain": "graph"
    },
    "cfd": {
        "cpu": {"cwd": str(CPU_ROOT / "cfd"), "command": f"./euler3d_cpu {D}/cfd/fvcorr.domn.193K", "time_regex": r"Compute time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "cfd"), "command": f"./euler3d {D}/cfd/fvcorr.domn.193K", "time_regex": r"([\d.]+(?:e[+-]?\d+)?)\s+seconds per iteration"},
        "input_size_mb": 43.5, "compute_intensity": 8, "parallelism_degree": 10, "data_dep": True, "transfer_bytes_mb": 43.5, "domain": "fluid_dynamics"
    },
    "hotspot": {
        "cpu": {"cwd": str(CPU_ROOT / "hotspot"), "command": f"./hotspot 512 2 2 {D}/hotspot/temp_512 {D}/hotspot/power_512 /tmp/out.txt", "time_regex": r"Compute time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "hotspot"), "command": f"./hotspot 512 2 2 {D}/hotspot/temp_512 {D}/hotspot/power_512 /tmp/out.txt", "time_regex": r"Time:\s*([\d.]+)"},
        "input_size_mb": 2.75, "compute_intensity": 6, "parallelism_degree": 9, "data_dep": True, "transfer_bytes_mb": 2.75, "domain": "thermal_simulation"
    },
    "lud": {
        "cpu": {"cwd": str(CPU_ROOT / "lud"), "command": "./omp/lud_omp -s 512", "time_regex": r"Time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "lud"), "command": "./cuda/lud_cuda -s 512 -v", "time_regex": r"Time:\s*([\d.]+)"},
        "input_size_mb": 2.48, "compute_intensity": 9, "parallelism_degree": 10, "data_dep": False, "transfer_bytes_mb": 2.48, "domain": "linear_algebra"
    },
    "kmeans": {
        "cpu": {"cwd": str(CPU_ROOT / "kmeans"), "command": f"./kmeans_openmp/kmeans -n 4 -i {D}/kmeans/819200.txt", "time_regex": r"Compute time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "kmeans"), "command": f"./kmeans -o -i {D}/kmeans/819200.txt", "time_regex": r"Time for kMeans:\s*([\d.]+)"},
        "input_size_mb": 97.0, "compute_intensity": 5, "parallelism_degree": 8, "data_dep": True, "transfer_bytes_mb": 97.0, "domain": "clustering"
    },
    "srad": {
        "cpu": {"cwd": str(CPU_ROOT / "srad"), "command": "./srad 100 0.5 502 458 4", "time_regex": r"Computation time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "srad" / "srad_v1"), "command": "./srad 100 0.5 502 458", "time_regex": r"Computation time:\s*([\d.]+)"},
        "input_size_mb": 0.75, "compute_intensity": 7, "parallelism_degree": 9, "data_dep": True, "transfer_bytes_mb": 0.75, "domain": "image_processing"
    },
    "nn": {
        "cpu": {"cwd": str(CPU_ROOT / "nn"), "command": f"./nn {D}/nn/filelist_4 5 30 90", "time_regex": r"Time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "nn"), "command": f"./nn {D}/nn/filelist_4 -r 5 -lat 30 -lng 90", "time_regex": r"Time:\s*([\d.]+)"},
        "input_size_mb": 0.50, "compute_intensity": 3, "parallelism_degree": 9, "data_dep": False, "transfer_bytes_mb": 0.50, "domain": "distance_calculation"
    },
    "nw": {
        "cpu": {"cwd": str(CPU_ROOT / "nw"), "command": "./needle 2048 10 4", "time_regex": r"Time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "nw"), "command": "./needle 2048 10", "time_regex": r"Time:\s*([\d.]+)"},
        "input_size_mb": 16.0, "compute_intensity": 6, "parallelism_degree": 6, "data_dep": True, "transfer_bytes_mb": 16.0, "domain": "bioinformatics"
    },
    "backprop": {
        "cpu": {"cwd": str(CPU_ROOT / "backprop"), "command": "./backprop 524288", "time_regex": r"Time:\s*([\d.]+)\s*sec"},
        "gpu": {"cwd": str(GPU_ROOT / "backprop"), "command": "./backprop 524288", "time_regex": r"Time:\s*([\d.]+)\s*sec"},
        "input_size_mb": 8.0, "compute_intensity": 8, "parallelism_degree": 9, "data_dep": False, "transfer_bytes_mb": 8.0, "domain": "machine_learning"
    },
    "pathfinder": {
        "cpu": {"cwd": str(CPU_ROOT / "pathfinder"), "command": "./pathfinder 100000 100 20", "time_regex": r"Time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "pathfinder"), "command": "./pathfinder 100000 100 20", "time_regex": r"Time:\s*([\d.]+)"},
        "input_size_mb": 40.0, "compute_intensity": 5, "parallelism_degree": 7, "data_dep": True, "transfer_bytes_mb": 40.0, "domain": "dynamic_programming"
    },
    "particlefilter": {
        "cpu": {"cwd": str(CPU_ROOT / "particlefilter"), "command": "./particle_filter -x 128 -y 128 -z 10 -np 1000", "time_regex": r"TOTAL TIME:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "particlefilter"), "command": "./particlefilter_float -x 128 -y 128 -z 10 -np 1000", "time_regex": r"TOTAL TIME:\s*([\d.]+)"},
        "input_size_mb": 6.0, "compute_intensity": 7, "parallelism_degree": 10, "data_dep": False, "transfer_bytes_mb": 6.0, "domain": "tracking"
    },
    "heartwall": {
        "cpu": {"cwd": str(CPU_ROOT / "heartwall"), "command": f"./heartwall {D}/heartwall/test.avi 50", "time_regex": r"Compute time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "heartwall"), "command": f"./heartwall {D}/heartwall/test.avi 50", "time_regex": r"Compute time:\s*([\d.]+)"},
        "input_size_mb": 37.5, "compute_intensity": 7, "parallelism_degree": 8, "data_dep": True, "transfer_bytes_mb": 37.5, "domain": "medical_imaging"
    },
    "b_tree": {
        "cpu": {"cwd": str(CPU_ROOT / "b_tree"), "command": "./b_tree core 4 file data.txt command command.txt", "time_regex": r"Total time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "b_tree"), "command": "./b_tree file data.txt command command.txt", "time_regex": r"Total time:\s*([\d.]+)"},
        "input_size_mb": 20.0, "compute_intensity": 3, "parallelism_degree": 5, "data_dep": True, "transfer_bytes_mb": 16.0, "domain": "data_structure"
    },
    "leukocyte": {
        "cpu": {"cwd": str(CPU_ROOT / "leukocyte"), "command": f"./leukocyte 10 4 {D}/leukocyte/testfile.avi", "time_regex": r"Total time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "leukocyte"), "command": f"./leukocyte 10 4 {D}/leukocyte/testfile.avi", "time_regex": r"Total time:\s*([\d.]+)"},
        "input_size_mb": 36.0, "compute_intensity": 7, "parallelism_degree": 8, "data_dep": False, "transfer_bytes_mb": 36.0, "domain": "medical_imaging"
    },
    "streamcluster": {
        "cpu": {"cwd": str(CPU_ROOT / "streamcluster"), "command": "./sc_cpu 10 20 64 65536 65536 1000 none output.txt 4", "time_regex": r"Total time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "streamcluster"), "command": "./sc_gpu 10 20 64 65536 65536 1000 none output.txt", "time_regex": r"Total time:\s*([\d.]+)"},
        "input_size_mb": 40.0, "compute_intensity": 4, "parallelism_degree": 6, "data_dep": True, "transfer_bytes_mb": 40.0, "domain": "clustering"
    },
    "hybridsort": {
        "cpu": {"cwd": str(CPU_ROOT / "hybridsort"), "command": "./hybridsort_cpu 1000000", "time_regex": r"Total time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "hybridsort"), "command": "./hybridsort_gpu 1000000", "time_regex": r"Total time:\s*([\d.]+)"},
        "input_size_mb": 16.0, "compute_intensity": 6, "parallelism_degree": 9, "data_dep": False, "transfer_bytes_mb": 16.0, "domain": "sorting"
    },
    "lavaMD": {
        "cpu": {"cwd": str(CPU_ROOT / "lavaMD"), "command": "./lavaMD -cores 4 -boxes1d 10", "time_regex": r"Total time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "lavaMD"), "command": "./lavaMD -boxes1d 10", "time_regex": r"Total time:\s*([\d.]+)"},
        "input_size_mb": 18.0, "compute_intensity": 9, "parallelism_degree": 10, "data_dep": False, "transfer_bytes_mb": 18.0, "domain": "molecular_dynamics"
    },
    "gaussian": {
        "cpu": {"cwd": str(CPU_ROOT / "gaussian"), "command": "./gaussian_cpu -s 2048", "time_regex": r"Total time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "gaussian"), "command": "./gaussian_gpu -s 2048", "time_regex": r"Total time:\s*([\d.]+)"},
        "input_size_mb": 32.0, "compute_intensity": 7, "parallelism_degree": 8, "data_dep": True, "transfer_bytes_mb": 32.0, "domain": "linear_algebra"
    },
    "mummergpu": {
        "cpu": {"cwd": str(CPU_ROOT / "mummergpu"), "command": "./mummer_cpu ref.fa query.fa", "time_regex": r"Total time:\s*([\d.]+)"},
        "gpu": {"cwd": str(GPU_ROOT / "mummergpu"), "command": "./mummergpu ref.fa query.fa", "time_regex": r"Total time:\s*([\d.]+)"},
        "input_size_mb": 48.0, "compute_intensity": 6, "parallelism_degree": 8, "data_dep": False, "transfer_bytes_mb": 48.0, "domain": "bioinformatics"
    },
}

ALL_WORKLOADS = list(WORKLOAD_CONFIG.keys())
