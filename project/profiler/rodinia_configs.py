"""
rodinia_configs.py — Centralised workload configs for Rodinia 3.1
=================================================================
For each workload:
  cpu / gpu:
    cwd       : working directory
    command   : run command (format string with {data})
    time_regex: regex to extract elapsed seconds from stdout
  input_size_mb     : approximate input data in MB
  compute_intensity : 1-10, higher = more FLOP/byte
  parallelism_degree: 1-10, higher = exploits more threads/warps
  data_dep          : True if stages are sequentially dependent
  transfer_bytes_mb : typical CPU->GPU transfer size in MB
  domain            : application domain label
"""

from pathlib import Path

# ─── Configure these paths to match your environment ──────────────────────
RODINIA_ROOT = Path("/kaggle/working/Rodinia")
DATA_ROOT    = Path("/kaggle/input/datasets/tanvibhardwaj24/rodinia-data/rodinia_data")
# ──────────────────────────────────────────────────────────────────────────

CPU_ROOT = RODINIA_ROOT / "openmp"
GPU_ROOT = RODINIA_ROOT / "cuda"
D        = str(DATA_ROOT)   # shorthand

WORKLOAD_CONFIG = {

    "bfs": {
        "cpu": {
            "cwd": str(CPU_ROOT / "bfs"),
            "command": f"./bfs 4 {D}/bfs/graph1MW_6.txt",
            "time_regex": r"Time:\s*([\d.]+)\s*s",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "bfs"),
            "command": f"./bfs 4 {D}/bfs/graph1MW_6.txt",
            "time_regex": r"Time:\s*([\d.]+)\s*s",
        },
        "input_size_mb": 28.0,
        "compute_intensity": 2,
        "parallelism_degree": 8,
        "data_dep": False,
        "transfer_bytes_mb": 28.0,
        "domain": "graph",
    },

    "cfd": {
        "cpu": {
            "cwd": str(CPU_ROOT / "cfd"),
            "command": f"./euler3d {D}/cfd/fvcorr.domn.193K",
            "time_regex": r"Compute time:\s*([\d.]+)",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "cfd"),
            "command": f"./euler3d {D}/cfd/fvcorr.domn.193K",
            "time_regex": r"Compute time:\s*([\d.]+)",
        },
        "input_size_mb": 15.0,
        "compute_intensity": 8,
        "parallelism_degree": 10,
        "data_dep": True,
        "transfer_bytes_mb": 15.0,
        "domain": "fluid_dynamics",
    },

    "hotspot": {
        "cpu": {
            "cwd": str(CPU_ROOT / "hotspot"),
            "command": f"./hotspot 512 2 2 {D}/hotspot/temp_512 {D}/hotspot/power_512 /tmp/out.txt",
            "time_regex": r"Time:\s*([\d.]+)",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "hotspot"),
            "command": f"./hotspot 512 2 2 {D}/hotspot/temp_512 {D}/hotspot/power_512 /tmp/out.txt",
            "time_regex": r"Time:\s*([\d.]+)",
        },
        "input_size_mb": 2.0,
        "compute_intensity": 6,
        "parallelism_degree": 9,
        "data_dep": True,
        "transfer_bytes_mb": 2.0,
        "domain": "thermal_simulation",
    },

    "lud": {
        "cpu": {
            "cwd": str(CPU_ROOT / "lud"),
            "command": "./lud -s 2048",
            "time_regex": r"Time:\s*([\d.]+)\s*s",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "lud"),
            "command": "./lud -s 2048",
            "time_regex": r"Time:\s*([\d.]+)\s*s",
        },
        "input_size_mb": 32.0,
        "compute_intensity": 9,
        "parallelism_degree": 10,
        "data_dep": False,
        "transfer_bytes_mb": 32.0,
        "domain": "linear_algebra",
    },

    "kmeans": {
        "cpu": {
            "cwd": str(CPU_ROOT / "kmeans"),
            "command": f"./kmeans -o -i {D}/kmeans/kdd_cup",
            "time_regex": r"Time for kMeans:\s*([\d.]+)",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "kmeans"),
            "command": f"./kmeans -o -i {D}/kmeans/kdd_cup",
            "time_regex": r"Time for kMeans:\s*([\d.]+)",
        },
        "input_size_mb": 73.0,
        "compute_intensity": 5,
        "parallelism_degree": 8,
        "data_dep": True,
        "transfer_bytes_mb": 73.0,
        "domain": "clustering",
    },

    "srad": {
        "cpu": {
            "cwd": str(CPU_ROOT / "srad" / "srad_v2"),
            "command": "./srad 2048 2048 0 127 0 127 0.5 2",
            "time_regex": r"Computation done in\s*([\d.]+)\s*s",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "srad" / "srad_v2"),
            "command": "./srad 2048 2048 0 127 0 127 0.5 2",
            "time_regex": r"Computation done in\s*([\d.]+)\s*s",
        },
        "input_size_mb": 32.0,
        "compute_intensity": 7,
        "parallelism_degree": 9,
        "data_dep": True,
        "transfer_bytes_mb": 32.0,
        "domain": "image_processing",
    },

    "nw": {
        "cpu": {
            "cwd": str(CPU_ROOT / "nw"),
            "command": "./needle 2048 10",
            "time_regex": r"Time:\s*([\d.]+)",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "nw"),
            "command": "./needle 2048 10",
            "time_regex": r"Time:\s*([\d.]+)",
        },
        "input_size_mb": 16.0,
        "compute_intensity": 5,
        "parallelism_degree": 6,
        "data_dep": True,
        "transfer_bytes_mb": 16.0,
        "domain": "bioinformatics",
    },

    "backprop": {
        "cpu": {
            "cwd": str(CPU_ROOT / "backprop"),
            "command": "./backprop 524288",
            "time_regex": r"Time:\s*([\d.]+)\s*sec",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "backprop"),
            "command": "./backprop 524288",
            "time_regex": r"Time:\s*([\d.]+)\s*sec",
        },
        "input_size_mb": 8.0,
        "compute_intensity": 6,
        "parallelism_degree": 7,
        "data_dep": True,
        "transfer_bytes_mb": 8.0,
        "domain": "machine_learning",
    },

    "pathfinder": {
        "cpu": {
            "cwd": str(CPU_ROOT / "pathfinder"),
            "command": "./pathfinder 100000 100 20",
            "time_regex": r"Time:\s*([\d.]+)",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "pathfinder"),
            "command": "./pathfinder 100000 100 20",
            "time_regex": r"Time:\s*([\d.]+)",
        },
        "input_size_mb": 40.0,
        "compute_intensity": 4,
        "parallelism_degree": 7,
        "data_dep": True,
        "transfer_bytes_mb": 40.0,
        "domain": "dynamic_programming",
    },

    "particlefilter": {
        "cpu": {
            "cwd": str(CPU_ROOT / "particlefilter"),
            "command": "./particle_filter -x 128 -y 128 -z 10 -np 1000",
            "time_regex": r"TOTAL TIME:\s*([\d.]+)",
        },
        "gpu": {
            "cwd": str(GPU_ROOT / "particlefilter"),
            "command": "./particlefilter_float -x 128 -y 128 -z 10 -np 1000",
            "time_regex": r"TOTAL TIME:\s*([\d.]+)",
        },
        "input_size_mb": 6.0,
        "compute_intensity": 7,
        "parallelism_degree": 10,
        "data_dep": False,
        "transfer_bytes_mb": 6.0,
        "domain": "tracking",
    },
}

ALL_WORKLOADS = list(WORKLOAD_CONFIG.keys())
