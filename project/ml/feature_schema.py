"""
ml/feature_schema.py
====================
Authoritative pre-execution feature definitions and frozen Rubric v1.0.
Strictly implements Sections 11, 12, 13, 14, 15, 16 of the Master Execution Prompt.

CRITICAL GUARANTEES:
1. STRICT ANTI-LEAKAGE: Excludes all post-execution telemetry (measured runtimes,
   measured transfer times, empirical speedups, power/energy measurements).
2. PRE-EXECUTION ONLY: Every feature is derived strictly before execution starts.
3. FROZEN RUBRIC v1.0: Deterministic mapping from source-level algorithmic characteristics.
4. MODEL A vs MODEL B SPECIFICATION:
   - Model A Features: Full pre-execution set including workload_type and workload_domain.
   - Model B Features: Intrinsic characteristics only, excluding workload_type and workload_domain.
"""

from typing import Dict, Any, List

# List of prohibited post-execution leakage fields
FORBIDDEN_LEAKAGE_COLUMNS = {
    "cpu_median_ms", "cpu_mean_ms", "cpu_std_ms", "cpu_process_wall_time_ms",
    "gpu_median_ms", "gpu_mean_ms", "gpu_std_ms", "gpu_phase_time_ms",
    "gpu_kernel_time_ms", "host_to_device_time_ms", "device_to_host_time_ms",
    "gpu_total_path_ms", "measured_speedup", "measured_energy", "measured_power",
    "actual_speedup", "post_execution_timing_ratio", "empirical_preferred_device"
}

# Pre-execution feature columns for Model A (Full Feature Set)
MODEL_A_FEATURES = [
    "input_size_mib",
    "workload_type",
    "workload_domain",
    "estimated_parallelism",
    "estimated_memory_boundness",
    "estimated_compute_intensity",
    "has_irregular_memory",
    "has_strong_serial_dependency",
    "estimated_transfer_size_mib",
    "transfer_estimation_method",
    "cpu_cores_available",
    "gpu_type_encoded",
    "gpu_memory_gb",
]

# Pre-execution feature columns for Model B (Workload-Identity-Free Ablation)
MODEL_B_FEATURES = [
    "input_size_mib",
    "estimated_parallelism",
    "estimated_memory_boundness",
    "estimated_compute_intensity",
    "has_irregular_memory",
    "has_strong_serial_dependency",
    "estimated_transfer_size_mib",
    "transfer_estimation_method",
    "cpu_cores_available",
    "gpu_type_encoded",
    "gpu_memory_gb",
]

CATEGORICAL_FEATURES = [
    "workload_type",
    "workload_domain",
    "transfer_estimation_method",
    "gpu_type_encoded"
]


# Frozen Rubric v1.0 Lookup Table for Rodinia Workloads (Extended to full 19 suite)
# Derived from source code inspection and algorithmic complexity analysis
FROZEN_RUBRIC_V1: Dict[str, Dict[str, Any]] = {
    "bfs": {
        "workload_type": "graph",
        "workload_domain": "graph_traversal",
        "estimated_parallelism": 3,               # Frontier-based variable parallelism, thread divergence on sparse vertices
        "estimated_memory_boundness": 5,          # AI < 0.2 FLOP/byte, heavy random memory access
        "estimated_compute_intensity": 1,         # Minimal computation per edge traversal (integer comparisons)
        "has_irregular_memory": 1,                # Irregular memory accesses via adjacency list indirection
        "has_strong_serial_dependency": 1,        # Iterative level-by-level traversal with barrier synchronization
        "transfer_estimation_ratio": 1.0,         # Transfers node list, edge list, mask, updating cost
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "cfd": {
        "workload_type": "iterative_solver",
        "workload_domain": "fluid_dynamics",
        "estimated_parallelism": 5,               # Massive element-wise parallelism over unstructured 3D mesh
        "estimated_memory_boundness": 4,          # High memory bandwidth demand for cell variables and fluxes (AI ~ 1.5 FLOP/byte)
        "estimated_compute_intensity": 4,         # 4-stage Runge-Kutta integration per cell per time step
        "has_irregular_memory": 0,                # Structured array index lookups over mesh edges
        "has_strong_serial_dependency": 0,        # Spatial steps are independent within each Runge-Kutta substep
        "transfer_estimation_ratio": 1.0,         # Mesh coordinates, elements, initial variables uploaded
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "hotspot": {
        "workload_type": "structured_grid",
        "workload_domain": "thermal_simulation",
        "estimated_parallelism": 4,               # 2D grid stencil simulation
        "estimated_memory_boundness": 3,          # Arithmetic intensity ~ 2.5 FLOP/byte
        "estimated_compute_intensity": 3,         # Differential equation update per grid cell
        "has_irregular_memory": 0,                # Regular 2D nearest-neighbor stencil
        "has_strong_serial_dependency": 0,        # Time-stepping simulation
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "lud": {
        "workload_type": "matrix_decomposition",
        "workload_domain": "linear_algebra",
        "estimated_parallelism": 4,               # Blocked 2D matrix decomposition
        "estimated_memory_boundness": 2,          # Compute-bound tiled matrix multiplication (O(N^3) ops on O(N^2) data)
        "estimated_compute_intensity": 5,         # Heavy floating-point arithmetic
        "has_irregular_memory": 0,                # Dense linear matrix memory layout
        "has_strong_serial_dependency": 1,        # Outer diagonal elimination sequence
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "kmeans": {
        "workload_type": "dense_clustering",
        "workload_domain": "data_mining",
        "estimated_parallelism": 5,               # Independent distance calculation for N objects
        "estimated_memory_boundness": 3,          # Feature vector distance evaluations
        "estimated_compute_intensity": 3,         # Euclidean distance calculations across K cluster centers
        "has_irregular_memory": 0,                # Contiguous feature arrays
        "has_strong_serial_dependency": 0,        # Centroid accumulation
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "srad": {
        "workload_type": "structured_grid",
        "workload_domain": "image_processing",
        "estimated_parallelism": 4,               # 2D image pixel grid filtering
        "estimated_memory_boundness": 3,          # Stencil diffusion computation
        "estimated_compute_intensity": 3,         # Exponential and differential equations per pixel
        "has_irregular_memory": 0,                # 2D image raster array
        "has_strong_serial_dependency": 0,        # Iterative diffusion
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "nn": {
        "workload_type": "distance_calculation",
        "workload_domain": "data_mining",
        "estimated_parallelism": 5,               # Independent record distance evaluation
        "estimated_memory_boundness": 4,          # Streaming record reads from file/memory
        "estimated_compute_intensity": 2,         # 2D coordinate distance calculation
        "has_irregular_memory": 0,                # Sequential record records
        "has_strong_serial_dependency": 0,        # Embarrassingly parallel search
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "nw": {
        "workload_type": "dynamic_programming",
        "workload_domain": "bioinformatics",
        "estimated_parallelism": 3,               # Wavefront anti-diagonal parallelism
        "estimated_memory_boundness": 3,          # Moderate memory traffic with score matrix
        "estimated_compute_intensity": 4,         # Scoring matrix cell computations
        "has_irregular_memory": 0,                # Regular matrix indexing
        "has_strong_serial_dependency": 1,        # Strict diagonal wavefront dependency
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "backprop": {
        "workload_type": "machine_learning",
        "workload_domain": "pattern_recognition",
        "estimated_parallelism": 5,               # High parallel neuron weight adjustments
        "estimated_memory_boundness": 2,          # High arithmetic reuse in forward/backward pass
        "estimated_compute_intensity": 5,         # Heavy matrix-vector multiplies and sigmoids
        "has_irregular_memory": 0,                # Contiguous layer activation arrays
        "has_strong_serial_dependency": 0,        # Layer-wise feedforward/backprop
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "pathfinder": {
        "workload_type": "dynamic_programming",
        "workload_domain": "grid_traversal",
        "estimated_parallelism": 4,               # Row-wise dynamic programming path updates
        "estimated_memory_boundness": 3,          # 2D grid cell reading
        "estimated_compute_intensity": 3,         # Min-path calculations per node
        "has_irregular_memory": 0,                # Regular rectangular grid
        "has_strong_serial_dependency": 1,        # Sequential row-by-row dependency
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "particlefilter": {
        "workload_type": "tracking",
        "workload_domain": "sequential_monte_carlo",
        "estimated_parallelism": 5,               # Independent particle state evolution
        "estimated_memory_boundness": 2,          # Compute-heavy weight calculation
        "estimated_compute_intensity": 4,         # Exponential likelihood updates
        "has_irregular_memory": 0,                # Contiguous particle array
        "has_strong_serial_dependency": 0,        # Embarrassingly parallel resampling
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "heartwall": {
        "workload_type": "medical_imaging",
        "workload_domain": "pattern_tracking",
        "estimated_parallelism": 4,               # Multi-region template search
        "estimated_memory_boundness": 4,          # High video frame memory bandwidth
        "estimated_compute_intensity": 4,         # Cross-correlation template matching
        "has_irregular_memory": 1,                # Irregular endocardium boundary masks
        "has_strong_serial_dependency": 1,        # Temporal frame-to-frame tracking
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "b_tree": {
        "workload_type": "data_structure",
        "workload_domain": "tree_traversal",
        "estimated_parallelism": 3,               # Concurrent query search
        "estimated_memory_boundness": 5,          # Pointer chasing in B-tree nodes
        "estimated_compute_intensity": 2,         # Binary key comparisons
        "has_irregular_memory": 1,                # Irregular tree branch memory layout
        "has_strong_serial_dependency": 1,        # Root-to-leaf traversal steps
        "transfer_estimation_ratio": 0.8,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "leukocyte": {
        "workload_type": "medical_imaging",
        "workload_domain": "cell_detection",
        "estimated_parallelism": 4,               # Gradient matrix calculation across frames
        "estimated_memory_boundness": 4,          # Video frame buffer access
        "estimated_compute_intensity": 4,         # Gradient matrix & active contour optimization
        "has_irregular_memory": 1,                # Cell boundary contour coordinates
        "has_strong_serial_dependency": 0,        # Multi-cell independent detection
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "streamcluster": {
        "workload_type": "dense_clustering",
        "workload_domain": "data_mining",
        "estimated_parallelism": 3,               # Center opening distance calculations
        "estimated_memory_boundness": 5,          # Heavy streaming memory bandwidth
        "estimated_compute_intensity": 2,         # Distance evaluations and gain calculations
        "has_irregular_memory": 0,                # Contiguous streaming data points
        "has_strong_serial_dependency": 1,        # Serial accumulation across iterations
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "hybridsort": {
        "workload_type": "parallel_sorting",
        "workload_domain": "sorting",
        "estimated_parallelism": 5,               # Multi-element bucket and merge sort
        "estimated_memory_boundness": 4,          # Memory bandwidth during scatter/gather
        "estimated_compute_intensity": 3,         # Sorting comparisons and key partitioning
        "has_irregular_memory": 1,                # Irregular bucket histogram distribution
        "has_strong_serial_dependency": 0,        # Independent sub-bucket merges
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "lavaMD": {
        "workload_type": "n_body",
        "workload_domain": "molecular_dynamics",
        "estimated_parallelism": 5,               # 3D box pair pairwise potential calculation
        "estimated_memory_boundness": 2,          # High arithmetic intensity (O(N^2) on local boxes)
        "estimated_compute_intensity": 5,         # Coulomb & Lennard-Jones floating point forces
        "has_irregular_memory": 0,                # Structured 3D space sub-boxes
        "has_strong_serial_dependency": 0,        # Independent box-pair interactions
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "gaussian": {
        "workload_type": "linear_algebra",
        "workload_domain": "dense_linear_system",
        "estimated_parallelism": 4,               # Row elimination across remaining matrix rows
        "estimated_memory_boundness": 3,          # Matrix row updates
        "estimated_compute_intensity": 4,         # Floating-point divide and multiply-subtract
        "has_irregular_memory": 0,                # Dense 2D array representation
        "has_strong_serial_dependency": 1,        # Sequential pivot row elimination
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    },
    "mummergpu": {
        "workload_type": "sequence_alignment",
        "workload_domain": "bioinformatics",
        "estimated_parallelism": 4,               # Suffix tree lookup across query sequences
        "estimated_memory_boundness": 4,          # Irregular tree traversals
        "estimated_compute_intensity": 3,         # Character matching across suffixes
        "has_irregular_memory": 1,                # Suffix tree pointer traversals
        "has_strong_serial_dependency": 0,        # Embarrassingly parallel across query chunks
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "source_buffer_analysis",
    }
}


def extract_pre_execution_features(
    workload: str,
    input_file: str,
    input_size_mib: float,
    cpu_cores: int = 4,
    gpu_type: str = "Tesla_T4",
    gpu_memory_gb: float = 16.0
) -> Dict[str, Any]:
    """
    Deterministically computes the pre-execution feature record for a workload configuration.
    Uses frozen Rubric v1.0 and pre-execution buffer estimates.
    Zero post-execution telemetry is accessed.
    """
    rubric = FROZEN_RUBRIC_V1.get(workload, {
        "workload_type": "unknown",
        "workload_domain": "general",
        "estimated_parallelism": 3,
        "estimated_memory_boundness": 3,
        "estimated_compute_intensity": 3,
        "has_irregular_memory": 0,
        "has_strong_serial_dependency": 0,
        "transfer_estimation_ratio": 1.0,
        "transfer_estimation_method": "heuristic_fallback",
    })

    # Pre-execution transfer estimate: derived from input buffer sizing ratio
    est_transfer_mib = input_size_mib * rubric["transfer_estimation_ratio"]

    features: Dict[str, Any] = {
        "workload": workload,
        "input_file": input_file,
        "input_size_mib": round(float(input_size_mib), 5),
        "workload_type": rubric["workload_type"],
        "workload_domain": rubric["workload_domain"],
        "estimated_parallelism": int(rubric["estimated_parallelism"]),
        "estimated_memory_boundness": int(rubric["estimated_memory_boundness"]),
        "estimated_compute_intensity": int(rubric["estimated_compute_intensity"]),
        "has_irregular_memory": int(rubric["has_irregular_memory"]),
        "has_strong_serial_dependency": int(rubric["has_strong_serial_dependency"]),
        "estimated_transfer_size_mib": round(float(est_transfer_mib), 5),
        "transfer_estimation_method": rubric["transfer_estimation_method"],
        "cpu_cores_available": int(cpu_cores),
        "gpu_type_encoded": str(gpu_type),
        "gpu_memory_gb": round(float(gpu_memory_gb), 1),
    }

    # Anti-leakage validation assertion
    for forbidden in FORBIDDEN_LEAKAGE_COLUMNS:
        assert forbidden not in features, f"LEAKAGE DETECTED: {forbidden} found in feature record!"

    return features
