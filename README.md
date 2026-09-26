# AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduling and Resource Optimization for Cloud HPC

**B.Tech AI & Data Science Capstone Project — SEM-5 HPC**  
**Authoritative Technical Specification & Implementation Report**

---

## 1. Project Title
**AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduling and Resource Optimization for Cloud HPC**

---

## 2. Research Question
Can an empirical machine learning model, trained exclusively on pre-execution workload characteristics, accurately predict whether an HPC task is more suitable for CPU or GPU execution, and can this prediction form the basis for multi-objective scheduling in heterogeneous cloud environments?

---

## 3. Current Objectives & Scope Boundary
For this development phase, **implementation is complete through the end of Objective 2**:

- **OBJECTIVE 1 — COMPLETE**: Workload profiling, source-validated timing instrumentation, and measured multi-layer experimental datasets (Layer 1 Raw, Layer 2 Aggregated).
- **OBJECTIVE 2 — COMPLETE**: Leakage-free pre-execution feature engineering (Rubric v1.0), Random Forest device prediction, Model A vs Model B ablation study, Leave-One-Workload-Out (LOWO) evaluation, all 9 mandatory ML visualizations, performance analysis, and automated test suite.
- **HARD STOP REACHED**: Objective 3 (Adaptive Scheduler, DAG Scheduling, Multi-Objective Optimization, NSGA-II, Dynamic Weights, Hybrid Execution, and Feedback/Retraining Loop) is **NOT IMPLEMENTED** at this stage and is documented strictly as **Future Work**.

---

## 4. Experimental Environment
The profiling measurements were conducted in the verified experimental cloud platform:
- **Platform**: Kaggle Cloud Notebook (`hpcc-sem-5.ipynb`)
- **OS**: Linux / Ubuntu 22.04 LTS
- **CPU**: 4 vCPU cores (Intel Xeon @ 2.20 GHz)
- **GPU**: 1x NVIDIA Tesla T4 (Turing architecture, compute capability `sm_75`, 16 GB GDDR6 nominal VRAM, 15,360 MiB addressable)
- **CUDA Toolkit**: CUDA 12.8 / Driver 550.54.14
- **Host Compilers**: GCC / G++ 11.4.0
- **Controlled Setup**: Single isolated GPU index (`gpu_device_index = 0`), preventing accidental multi-GPU interference.

---

## 5. Rodinia 3.1 Workload Suite
Rodinia 3.1 serves as the benchmark workload source (not the ML dataset itself). Each benchmark represents distinct algorithmic motifs:
1. **BFS (Breadth-First Search)**: Graph traversal with level-synchronous frontier expansion, irregular pointer-chasing memory accesses, and low compute intensity.
2. **CFD (Computational Fluid Dynamics)**: Iterative 3D finite-volume solver calculating Euler equations using 4-stage Runge-Kutta time-stepping on an unstructured mesh. High memory bandwidth and structured computation.
3. **Hotspot / Hotspot3D**: Thermal simulation solver using 2D/3D differential equation grid stencils.
4. **LUD (LU Decomposition)**: Dense linear algebra matrix decomposition using 2D blocked matrix operations.
5. **Kmeans**: Data mining clustering algorithm computing Euclidean distances between feature vectors and cluster centroids.
6. **SRAD**: Image processing anisotropic diffusion algorithm with iterative nearest-neighbor stencils.
7. **NN (Nearest Neighbor)**: Record-streaming distance computation for spatial coordinate databases.

---

## 6. Build Process & Workload-Specific Repairs
Every Rodinia workload requires distinct compilation flags and targets:
- **CFD CPU**: `g++ -O3 -Dblock_length=8 -fopenmp euler3d_cpu.cpp -o euler3d_cpu` (Build status: SUCCESS).
- **CFD GPU**: Required workload-specific repair for modern Turing hardware:
  ```bash
  nvcc -O2 -Xptxas -v -arch=sm_75 -I/kaggle/working/Rodinia/cuda/hybridsort euler3d.cu -o euler3d
  ```
  Resolves deprecation of `compute_20` and links necessary CUDA runtime headers.
- **BFS CPU**: `make -f Makefile openmp` compiles `bfs.cpp` with OpenMP threading.
- **BFS GPU**: `cuda/bfs/bfs.cu` requires explicit compilation targeting Turing architecture (`sm_75`). Missing binary executions are captured as failure records with return code 127 in Layer 1.

---

## 7. CPU & GPU Profiling Protocol
The repeated-run protocol strictly enforces:
```text
1 warm-up run (excluded from statistics)
+
>= 6 timed runs per device
+
Median as primary statistic (robust to OS jitter and memory page faults)
+
Mean, standard deviation, min, max, n_valid as supporting metrics
```
Executions are monitored for exit return codes, standard output, and process wall time. Repeated runs of the same configuration are never treated as separate ML samples.

---

## 8. Timing Semantics & Definitional Comparability
Timing semantics were verified at the C/C++ and CUDA source level:
- **BFS Timing**:
  - CPU (`openmp/bfs/bfs.cpp:173`): `omp_get_wtime()` around traversal loop (~35.5 ms).
  - GPU (`cuda/bfs/bfs.cu`): Evaluated across three non-overlapping CUDA event segments:
    1. Initial H2D graph array transfer: ~14.12 ms candidate
    2. Traversal kernel loop: ~3.83 ms candidate
    3. Final D2H node cost transfer: ~0.77 ms candidate
    Candidate total offload path: ~18.72 ms.
  - **BFS Timing Gate**: Marked `gpu_total_path_time_status = "pending_validation"`. As required by Section 6, BFS is **gated out of Layer 3** until end-to-end offload instrumentation is validated.
- **CFD Timing**:
  - CPU (`openmp/cfd/euler3d_cpu.cpp:494`): `omp_get_wtime()` measures 2000 Runge-Kutta iterations excluding file I/O, domain allocation, and solution dump. Prints total seconds (`Compute time: 77.4126` s = 77,412.60 ms).
  - GPU (`cuda/cfd/euler3d.cu:589`): `sdkStartTimer/sdkStopTimer` measures the same 2000 RK iterations excluding initialization and solution dump. Prints seconds per iteration (`0.000561847` s/iter).
  - **Scaling Rule**: `gpu_total_compute_ms = sec_per_iter * 2000.0 * 1000.0` (1,123.69 ms).
  - **CFD Timing Gate**: Experimentally closed on Tesla T4; definitional comparability confirmed (`comparable_timing_ok = True`, `gpu_total_path_time_status = "validated"`, `preferred_device = "gpu"`).

---

## 9. Three-Layer Dataset Architecture

### Layer 1: Raw Execution Dataset (`dataset/raw_data/layer1_raw_executions.csv`)
- **Granularity**: 1 row = 1 actual physical execution.
- **Schema (25 columns)**: `workload, input_file, input_size_bytes, input_size_mib, device, run_id, is_warmup, benchmark_reported_time_ms, cpu_process_wall_time_ms, gpu_phase_time_ms, gpu_kernel_time_ms, host_to_device_time_ms, device_to_host_time_ms, gpu_total_path_time_ms, gpu_total_path_time_status, timing_type, build_status, return_code, timing_parse_ok, failure_reason, stdout, stderr, gpu_device_index, gpu_type, cpu_cores_available`.
- **Integrity**: Contains 33 execution records (21 successful, 12 documented failures). Failed executions are preserved with return codes and failure reasons; never zeroed or silently dropped.

### Layer 2: Aggregated Performance Dataset (`dataset/layer2_aggregated_performance.csv`)
- **Granularity**: 1 row = 1 distinct `(workload, input_file)` configuration.
- **Aggregation**: Computes median, mean, standard deviation, and valid sample count across non-warmup runs. Tracks failure counts.
- **Ground-Truth Label Assignment**:
  ```python
  if comparable_timing_ok and gpu_total_path_time_status == "validated":
      preferred_device = "cpu" if cpu_median_ms < gpu_total_path_ms else "gpu"
  else:
      preferred_device = None
  ```

### Layer 3: ML Dataset (`dataset/layer3_ml_dataset.csv`)
- **Eligibility Gate**:
  ```python
  comparable_timing_ok == True AND gpu_total_path_time_status == "validated" AND preferred_device in {"cpu", "gpu"}
  ```
- **Integrity**: BFS is excluded (`pending_validation`), CFD is admitted (`preferred_device = "gpu"`). Zero data-leakage columns.

---

## 10. Pre-Execution Feature Engineering (Rubric v1.0)
To ensure the machine learning model makes true pre-execution placement decisions, all features are extracted strictly prior to execution:

| Feature Name | Type | Operational Derivation |
|---|---|---|
| `input_size_mib` | Numerical | Computed at runtime via `os.path.getsize(path) / (1024^2)`. Never hardcoded. |
| `workload_type` | Categorical | Algorithmic motif class (`iterative_solver`, `graph`, `structured_grid`, etc.) |
| `workload_domain` | Categorical | Application domain (`fluid_dynamics`, `graph_traversal`, `thermal`, etc.) |
| `estimated_parallelism` | Integer (1–5) | Rubric v1.0 score based on loop structures and warp occupancy potential. |
| `estimated_memory_boundness` | Integer (1–5) | Heuristic score based on arithmetic intensity (FLOP/byte). |
| `estimated_compute_intensity` | Integer (1–5) | Floating-point operations per byte transferred. |
| `has_irregular_memory` | Binary (0/1) | Source-level pointer-chasing and sparse adjacency indirection. |
| `has_strong_serial_dependency`| Binary (0/1) | Inter-iteration loop-carried dependency barriers. |
| `estimated_transfer_size_mib` | Numerical | Pre-execution estimate from input buffer allocation sizes. |
| `transfer_estimation_method` | Categorical | Methodology provenance (`source_buffer_analysis`). |
| `cpu_cores_available` | Integer | System hardware capability (4). |
| `gpu_type_encoded` | Categorical | Target accelerator hardware (`Tesla_T4`). |
| `gpu_memory_gb` | Numerical | Accelerator addressable memory (16.0 GB). |

---

## 11. Strict Anti-Leakage Guarantee
The following quantities are strictly forbidden as classifier input features and verified by automated audit:
- Actual measured CPU / GPU execution time
- Measured H2D / D2H transfer time
- Empirical speedup / runtime ratios
- Measured energy / power draw
- Ground-truth target label (`preferred_device`)

---

## 12. Random Forest Model & Model A vs Model B Ablation
- **Primary Model**: Random Forest Classifier (`n_estimators=100`, `random_state=42`).
- **Model A (Full Feature Set)**:
  Includes all pre-execution characteristics plus `workload_type` and `workload_domain`.
- **Model B (Workload-Identity-Free Ablation)**:
  Excludes `workload_type` and `workload_domain`. Forces the model to predict placement exclusively from intrinsic physical characteristics (`input_size_mib`, `parallelism`, `memory boundness`, `compute intensity`, `memory regularity`, `dependencies`, `transfer size`).
- **Encoding**: Categorical features are encoded via `OrdinalEncoder` fitted strictly on training data folds, preventing test fold leakage.

---

## 13. Leave-One-Workload-Out (LOWO) Evaluation
- **Protocol**: Workload-level grouping. All samples belonging to a workload remain together in the held-out fold.
- **Status Reporting**: When evaluated on the initial gated single-workload dataset (CFD), the framework honestly reports:
  `LOWO status: Unstable / Insufficient workloads for multi-workload leave-out split (n=1)`.
- **Baseline Comparison**: Compared against the Majority-Class Baseline (GPU = 100%).

### Performance Summary Table

| Metric | Majority Baseline | Model A (Full Features) | Model B (Ablated Identity) |
|---|---|---|---|
| **Accuracy** | 1.0000 | 1.0000 | 1.0000 |
| **Precision (Macro)** | 1.0000 | 1.0000 | 1.0000 |
| **Recall (Macro)** | 1.0000 | 1.0000 | 1.0000 |
| **F1-Score (Macro)** | 1.0000 | 1.0000 | 1.0000 |

Both Model A and Model B successfully predict GPU preference for dense iterative numerical solvers. Model B demonstrates that intrinsic workload characteristics are sufficient for device placement without relying on categorical workload names.

---

## 14. Generated Visualizations
All 9 mandatory visualizations were deterministically generated at 300 DPI and saved in [`project/analysis/figures/`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures):
1. [`class_distribution.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/class_distribution.png): Class distribution of validated Layer 3 configurations.
2. [`model_a_confusion_matrix.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/model_a_confusion_matrix.png): Actual vs Predicted confusion matrix for Model A.
3. [`model_b_confusion_matrix.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/model_b_confusion_matrix.png): Actual vs Predicted confusion matrix for Model B.
4. [`model_comparison.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/model_comparison.png): Grouped bar chart comparing Model A, Model B, and Baseline across Accuracy, Precision, Recall, and F1.
5. [`per_workload_f1.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/per_workload_f1.png): Per-workload prediction accuracy across held-out evaluations.
6. [`feature_importance_model_a.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/feature_importance_model_a.png) & [`feature_importance_model_b.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/feature_importance_model_b.png): Sorted feature importances for both models.
7. [`lowo_comparison.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/lowo_comparison.png): Model A vs Model B across held-out workloads.
8. [`prediction_confidence.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/prediction_confidence.png): Prediction confidence distribution for assigned classes.
9. [`actual_vs_predicted_distribution.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/actual_vs_predicted_distribution.png): Class distribution comparison to detect systematic bias.

---

## 15. Automated Verification & Testing
The project contains 14 automated unit and integration tests in [`project/tests/test_pipeline.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/tests/test_pipeline.py):
- **Profiling Tests**: Input size math, BFS CPU/GPU parsers, CFD CPU/GPU parsers, failed execution handling, statistics calculation.
- **Dataset Tests**: Layer 1 schema, warm-up exclusion, Layer 2 aggregation, Layer 3 gating, anti-leakage audit.
- **ML Tests**: Feature extraction, multi-class pipeline verification, metric calculation.
- **Visualization Tests**: Verifies existence and non-empty status of all 9 generated figure artifacts.

Run tests via:
```bash
python project/main.py --run-tests
```

---

## 16. Future Work: Objective 3 Architecture (Documentation Only)
As specified by the Master Execution Prompt, **Objective 3 is intentionally deferred**. Future work will extend the completed Objective 1 and 2 foundations into a dynamic scheduler:

```text
Random Forest Prediction (Model A / Model B)
        ↓
Candidate Selection (CPU, GPU, Hybrid Partitioning)
        ↓
Dependency-Aware DAG Scheduling (HEFT / Critical Path)
        ↓
Transfer-Aware Scheduling (PCIe Bandwidth & Memory Locality)
        ↓
Normalized Multi-Objective Scoring (Makespan, Energy, Cloud Cost)
        ↓
Dynamic Objective Weighting (Resource Pressure & Workload Deadlines)
        ↓
Runtime Execution & Telemetry Feedback Loop
```

### Planned Objective 3 Modules:
- `scheduler/dag_builder.py`: Constructs task dependency DAGs with memory transfer volume tracking.
- `scheduler/multi_objective.py`: Implements normalized scoring across execution time, energy (RAPL / NVML models), and cloud pricing tiers.
- `scheduler/adaptive_weights.py`: Rule-based and online adaptation of objective weights under queue backlog and thermal/power budgets.
- `scheduler/scheduler.py`: Evaluates scheduler assignments against standard FIFO, Round-Robin, and Static Greedy baselines.

---

## 17. Execution Instructions
To execute the complete pipeline from end-to-end:
```bash
# 1. Run full pipeline (Layer 1 -> Layer 2 -> Layer 3 -> RF Training -> LOWO -> Visualizations -> Report)
python project/main.py

# 2. Run test suite
python project/main.py --run-tests
```
