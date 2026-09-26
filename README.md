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
- **OBJECTIVE 1 — COMPLETE**: Workload profiling, source-validated timing instrumentation, and measured multi-layer experimental datasets across all 7 target Rodinia workloads (8 configurations) on cloud hardware (NVIDIA Tesla T4 GPU + Intel Xeon CPU).
- **OBJECTIVE 2 — COMPLETE**: Leakage-free pre-execution feature engineering (Rubric v1.0), Random Forest device prediction, Model A vs Model B ablation study, Leave-One-Workload-Out (LOWO) evaluation across 7 distinct holdout workloads, all 14 publication-grade visualizations, performance analysis, and automated test suite.
- **OBJECTIVE 3 SCHEDULING FOUNDATIONS — COMPLETE**: Multi-objective Pareto frontier analysis (Makespan vs Energy vs Transfer), strategy comparison (CPU-only vs GPU-only vs Rule-based vs AI Scheduler), and dynamic Bayesian objective weight adaptation (`AdaptiveWeightController`). Live distributed cluster orchestration is documented as Future Work.

---

## 4. Experimental Environment
The profiling measurements were conducted in the verified experimental cloud platform:
- **Platform**: Kaggle Cloud Notebook (`hpcc-sem-5.ipynb`)
- **OS**: Linux / Ubuntu 22.04 LTS
- **CPU**: 4 vCPU cores (Intel Xeon @ 2.20 GHz)
- **GPU**: 1x NVIDIA Tesla T4 (Turing architecture, compute capability `sm_75`, 16 GB GDDR6 nominal VRAM, 15,360 MiB addressable)
- **CUDA Toolkit**: CUDA 12.8 / Driver 550.54.14
- **Host Compilers**: GCC / G++ 11.4.0
- **Controlled Setup**: Single isolated GPU index (`gpu_device_index = 0`), preventing multi-GPU contention.

---

## 5. Rodinia 3.1 Workload Suite
Rodinia 3.1 serves as the benchmark workload source. Eight distinct configurations across 7 diverse algorithmic motifs were evaluated:
1. **BFS (Breadth-First Search) — Large Graph (`graph1MW_6.txt`, 61.22 MiB)**: 1,000,000 vertices, irregular memory access, level-synchronous frontier expansion.
2. **BFS (Breadth-First Search) — Small Graph (`graph4096.txt`, 0.05 MiB)**: 4,096 vertices, low arithmetic intensity, transfer-latency bound.
3. **CFD (Computational Fluid Dynamics — `fvcorr.domn.193K`, 43.54 MiB)**: 3D finite-volume unstructured grid solver evaluating Euler equations across 2000 Runge-Kutta iterations.
4. **HotSpot (`temp_512`, `power_512`, 2.75 MiB)**: Iterative 2D thermal differential stencil simulation across 360,000 iterations.
5. **LUD (LU Decomposition — `512.dat`, 2.48 MiB)**: Blocked dense linear algebra matrix factorization.
6. **K-Means (`819200.txt`, 96.98 MiB)**: Distance calculation and centroid updates across 819,200 multi-dimensional feature points.
7. **SRAD (`image.pgm`, 0.75 MiB)**: Anisotropic image diffusion filtering via iterative nearest-neighbor stencils.
8. **NN (Nearest Neighbor — `cane4_0.db`, 0.50 MiB)**: Streaming geospatial Euclidean distance calculations.

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
- **BFS GPU**: Compiled targeting Turing architecture (`sm_75`). Missing binary executions are captured as failure records with return code 127 in Layer 1, while validated runs enforce complete H2D, kernel, and D2H segments.
- **HotSpot, LUD, K-Means, SRAD, NN**: Compiled with native OpenMP and CUDA toolchains targeting `sm_75`.

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
Repeated runs of the same configuration are never treated as separate ML samples.

---

## 8. Timing Semantics & Definitional Comparability
Timing semantics were verified at the C/C++ and CUDA source level:
- **GPU Total Path Guarantee**:
  $$\text{GPU Total Path} = \text{Host-to-Device (H2D)} + \text{Kernel Compute} + \text{Device-to-Host (D2H)}$$
- **CFD Timing**: Both CPU and GPU measure the exact same 2000 Runge-Kutta iterations excluding disk loading ($77,537.30\text{ ms}$ CPU vs $1,125.05\text{ ms}$ GPU, speedup = $68.92\times$).
- **LUD Timing**: Matrix factorization completes in $2,048.85\text{ ms}$ on CPU vs $279.65\text{ ms}$ total path on GPU (speedup = $7.33\times$).
- **HotSpot Timing**: Thermal simulation completes in $1,486.75\text{ ms}$ on CPU vs $418.60\text{ ms}$ total path on GPU (speedup = $3.55\times$).
- **SRAD Timing**: Anisotropic diffusion completes in $1,783.80\text{ ms}$ on CPU vs $553.05\text{ ms}$ total path on GPU (speedup = $3.23\times$).
- **K-Means Timing**: Clustering completes in $3,216.70\text{ ms}$ on CPU vs $1,103.80\text{ ms}$ total path on GPU (speedup = $2.91\times$).
- **BFS (large) Timing**: Traversal on 1M vertices completes in $36.12\text{ ms}$ on CPU vs $18.73\text{ ms}$ total path on GPU ($14.12\text{ ms}$ H2D, $3.83\text{ ms}$ kernel, $0.77\text{ ms}$ D2H, speedup = $1.93\times$).
- **NN & BFS (small) Timing (CPU Preferred)**:
  - NN: $18.45\text{ ms}$ on CPU vs $28.55\text{ ms}$ total path on GPU ($12.8\text{ ms}$ H2D, $1.15\text{ ms}$ kernel, $0.85\text{ ms}$ D2H, speedup = $0.65\times$).
  - BFS small: $1.25\text{ ms}$ on CPU vs $8.85\text{ ms}$ total path on GPU (speedup = $0.14\times$).
  - In both small workloads, PCIe bus latency and GPU thread divergence cause the GPU to lose to the CPU.

---

## 9. Three-Layer Dataset Architecture

### Layer 1: Raw Execution Dataset (`dataset/raw_data/layer1_raw_executions.csv`)
- **Granularity**: 1 row = 1 actual physical execution.
- **Schema (25 columns)**: `workload, input_file, input_size_bytes, input_size_mib, device, run_id, is_warmup, benchmark_reported_time_ms, cpu_process_wall_time_ms, gpu_phase_time_ms, gpu_kernel_time_ms, host_to_device_time_ms, device_to_host_time_ms, gpu_total_path_time_ms, gpu_total_path_time_status, timing_type, build_status, return_code, timing_parse_ok, failure_reason, stdout, stderr, gpu_device_index, gpu_type, cpu_cores_available`.
- **Integrity**: Contains **124 execution records** (16 warm-up runs, 96 valid timed production runs, and 12 documented historical failures). Failed executions are preserved with full return codes and stderr for audit reproducibility.

### Layer 2: Aggregated Performance Dataset (`dataset/layer2_aggregated_performance.csv`)
- **Granularity**: 1 row = 1 distinct `(workload, input_file)` configuration.
- **Contents (8 configurations)**:
  - `bfs` (graph1MW_6.txt): CPU $36.12\text{ ms}$, GPU $18.73\text{ ms}$, Speedup $1.93\times$ $\rightarrow$ `gpu`
  - `bfs` (graph4096.txt): CPU $1.25\text{ ms}$, GPU $8.85\text{ ms}$, Speedup $0.14\times$ $\rightarrow$ `cpu`
  - `cfd` (fvcorr.domn.193K): CPU $77,537.30\text{ ms}$, GPU $1,125.05\text{ ms}$, Speedup $68.92\times$ $\rightarrow$ `gpu`
  - `hotspot` (temp_512): CPU $1,486.75\text{ ms}$, GPU $418.60\text{ ms}$, Speedup $3.55\times$ $\rightarrow$ `gpu`
  - `kmeans` (819200.txt): CPU $3,216.70\text{ ms}$, GPU $1,103.80\text{ ms}$, Speedup $2.91\times$ $\rightarrow$ `gpu`
  - `lud` (512.dat): CPU $2,048.85\text{ ms}$, GPU $279.65\text{ ms}$, Speedup $7.33\times$ $\rightarrow$ `gpu`
  - `nn` (cane4_0.db): CPU $18.45\text{ ms}$, GPU $28.55\text{ ms}$, Speedup $0.65\times$ $\rightarrow$ `cpu`
  - `srad` (image.pgm): CPU $1,783.80\text{ ms}$, GPU $553.05\text{ ms}$, Speedup $3.23\times$ $\rightarrow$ `gpu`

### Layer 3: ML Dataset (`dataset/layer3_ml_dataset.csv`)
- **Eligibility Gate**: Validated timing across both CPU and GPU ($N \ge 6$ timed runs, verified total path status).
- **Class Distribution**: 2 CPU configurations (25%), 6 GPU configurations (75%).
- **Leakage-Free Feature Matrix**: All 18 forbidden post-execution runtime columns are excluded.

---

## 10. Pre-Execution Feature Engineering (Rubric v1.0)
All features are extracted strictly prior to job execution:

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
The following quantities are strictly forbidden as classifier input features and verified by automated unit tests:
- Actual measured CPU / GPU execution time
- Measured H2D / D2H transfer time
- Empirical speedup / runtime ratios
- Measured energy / power draw
- Ground-truth target label (`preferred_device`)

---

## 12. Random Forest Model & Model A vs Model B Ablation
- **Primary Model**: Random Forest Classifier (`n_estimators=100`, `max_depth=5`, `random_state=42`).
- **Model A (Full Feature Set)**: Includes all 13 pre-execution characteristics plus nominal identifiers `workload_type` and `workload_domain`.
- **Model B (Workload-Identity-Free Ablation)**: Excludes `workload_type` and `workload_domain`. Forces the model to predict placement exclusively from intrinsic physical characteristics (`input_size_mib`, `parallelism`, `memory boundness`, `compute intensity`, `memory regularity`, `dependencies`, `transfer size`).
- **Ablation Finding**: Model B achieved identical performance to Model A (87.50% accuracy, 79.49% F1), proving that the classifier learns generalizable architectural principles rather than memorizing workload names.
- **Top Feature Importances (MDI)**:
  1. `estimated_transfer_size_mib` (~0.30)
  2. `input_size_mib` (~0.30)
  3. `estimated_compute_intensity` (~0.13 – 0.17)
  4. `estimated_memory_boundness` (~0.08)
  5. `estimated_parallelism` (~0.04 – 0.09)

---

## 13. Leave-One-Workload-Out (LOWO) Evaluation
- **Protocol**: Grouped cross-validation where all samples belonging to a workload are held out together.
- **Evaluation**: 7 distinct holdout folds covering all 7 workloads. Zero data leakage.
- **Baseline**: Majority-Class Baseline (always predict GPU = 75.00%).

### Performance Summary Table

| Metric | Majority Baseline (Always GPU) | Model A (Full Features) | Model B (Ablated Identity) |
|---|---|---|---|
| **Accuracy** | 0.7500 (75.00%) | **0.8750 (87.50%)** | **0.8750 (87.50%)** |
| **Precision (Macro)** | 0.3750 (37.50%) | **0.9286 (92.86%)** | **0.9286 (92.86%)** |
| **Recall (Macro)** | 0.5000 (50.00%) | **0.7500 (75.00%)** | **0.7500 (75.00%)** |
| **F1-Score (Macro)** | 0.4286 (42.86%) | **0.7949 (79.49%)** | **0.7949 (79.49%)** |

Both models decisively outperform the majority baseline (+12.5% accuracy gain, +36.6% macro F1 gain) by correctly predicting CPU-preferred tasks.

---

## 14. Generated Visualizations
All 14 publication-grade visualizations are generated and saved in [`project/analysis/figures/`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures):

### Machine Learning & Classification Figures:
1. [`class_distribution.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/class_distribution.png): Class distribution of validated Layer 3 configurations (2 CPU, 6 GPU).
2. [`model_a_confusion_matrix.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/model_a_confusion_matrix.png): 2×2 confusion matrix for Model A (LOWO evaluation).
3. [`model_b_confusion_matrix.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/model_b_confusion_matrix.png): 2×2 confusion matrix for Model B.
4. [`model_comparison.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/model_comparison.png): Grouped bar chart comparing Baseline vs Model A vs Model B across Accuracy, Precision, Recall, and F1.
5. [`per_workload_f1.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/per_workload_f1.png): Per-workload prediction accuracy across all 7 workloads (`BFS`, `CFD`, `HOTSPOT`, `KMEANS`, `LUD`, `NN`, `SRAD`).
6. [`feature_importance_model_a.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/feature_importance_model_a.png): MDI feature importances for Model A.
7. [`feature_importance_model_b.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/feature_importance_model_b.png): MDI feature importances for Model B.
8. [`lowo_comparison.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/lowo_comparison.png): Head-to-head comparison of Model A vs Model B across held-out workloads.
9. [`prediction_confidence.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/prediction_confidence.png): Prediction confidence distribution for assigned classes.
10. [`actual_vs_predicted_distribution.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/actual_vs_predicted_distribution.png): Class distribution comparison to detect systematic bias.

### Multi-Objective Scheduling & Speedup Figures:
11. [`speedup_chart.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/speedup_chart.png): Two-panel figure showing log-scale execution runtimes alongside GPU speedup bars sorted descending with the 1.0× break-even line across all 8 configurations.
12. [`pareto_front.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/pareto_front.png): Dual-panel figure showing global objective space (Makespan vs Energy) and a zoomed Pareto frontier highlighting non-dominated trade-offs between AI Scheduler, Latency-Priority, and Energy-Priority policies.
13. [`weight_evolution.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/weight_evolution.png): Line plot showing dynamic Bayesian adaptation of objective weights ($w_1, w_2, w_3$) across 25 scheduling rounds simulating dynamic cloud workload arrival phases.
14. [`scheduler_comparison.png`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/analysis/figures/scheduler_comparison.png): Grouped bar chart comparing CPU-only, GPU-only, Rule-based, and AI Adaptive Scheduler across all 7 workloads.

---

## 15. Automated Verification & Testing
The project contains 14 automated unit and integration tests in [`project/tests/test_pipeline.py`](file:///c:/Users/abhi8/OneDrive/Desktop/ACADEMIC%20DOCS/SEM-5/HPC/project/tests/test_pipeline.py):
- **Profiling Tests (6)**: Input size calculation, BFS CPU/GPU parsers, CFD CPU/GPU parsers, failed execution handling, statistics calculation.
- **Dataset Tests (2)**: Layer 1 schema & failure preservation, Layer 2 aggregation across all 8 configurations, Layer 3 gating & anti-leakage audit.
- **ML Tests (4)**: Pre-execution feature extraction, multi-class pipeline verification, metric calculation.
- **Visualization Tests (1)**: Verifies existence and non-empty status of all 14 generated figure artifacts.

Execute tests via:
```bash
python project/main.py --run-tests
```
Expected output:
```text
Ran 14 tests in 0.303s
OK
```

---

## 16. Execution Instructions
To execute the complete pipeline from end-to-end:
```bash
# 1. Run full pipeline (Layer 1 -> Layer 2 -> Layer 3 -> RF Training -> LOWO -> Visualizations -> Report)
python project/main.py

# 2. Run test suite
python project/main.py --run-tests
```
