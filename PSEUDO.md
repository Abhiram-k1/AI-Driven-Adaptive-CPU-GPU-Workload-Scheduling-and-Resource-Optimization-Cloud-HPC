# PSEUDO.md — Capstone Plain-Language Guide & Viva Cheatsheet

**Project Title:** AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduling and Resource Optimization for Cloud HPC  
**Purpose:** This document explains the entire implemented system in simple, layman-friendly language. It allows you to explain every concept, pipeline stage, design decision, and empirical result clearly in an exam, presentation, or viva without needing to read raw code.

---

# PART 1: THE BIG PICTURE

### What is this project about?
Modern high-performance computing (HPC) cloud servers contain both **CPUs** (general-purpose host processors) and **GPUs** (massively parallel accelerator processors).
- Some programs run dramatically faster on GPUs (e.g., fluid simulations, thermal stencils, matrix factorizations) because they have thousands of CUDA cores that execute math simultaneously.
- Other programs run faster on CPUs (e.g., small nearest-neighbor queries, small graph searches) because transferring data over the PCIe bus takes longer than the actual computation, and GPU threads suffer from divergence and memory stalls.
- Historically, system engineers choose execution devices using rigid rules of thumb or trial-and-error.

### What is our solution?
We build a machine learning framework that evaluates a workload's characteristics **strictly before execution** (such as file size, arithmetic intensity, memory access regularity, and estimated PCIe transfer volume) and predicts whether the CPU or GPU will yield the fastest total runtime. This prediction directly drives an adaptive multi-objective scheduler that balances makespan, energy consumption, and PCIe transfer overhead.

---

# PART 2: STEP-BY-STEP IMPLEMENTATION PIPELINE

---

## STEP 1 — RUNNING THE WORKLOADS (EMPIRICAL PROFILING)

### What are we doing?
We profile benchmark applications from the standard **Rodinia 3.1** benchmark suite on both CPU (OpenMP) and GPU (CUDA) on cloud hardware (Intel Xeon 4 vCPU cores + NVIDIA Tesla T4 16GB GPU).

### What workloads are covered?
We profile **8 distinct configurations across 7 representative Rodinia workloads**:
1. **BFS (Breadth-First Search) — Large Graph (`graph1MW_6.txt`, 61.22 MiB)**: 1 million vertices, irregular memory access.
2. **BFS (Breadth-First Search) — Small Graph (`graph4096.txt`, 0.05 MiB)**: 4,096 vertices, low parallelism.
3. **CFD (Computational Fluid Dynamics — `fvcorr.domn.193K`, 43.54 MiB)**: 3D finite-volume Euler solver, 2000 Runge-Kutta iterations.
4. **HotSpot (`temp_512`, `power_512`, 2.75 MiB)**: 2D thermal differential grid simulation across 360,000 iterations.
5. **LUD (LU Decomposition — `512.dat`, 2.48 MiB)**: Blocked dense linear algebra matrix factorization.
6. **K-Means (`819200.txt`, 96.98 MiB)**: Multi-dimensional cluster centroid distance calculations across 819,200 data points.
7. **SRAD (`image.pgm`, 0.75 MiB)**: Anisotropic image diffusion filtering across iterative 2D stencils.
8. **NN (Nearest Neighbor — `cane4_0.db`, 0.50 MiB)**: Streaming geospatial Euclidean distance calculations.

### What happens in each run?
1. We compile the source code with architecture-specific compiler flags (`-O3 -fopenmp` for CPU, `nvcc -arch=sm_75 -O2` for GPU).
2. We execute **1 warm-up run** first. (Computers are slower on the first run due to OS disk page caching and driver initialization; discarding this warm-up prevents biased data).
3. We execute **at least 6 timed runs** on the CPU and **6 timed runs** on the GPU.
4. We capture stdout, stderr, process wall time, and the operating system exit return code.

### What comes out?
A comprehensive dataset of 124 physical executions capturing valid timings, warm-up runs, and historical audit failures.

---

## STEP 2 — TIMING VALIDATION GATES (APPLES-TO-APPLES COMPARABILITY)

### What are we doing?
We verify source-level timer instrumentation in C++ and CUDA to ensure CPU and GPU timings are strictly comparable.

### Why is this critical?
A common flaw in GPU academic literature is comparing total CPU wall time against GPU kernel-only time, ignoring the PCIe bus data transfer overhead (Host-to-Device and Device-to-Host). If transfer time is omitted, the GPU appears artificially fast.

### How do our gates work?
- **Total Path Instrumentation:** For GPU runs, total time must encompass:
  $$\text{GPU Total Path} = \text{Host-to-Device (H2D)} + \text{Kernel Compute} + \text{Device-to-Host (D2H)}$$
- **CFD Verification:** CPU OpenMP measures 2000 Runge-Kutta iterations excluding disk loading ($77,537.30\text{ ms}$). GPU CUDA measures the exact same 2000 iterations ($1,125.05\text{ ms}$). Both measure the exact same numerical phase.
- **BFS Verification:** On `graph1MW_6.txt`, CPU OpenMP executes traversal in $36.12\text{ ms}$. GPU CUDA requires $14.12\text{ ms}$ H2D, $3.83\text{ ms}$ kernel, and $0.77\text{ ms}$ D2H ($\text{Total Path} = 18.73\text{ ms}$). Both include full execution paths.
- **NN & Small BFS Verification:** For small workloads, PCIe setup and transfer times dominate over tiny kernel runtimes, revealing the break-even threshold where GPUs lose to CPUs.

---

## STEP 3 — LAYER 1: RAW EXECUTION DATASET

### What are we doing?
We log every single execution run into `dataset/raw_data/layer1_raw_executions.csv`. One row represents one physical program launch.

### What does it contain?
- **124 total execution rows** across 25 schema columns:
  - 16 warm-up runs (strictly tracked, tagged `is_warmup = True`).
  - 96 valid timed production runs (6 CPU + 6 GPU runs per configuration).
  - 12 documented historical failure runs (e.g. `./bfs: not found`, exit code 127) preserving the full audit trail.

---

## STEP 4 — LAYER 2: AGGREGATED PERFORMANCE DATASET

### What are we doing?
We condense repeated runs of the same configuration into a single statistical profile in `dataset/layer2_aggregated_performance.csv`.

### What numbers do we calculate?
- **Median Runtime:** Our primary metric because it is immune to outlier operating system jitter.
- **Mean & Standard Deviation:** Supporting metrics to prove statistical measurement stability.
- **Sample Counts:** Verifies that $N \ge 6$ valid runs support every statistical estimate.

### Summary of Layer 2 Measured Empirical Results:
| Workload | Input File | Input Size | CPU Median | GPU Total Path | GPU Speedup | Preferred Device |
|---|---|---|---|---|---|---|
| **CFD** | `fvcorr.domn.193K` | 43.54 MiB | 77,537.30 ms | 1,125.05 ms | **68.92×** | **GPU** |
| **LUD** | `512.dat` | 2.48 MiB | 2,048.85 ms | 279.65 ms | **7.33×** | **GPU** |
| **HOTSPOT** | `temp_512` | 2.75 MiB | 1,486.75 ms | 418.60 ms | **3.55×** | **GPU** |
| **SRAD** | `image.pgm` | 0.75 MiB | 1,783.80 ms | 553.05 ms | **3.23×** | **GPU** |
| **KMEANS** | `819200.txt` | 96.98 MiB | 3,216.70 ms | 1,103.80 ms | **2.91×** | **GPU** |
| **BFS (large)** | `graph1MW_6.txt` | 61.22 MiB | 36.12 ms | 18.73 ms | **1.93×** | **GPU** |
| **NN** | `cane4_0.db` | 0.50 MiB | 18.45 ms | 28.55 ms | **0.65×** | **CPU** |
| **BFS (small)** | `graph4096.txt` | 0.05 MiB | 1.25 ms | 8.85 ms | **0.14×** | **CPU** |

---

## STEP 5 — GROUND-TRUTH TARGET LABELLING

### What are we doing?
We assign the official target label (`preferred_device = "cpu"` or `"gpu"`) using the validated Layer 2 measurements:

$$
\text{preferred\_device} = \begin{cases} 
\text{"cpu"}, & \text{if } \text{Speedup} < 1.0 \quad (\text{CPU Median} < \text{GPU Total Path}) \\ 
\text{"gpu"}, & \text{if } \text{Speedup} \ge 1.0 \quad (\text{CPU Median} \ge \text{GPU Total Path}) 
\end{cases}
$$

### Class Balance:
- **CPU Labels:** 2 configurations (`NN`, `BFS small`) = **25%**
- **GPU Labels:** 6 configurations (`CFD`, `LUD`, `HOTSPOT`, `SRAD`, `KMEANS`, `BFS large`) = **75%**
This 1:3 ratio reflects real-world HPC systems: large parallel compute tasks benefit heavily from accelerators, while latency-sensitive, memory-transfer-dominated tasks belong on host CPUs.

---

## STEP 6 — PRE-EXECUTION FEATURE ENGINEERING (RUBRIC V1.0) & ANTI-LEAKAGE

### What are we doing?
We extract descriptive numerical and categorical properties of each task **strictly before it runs**, saving them into `dataset/layer3_ml_dataset.csv`.

### Why is Data Leakage Prevention crucial?
If an AI model is given post-execution metrics (like actual measured runtime, power draw, or kernel execution time), it is cheating. A cloud scheduler must make its placement decision *prior to job execution*.
- **Forbidden Columns (18 fields):** All actual runtimes, GPU phase times, H2D/D2H transfer durations, speedup values, and error logs are barred from the feature matrix.

### What pre-execution features are used?
1. `input_size_mib`: Size of input file in megabytes.
2. `workload_type`: Algorithmic class (`iterative_solver`, `graph`, `dense_linear_algebra`, etc.).
3. `workload_domain`: Scientific field (`fluid_dynamics`, `graph_traversal`, `data_mining`, etc.).
4. `estimated_parallelism`: 1 to 5 scale representing thread grid scalability.
5. `estimated_memory_boundness`: 1 to 5 scale representing memory bandwidth demand.
6. `estimated_compute_intensity`: 1 to 5 scale representing floating-point operations per byte.
7. `has_irregular_memory`: Binary flag (1 if pointer chasing / sparse indirection; 0 if contiguous).
8. `has_strong_serial_dependency`: Binary flag (1 if loop-carried barriers exist; 0 if independent).
9. `estimated_transfer_size_mib`: Pre-execution buffer size to be copied across PCIe.
10. `transfer_estimation_method`: Provenance label (`source_buffer_analysis`).
11. `cpu_cores_available`: Target host core count (4 cores).
12. `gpu_type_encoded`: Accelerator type (`Tesla_T4`).
13. `gpu_memory_gb`: Available GPU VRAM (16.0 GB).

---

## STEP 7 & 8 — RANDOM FOREST CLASSIFIER & MODEL A VS B ABLATION

### What are we doing?
We train a **Random Forest Classifier** (`n_estimators=100`, `max_depth=5`) to predict `preferred_device`.

### Why the Model A vs Model B Ablation Study?
- **Model A (Full Features):** Includes all 13 features, including nominal identifiers `workload_type` and `workload_domain`.
- **Model B (Ablated Intrinsic Features Only):** **Strips out** `workload_type` and `workload_domain`! It only sees the physics of the code (compute intensity, parallelism, memory regularity, transfer size, input size).

### What does the ablation prove?
If Model A performed well but Model B failed, the AI would merely be memorizing program names. Because **Model B achieved the exact same 87.50% accuracy and 79.49% F1 as Model A**, we prove that the classifier learns generalizable architectural principles that transfer to new, unseen algorithms!

### What features matter most according to the Random Forest?
1. `estimated_transfer_size_mib` (~0.30): Determines whether PCIe transfer overhead cancels GPU speedup.
2. `input_size_mib` (~0.30): Determines whether the problem scale is large enough to saturate GPU hardware.
3. `estimated_compute_intensity` (~0.13 – 0.17): Separates memory-bandwidth-bound tasks from math-heavy kernels.
4. `estimated_memory_boundness` (~0.08): Reflects cache locality and DRAM latency sensitivity.
5. `estimated_parallelism` (~0.04 – 0.09): Captures thread concurrency potential.

---

## STEP 9 — LEAVE-ONE-WORKLOAD-OUT (LOWO) CROSS-VALIDATION

### What are we doing?
We evaluate the model using **Leave-One-Workload-Out (LOWO)** cross-validation across all 7 distinct workloads.

### How does LOWO work?
In each round:
1. One entire workload (e.g., all BFS configurations) is held out in a test vault.
2. The Random Forest is trained exclusively on the remaining 6 workloads.
3. The trained model predicts the held-out workload.
4. This is repeated 7 times so every workload is tested as an unseen program.

### Why not standard K-Fold?
In HPC scheduling, standard K-Fold leaks information because training and testing folds would contain samples of the same program. LOWO guarantees zero cross-contamination and tests true generalization.

### Performance Results:
| Metric | Majority Baseline (Always GPU) | Model A (Full Features) | Model B (Intrinsic Only) |
|---|---|---|---|
| **Accuracy** | 75.00% | **87.50% (+12.5%)** | **87.50% (+12.5%)** |
| **Macro Precision** | 37.50% | **92.86%** | **92.86%** |
| **Macro Recall** | 50.00% | **75.00%** | **75.00%** |
| **Macro F1-Score** | 42.86% | **79.49% (+36.6%)** | **79.49% (+36.6%)** |

Both models decisively beat the majority baseline by correctly identifying CPU-preferred workloads rather than naively offloading everything to the GPU.

---

## STEP 10 — ALL 14 PUBLICATION-GRADE VISUALIZATIONS

The system automatically generates 14 high-resolution figures in `project/analysis/figures/`:

### Machine Learning & Classification Visualizations:
1. `class_distribution.png`: Bar chart of ground-truth classes (2 CPU, 6 GPU).
2. `model_a_confusion_matrix.png`: 2×2 confusion matrix for Model A (1 CPU correct, 1 CPU misclassified; 6 GPU correct).
3. `model_b_confusion_matrix.png`: 2×2 confusion matrix for Model B.
4. `model_comparison.png`: Grouped bar chart comparing Baseline vs Model A vs Model B across Accuracy, Precision, Recall, and F1.
5. `per_workload_f1.png`: Per-workload prediction accuracy across all 7 workloads (`BFS`, `CFD`, `HOTSPOT`, `KMEANS`, `LUD`, `NN`, `SRAD`).
6. `feature_importance_model_a.png`: Horizontal bar chart of MDI feature importances for Model A.
7. `feature_importance_model_b.png`: Horizontal bar chart of MDI feature importances for Model B.
8. `lowo_comparison.png`: Direct comparison of Model A vs Model B across held-out workloads.
9. `prediction_confidence.png`: Histogram showing assigned class probability for correct vs incorrect predictions.
10. `actual_vs_predicted_distribution.png`: Class distribution bias check.

### Multi-Objective Scheduling & Speedup Visualizations:
11. `speedup_chart.png`: Two-panel figure with runtimes on log-scale and GPU speedup bars sorted descending with the 1.0× break-even line.
12. `pareto_front.png`: Dual-panel figure showing global objective space (Makespan vs Energy) and a zoomed Pareto frontier highlighting the trade-off curve between AI Scheduler, Latency-Priority, and Energy-Priority policies.
13. `weight_evolution.png`: Line plot showing dynamic Bayesian adaptation of objective weights ($w_1$ Makespan, $w_2$ Energy, $w_3$ Transfer) across 25 scheduling rounds simulating dynamic cloud workload arrival phases.
14. `scheduler_comparison.png`: Grouped bar chart comparing execution time across CPU-only, GPU-only, Rule-based, and AI Adaptive Scheduler across all 7 workloads.

---

# PART 3: VIVA QUESTIONS & DIRECT ANSWERS

### Q1: Why did you previously see only one or two workloads, and how was that resolved?
**Answer:** In the initial exploratory Jupyter notebook, only CFD and BFS were executed, and BFS GPU failed because the binary was missing (`./bfs: not found`), which caused BFS to be gated out. We resolved this by establishing full empirical profiling across 8 configurations for all 7 target Rodinia workloads (`BFS` large & small, `CFD`, `HotSpot`, `LUD`, `K-Means`, `SRAD`, `NN`), preserving all historical failure logs for audit integrity, and validating the complete timing paths.

### Q2: Why are some workloads empirically CPU-preferred?
**Answer:** Workloads like NN ($0.65\times$) and small-graph BFS ($0.14\times$) run faster on CPU because the total GPU path includes PCIe data transfer time ($13.65\text{ ms}$ for NN, $4.55\text{ ms}$ for small BFS). Since the computation is small, PCIe communication dwarfs kernel compute. On small BFS, low vertex count also leads to severe GPU warp underutilization and thread divergence.

### Q3: Why did CFD achieve a 68.92× speedup on the GPU?
**Answer:** CFD (Euler 3D) calculates fluid dynamics across 193,000 mesh cells across 2000 Runge-Kutta iterations. Its regular spatial grid and high floating-point intensity allow thousands of CUDA threads to run simultaneously, amortizing PCIe transfer overhead.

### Q4: What does the Model A vs Model B ablation prove?
**Answer:** Model A includes program identity (`workload_type`, `workload_domain`), while Model B omits them. Because Model B achieves the exact same 87.50% accuracy and 79.49% F1 as Model A, it proves that the model makes decisions based on physical principles (arithmetic intensity, memory regularity, transfer size) rather than memorizing workload names.

### Q5: What is Data Leakage and how do you guarantee it is zero?
**Answer:** Data leakage occurs when post-execution knowledge (e.g., actual measured execution time, measured transfer time, or speedup) is included in the feature set. We enforce a strict pre-execution feature rule: all 13 features are calculated before the task starts. An automated test in `test_pipeline.py` verifies that all 18 forbidden runtime columns are completely absent.

### Q6: How does the AI Scheduler balance multiple objectives?
**Answer:** It uses multi-objective optimization to balance makespan (latency), energy consumption (kJ proxy), and PCIe transfer volume. Using Bayesian Online Learning (`AdaptiveWeightController`), it updates objective weights ($w_1, w_2, w_3$) dynamically in response to cloud workload arrival bursts (compute bursts vs transfer bottlenecks vs energy-saving modes).

---

# PART 4: SUMMARY OF PROJECT STATUS

### COMPLETED IN CURRENT IMPLEMENTATION (OBJECTIVES 1 & 2 + SCHEDULING FOUNDATIONS):
- [x] Full empirical profiling across all 7 Rodinia workloads (8 configurations)
- [x] 124 physical execution records in Layer 1 Raw Dataset (including warm-up runs and failure logs)
- [x] Layer 2 Aggregated Performance Dataset with verified timing gates
- [x] Ground-truth labelling with both CPU (25%) and GPU (75%) classes
- [x] Strict leakage-free Pre-Execution Feature Engineering (Rubric v1.0)
- [x] Layer 3 ML Dataset (8 verified configurations)
- [x] Random Forest Model A (Full) and Model B (Ablated)
- [x] Leave-One-Workload-Out (LOWO) cross-validation (87.5% accuracy, 79.5% macro F1)
- [x] All 14 publication-grade figures in `project/analysis/figures/`
- [x] Multi-objective Pareto frontier and speedup analysis
- [x] Dynamic Bayesian objective weight adaptation controller (`AdaptiveWeightController`)
- [x] Empirical performance analysis report answering all 10 project evaluation questions
- [x] 14/14 automated unit and integration tests passing (`python project/main.py --run-tests`)
