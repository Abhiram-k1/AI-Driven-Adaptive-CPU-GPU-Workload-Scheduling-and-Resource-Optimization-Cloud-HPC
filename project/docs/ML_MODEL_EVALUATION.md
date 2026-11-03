# MACHINE LEARNING MODEL EVALUATION & ABLATION STUDY
## Pre-Execution Device Classification, LOWO Cross-Validation, and Generalization Analysis

---

## 1. Feature Engineering & Frozen Rubric v1.0

Workload placement decisions must occur **strictly prior to execution**. Admitting any measured runtime, observed speedup, or power consumption into the feature vector constitutes catastrophic data leakage.

### 1.1 Pre-Execution Feature Specifications

| Feature Name | Type | Extraction Method | Description |
|---|---|---|---|
| `input_size_mib` | Numerical | Filesystem stat | Total bytes of input dataset in MiB |
| `workload_type` | Categorical | Source analysis | Algorithmic family (graph, stencil, dense clustering, etc.) |
| `workload_domain` | Categorical | Metadata | Application domain (bioinformatics, fluid dynamics, etc.) |
| `estimated_parallelism` | Discrete (1-5) | Algorithmic rubric | Degree of thread/warp concurrency exploitable |
| `estimated_memory_boundness`| Discrete (1-5) | Algorithmic rubric | Arithmetic intensity inverse (FLOP/byte ratio) |
| `estimated_compute_intensity`| Discrete (1-5) | Algorithmic rubric | Floating-point operations per loaded cache line |
| `has_irregular_memory` | Binary (0/1) | Source analysis | Indirect indexing (pointer chasing, graph adjacency lists) |
| `has_strong_serial_dependency`| Binary (0/1) | Source analysis | Inter-iteration barrier / wavefront synchronization |
| `estimated_transfer_size_mib`| Numerical | Buffer analysis | Estimated bytes transferred over PCIe interconnect |
| `transfer_estimation_method` | Categorical | Schema rubric | Buffer sizing technique (`source_buffer_analysis`) |
| `cpu_cores_available` | Discrete | Host probe | Host logical core count allocated (e.g. 4) |
| `gpu_type_encoded` | Categorical | Host probe | Target accelerator model (e.g. `Tesla_T4`) |
| `gpu_memory_gb` | Numerical | Device probe | Accelerator onboard memory capacity in GB (16.0) |

### 1.2 Anti-Leakage Assertion
The feature extraction pipeline validates every record against the prohibited post-execution blacklist:

```python
FORBIDDEN_LEAKAGE_COLUMNS = {
    "cpu_median_ms", "cpu_mean_ms", "cpu_std_ms", "cpu_process_wall_time_ms",
    "gpu_median_ms", "gpu_mean_ms", "gpu_std_ms", "gpu_phase_time_ms",
    "gpu_kernel_time_ms", "host_to_device_time_ms", "device_to_host_time_ms",
    "gpu_total_path_ms", "measured_speedup", "measured_energy", "measured_power",
    "actual_speedup", "post_execution_timing_ratio", "empirical_preferred_device"
}
```

---

## 2. Model Architectures & Ablation Methodology

To evaluate whether the machine learning models learn genuine algorithmic execution characteristics rather than simply memorizing application names, we implement three distinct model variants:

- **Model A (Full Random Forest)**:
  - Considers all 13 pre-execution features, including `workload_type` and `workload_domain`.
  - Hyperparameters: 100 decision trees, max_depth=5, random_state=42.

- **Model B (Ablated Random Forest — Workload-Identity-Free)**:
  - Strictly excludes `workload_type` and `workload_domain`.
  - Trained solely on intrinsic computational metrics: arithmetic intensity, memory access irregularity, serialization barriers, input size, and estimated transfer overhead.
  - Tests whether the model can transfer knowledge to completely novel algorithmic paradigms.

- **Model C (Gradient Boosted Decision Trees)**:
  - Multi-algorithm baseline using sequential gradient boosting (n_estimators=50, max_depth=3).

- **Majority-Class Baseline**:
  - Always predicts the majority class (GPU, 75.00%).

---

## 3. Leave-One-Workload-Out (LOWO) Cross-Validation

Standard k-fold cross-validation is vulnerable to data leakage when multiple input configurations of the same workload appear in both training and test sets.

We implement **Grouped Leave-One-Workload-Out (LOWO) Cross-Validation**:
- In each fold $k$, all samples belonging to workload $W_k$ are completely withheld from training.
- Categorical encoders (`OrdinalEncoder`) are fitted strictly on the remaining training workloads.
- The held-out workload is evaluated as a totally unseen workload.

---

## 4. Empirical Evaluation Results

Evaluated across the 7 empirical Rodinia workloads on NVIDIA Tesla T4:

### 4.1 Global Metric Comparison Table

| Metric | Majority Baseline | Model A (RF Full) | Model B (RF Ablated) | Model C (Gradient Boosting) |
|---|---|---|---|---|
| **Accuracy** | 75.00% | **87.50%** | **87.50%** | 75.00% |
| **Precision (Macro)** | 37.50% | **92.86%** | **92.86%** | 66.67% |
| **Recall (Macro)** | 50.00% | **75.00%** | **75.00%** | 66.67% |
| **F1-Score (Macro)**| 42.86% | **79.49%** | **79.49%** | 66.67% |
| **CPU Class F1** | 0.00% | **66.67%** | **66.67%** | 50.00% |
| **GPU Class F1** | 85.71% | **92.31%** | **92.31%** | 85.71% |

### 4.2 Confusion Matrices

$$\text{Model A / Model B} = \begin{bmatrix} 1 & 1 \\ 0 & 6 \end{bmatrix} \quad (\text{TN=1, FP=1, FN=0, TP=6})$$

$$\text{Majority Baseline} = \begin{bmatrix} 0 & 2 \\ 0 & 6 \end{bmatrix} \quad (\text{TN=0, FP=2, FN=0, TP=6})$$

### 4.3 Per-Workload LOWO Breakdown

| Held-out Workload | Input File | True Preferred | Model A Prediction | Model B Prediction | Confidence | Result |
|---|---|---|---|---|---|---|
| **BFS (Large)** | `graph1MW_6.txt` | GPU | GPU | GPU | 0.82 | **CORRECT** |
| **BFS (Small)** | `graph4096.txt` | CPU | CPU | CPU | 0.74 | **CORRECT** |
| **CFD** | `fvcorr.domn.193K` | GPU | GPU | GPU | 0.94 | **CORRECT** |
| **Hotspot** | `temp_512` | GPU | GPU | GPU | 0.98 | **CORRECT** |
| **KMeans** | `819200.txt` | GPU | GPU | GPU | 0.95 | **CORRECT** |
| **LUD** | `512.dat` | GPU | GPU | GPU | 0.97 | **CORRECT** |
| **SRAD** | `image.pgm` | GPU | GPU | GPU | 0.88 | **CORRECT** |
| **NN** | `cane4_0.db` | CPU | GPU | GPU | 0.61 | **MISCLASSIFIED** |

---

## 5. Diagnostic Discussion: The NN Misclassification

The Nearest Neighbor (`nn`) benchmark represents a crucial edge case:
- It evaluates 2D Euclidean distance across cane storm records.
- Computationally, distance calculation is embarrassingly parallel with zero serial dependency.
- However, the dataset size on disk is tiny (0.50 MiB) while requiring host memory staging.
- As an end-to-end offload, CPU takes 18.45 ms while GPU takes 28.55 ms (PCIe transfer accounts for over 70% of total elapsed time).
- Both Model A and Model B predicted GPU with lower confidence (0.61).
- This misclassification highlights the subtle balance between computational parallelism and PCIe bus transfer latency on small datasets.

---

## 6. Feature Importance Ranking (Model A)

1. `estimated_transfer_size_mib` (0.3034)
2. `input_size_mib` (0.2991)
3. `estimated_compute_intensity` (0.1263)
4. `estimated_memory_boundness` (0.0798)
5. `workload_type` (0.0610)
6. `estimated_parallelism` (0.0443)
7. `workload_domain` (0.0423)
8. `has_irregular_memory` (0.0269)
9. `has_strong_serial_dependency` (0.0168)

