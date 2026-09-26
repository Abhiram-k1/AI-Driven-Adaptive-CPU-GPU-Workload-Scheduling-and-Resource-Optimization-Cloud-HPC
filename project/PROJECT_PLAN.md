# AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduling
# and Resource Optimization for Cloud HPC
## Near-Final Implementation-Gated Project Plan — SEM-5 HPC Capstone
### Incorporating corrections from PROJECT_PLAN_3, PROJECT_PLAN_4, and verified source-level analyses

---

## 1. Project Title

**AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduling and Resource Optimization
for Cloud HPC**

---

## 2. Research Question

> **How can measured workload characteristics, ML-based device prediction, dependency
> awareness, data-transfer cost awareness, and adaptive multi-objective optimization be
> combined to make better CPU/GPU scheduling decisions in a heterogeneous cloud-hosted
> HPC environment?**

---

## 3. Objectives

1. Profile Rodinia 3.1 workloads on CPU and GPU using workload-specific commands, parsers,
   and source-validated timing definitions.
2. Construct a three-layer measured dataset (raw runs → aggregated performance → ML/scheduler).
3. Engineer pre-execution features free of post-execution data leakage.
4. Train and evaluate a Random Forest classifier for device suitability prediction, including
   an ablation study on workload-identity features.
5. Implement an adaptive scheduler integrating ML prediction, dependency state,
   transfer-overhead awareness, and normalized multi-objective scoring with measurable
   dynamic weights.
6. Evaluate the scheduler against CPU-only, GPU-only, and rule-based baselines using
   identical task graphs and measurement protocols.
7. Implement a feedback loop for dataset growth and periodic model retraining.

---

## 4. Scope

**In scope:**
- Single-node heterogeneous CPU/GPU scheduling on a cloud-hosted experimental platform.
- Rodinia 3.1 as the workload source.
- Leakage-free ML device prediction from pre-execution features.
- Dependency-aware, transfer-aware, resource-aware scheduling.
- Normalized multi-objective weighted optimization with measurable dynamic weights.
- Hybrid execution for technically partitionable workloads only (one proof-of-concept,
  evaluated empirically across discrete partition ratios).
- Feedback-driven retraining.

**Out of scope:**
- Multi-node distributed HPC scheduling.
- Multi-GPU scheduling (one GPU is the controlled scheduling resource; §5.1).
- Production cloud orchestration.
- Direct hardware energy metering.
- Arbitrary unseen application generalisation.
- Actual Kaggle billing.

---

## 5. Experimental Platform

### 5.1 Environment

| Resource | Nominal specification | Runtime observation |
|---|---|---|
| Platform | Kaggle cloud notebook | Single-node environment |
| CPU | x86-64 | **4 cores** recorded in verified Kaggle runtime |
| GPU | NVIDIA Tesla T4 | 16 GB nominal VRAM; ~15 GB runtime-visible |
| GPU count | — | **Two T4 devices visible** in runtime |
| OS | Linux (Ubuntu) | |
| CUDA | Current Kaggle toolchain | |
| Benchmark suite | Rodinia 3.1 | |

**Single-GPU scheduling decision (Option A):**
The scheduler uses **one GPU device** as the controlled scheduling resource for all
primary experiments. The second visible T4 is not part of the scheduling decision.
This is explicitly stated so that `gpu_free ∈ {True, False}` remains a valid and
unambiguous binary resource indicator throughout the project.

> **Platform limitation**: Kaggle is a cloud-hosted heterogeneous experimental environment,
> not a production multi-node HPC cluster. Multi-node scheduling, real cluster resource
> management, and actual Kaggle billing are not claimed.

---

## 6. Role of Rodinia 3.1

Rodinia 3.1 is the **benchmark and workload source**, not a pre-existing ML dataset.

It provides benchmark applications, CPU/OpenMP implementations, GPU/CUDA implementations,
and standard benchmark input data.

The ML dataset is **constructed** by executing Rodinia workloads, collecting actual
measurements, and processing them through three dataset layers.

Rodinia does not provide a universal scheduler DAG; the project constructs one.

---

## 7. Overall Architecture

```
Rodinia 3.1
    │
    ▼
Workload-specific profiling
    │
    ▼
Raw execution data (Layer 1)
    │
    ▼
Timing validation + aggregation (Layer 2)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Leakage-free pre-execution        Post-execution
features (Layer 3)                telemetry
[only valid-label configs]            │
    │                                  ▼
    ▼                             Evaluation +
Random Forest device              Feedback / retraining
suitability predictor
    │
    ▼
Candidate CPU / GPU / Hybrid
    │
    ▼
Adaptive runtime scheduler
  ├── dependencies / DAG state
  ├── transfer state
  ├── CPU/GPU availability (single GPU resource)
  ├── critical-path proxy
  └── dynamic objective weights
    │
    ▼
Multi-objective scoring (frozen-reference normalisation)
    │
    ▼
Final assignment
    │
    ▼
Execution
    │
    ▼
Runtime telemetry
    │
    ├──► weight adaptation (fast loop)
    └──► dataset growth / retraining (slow loop)
```

**Key architectural separation:**
- The Random Forest operates on **static pre-execution workload features**.
- The adaptive scheduler is a separate runtime component using ML output + live resource state.

---

## 8. Workload-Specific Profiling

### 8.1 Principle

Every Rodinia workload has its own build system, executable names, arguments, input formats,
timing mechanism, and output format. No single generic profiler or parser applies universally.

### 8.2 Workload Configuration

```python
# profiler/workload_configs.py

WORKLOAD_CONFIGS = {

    "bfs": {
        "cpu_command":  "./bfs 4 ../../data/bfs/graph1MW_6.txt",
        "gpu_command":  "./bfs ../../data/bfs/graph1MW_6.txt",
        # CPU takes thread-count argument (4); GPU does not.
        "cpu_build":    "make -f Makefile openmp",
        "gpu_build":    "make -f Makefile cuda",
        "input_files":  ["graph1MW_6.txt"],
        "input_size_fn":"os.path.getsize(input_path)",   # never hard-coded (~61.2 MB)
        # Workload-specific parsers:
        "cpu_time_regex":  r"Compute time:\s*([\d.]+)",          # seconds (openmp/bfs/bfs.cpp:173)
        "gpu_time_regex":  r"GPU Kernel Time:\s*([\d.]+)\s*ms",  # ms (cuda/bfs/bfs.cu loop)
        "h2d_regex":       r"Initial Host to Device Time:\s*([\d.]+)\s*ms",
        "d2h_regex":       r"Final Device to Host Time:\s*([\d.]+)\s*ms",
        "h2d_instrumented":  True,
        "d2h_instrumented":  True,
        "timing_notes": (
            "[PENDING end-to-end validation] Source-level segment structure identified: "
            "(1) Initial H2D (~14.12 ms candidate) transfers graph arrays before traversal loop. "
            "(2) GPU Kernel/Loop phase (~3.83 ms candidate) executes 12 iterations of Kernel/Kernel2; "
            "internal 1-byte control copies (d_over) are intrinsic to loop control. "
            "(3) Final D2H (~0.77 ms candidate) copies node costs to host after loop. "
            "Candidate total GPU offload path = H2D + Loop + D2H = 14.12 + 3.83 + 0.77 ≈ 18.72 ms. "
            "END-TO-END VALIDATION REQUIRED: instrumentation must be confirmed to cover the complete "
            "offload path with no omitted setup or transfer intervals before this sum is treated as "
            "a validated total GPU execution time. "
            "Speedup_compute candidate = 35.5 ms / 3.83 ms ≈ 9.27x (pending validation). "
            "Speedup_offload candidate = 35.5 ms / 18.72 ms ≈ 1.90x (pending validation)."
        ),
        "gpu_total_path_time_status": "pending_validation",
        "status": "source_inspection_complete__end_to_end_validation_pending",
    },

    "cfd": {
        "cpu_command":  "./euler3d_cpu {input_file}",
        "gpu_command":  "./euler3d {input_file}",
        "cpu_build":    "make -f Makefile openmp",
        "gpu_build": (
            "nvcc -O2 -Xptxas -v -arch=sm_75 "
            "-I/kaggle/working/Rodinia/cuda/hybridsort "
            "euler3d.cu -o euler3d"
        ),
        "input_files":  ["fvcorr.domn.097K", "fvcorr.domn.193K"],
        "input_size_fn":"os.path.getsize(input_path)",
        # CFD parsers derived directly from source inspection:
        # GPU (cuda/cfd/euler3d.cu:589): outputs "<val> seconds per iteration"
        "gpu_time_regex": r"([\d.]+)\s+seconds per iteration",
        # CPU (openmp/cfd/euler3d_cpu.cpp:494): outputs "Compute time: <val>" (total seconds for 2000 iter)
        "cpu_time_regex": r"Compute time:\s*([\d.]+)",
        "h2d_instrumented":  False,
        "d2h_instrumented":  False,
        "timing_notes": (
            "[PENDING experimental closure] Source inspection specifies timing scope: "
            "Both CPU and GPU nominally measure 2000 Runge-Kutta iterations, "
            "excluding file I/O, domain setup, and solution dump. "
            "GPU: sdkStartTimer/sdkStopTimer around iteration loop; outputs seconds PER ITERATION. "
            "CPU: omp_get_wtime() around iteration loop; outputs TOTAL seconds for all 2000 iterations. "
            "Scaling rule (specified, not yet run-validated): "
            "gpu_total_compute_ms = float(gpu_match) * 2000.0 * 1000.0; "
            "cpu_compute_ms = float(cpu_match) * 1000.0. "
            "REQUIRED: run euler3d_cpu and euler3d, capture actual stdout, "
            "verify parsers against real output before marking gate closed."
        ),
        "gpu_total_path_time_status": "source_inspection_complete__parser_execution_pending",
        "status": "source_inspection_complete__experimental_closure_pending",
    },

    # "hotspot": { ... },   # Added after per-workload validation
    # "lud":     { ... },
    # "kmeans":  { ... },
    # "srad":    { ... },
    # "nw":      { ... },
    # "backprop":{ ... },
    # "pathfinder":    { ... },
    # "particlefilter":{ ... },
}
```

### 8.3 Input Size

Always computed at runtime:
```python
input_size_bytes = os.path.getsize(input_file_path)
input_size_mb    = input_size_bytes / (1024 * 1024)
```
`graph1MW_6.txt` is approximately **62 MB** — not 28 MB. No size is hard-coded.

### 8.4 Per-Workload Procedure

1. Inspect source code and build system.
2. Compile CPU implementation; record build status.
3. Compile GPU implementation; resolve issues as workload-specific repairs.
4. Run one test execution per device; capture complete stdout and stderr.
5. Identify and document the timing mechanism from source.
6. Create a workload-specific timing parser matching actual output.
7. Run ≥ 6 timed runs per device (after one warm-up).
8. Validate output correctness where feasible.
9. Store each run as a complete record before beginning the next run.
10. Only then admit the workload to the raw dataset.

---

## 9. Timing Measurement Methodology

### 9.1 Timing Semantics Must Be Source-Validated

For every workload, source inspection determines:
- Where timing starts and ends.
- Whether `cudaMemcpy` calls are inside the interval.
- Whether `cudaDeviceSynchronize` is inside.
- Whether initialisation, output transfer, or synchronisation is included.

**"GPU timing = kernel execution time" is never assumed.**

### 9.2 BFS Timing — Source Inspection Status & Pending End-to-End Validation

**STATUS: Source inspection COMPLETE. End-to-end validation PENDING.**

Source-level inspection of `cuda/bfs/bfs.cu` and `openmp/bfs/bfs.cpp` identifies four
sequential segments. Individual intervals have been instrumented and candidate values
recorded. The *candidate total-path sum* requires an end-to-end validation experiment
to confirm that instrumentation covers the complete offload path without omitted intervals.

1. **Segment A (Host Setup)**: File reading and host buffer allocations. Outside GPU path.
2. **Segment B (Initial H2D Transfer)**: Copies graph topology to device before traversal.
   Instrumented via CUDA events (`h2d_start_event`→`h2d_stop_event`): **~14.12 ms (candidate)**.
   Executes and synchronizes *before* the traversal loop begins.
3. **Segment C (GPU Traversal Loop Phase)**: The `do { ... } while(stop);` loop.
   Instrumented via CUDA events (`bfs_start_event`→`bfs_stop_event`): **~3.83 ms (candidate)**.
   Contains two kernel launches (`Kernel`, `Kernel2`) and two 1-byte convergence-control transfers
   (`d_over`). These 1-byte copies are intrinsic to loop control; they do not transfer graph or
   cost data and are disjoint from Segments B and D at the source level.
4. **Segment D (Final D2H Transfer)**: Copies `d_cost` to host after loop completes.
   Instrumented via CUDA events (`d2h_start_event`→`d2h_stop_event`): **~0.77 ms (candidate)**.

**Candidate Total GPU Offload Path (PENDING validation):**
```
gpu_total_path_time_ms_candidate = H2D + Loop + D2H
                                 = 14.12 + 3.83 + 0.77 = 18.72 ms
```
Segments are non-overlapping at the source level. However, final validation requires
confirming that instrumentation covers the complete intended end-to-end GPU offload
path with no omitted setup or transfer intervals. Until this experiment is performed,
the sum is treated as an instrumentation-derived candidate, NOT validated ground truth.

**Candidate Speedup Metrics (PENDING — depend on validated total path):**
1. **Algorithmic / Pure Compute Speedup (candidate)**:
   ```
   Speedup_compute_candidate = cpu_median_ms / gpu_kernel_time_ms
                             = 35.5 ms / 3.83 ms ≈ 9.27x
   ```
2. **System Offload Speedup (candidate)**:
   ```
   Speedup_offload_candidate = cpu_median_ms / gpu_total_path_time_ms_candidate
                             = 35.5 ms / 18.72 ms ≈ 1.90x
   ```
Neither figure is reported as validated until the end-to-end validation experiment closes.

### 9.3 BFS Timing — Current Measurement Status

| Measurement | Scope | Status | Value |
|---|---|---|---|
| CPU benchmark time | OpenMP traversal loop (`bfs.cpp:173`) | Source-verified | ~35.5 ms |
| GPU kernel/loop time | Traversal loop + control sync (`bfs.cu:230`) | Source-verified (candidate) | ~3.83 ms |
| Initial H2D phase | Initial graph topology transfer | Source-verified (candidate) | ~14.12 ms |
| Final D2H phase | Final result cost vector transfer | Source-verified (candidate) | ~0.77 ms |
| **GPU total offload path** | Candidate sum (H2D + Loop + D2H) | **PENDING end-to-end validation** | **~18.72 ms (candidate)** |

`gpu_total_path_time_status = "pending_validation"`. BFS is **NOT yet cleared** for Layer 3 labeling.
Layer 3 admission and `preferred_device` assignment remain blocked until the validation
experiment is completed (see Frozen Implementation Order §26, step 4).

### 9.4 CFD Timing — Source Inspection Complete, Experimental Closure PENDING

**STATUS: Source inspection COMPLETE. Parser execution against real output PENDING.**

What is included / excluded in each implementation's timing interval (source-specified):

| | CPU (`euler3d_cpu.cpp`) | GPU (`euler3d.cu`) |
|---|---|---|
| Timing mechanism | `omp_get_wtime()` (lines 472, 493) | `sdkStartTimer/sdkStopTimer` (lines 566–588) |
| What is timed | 2000 Runge-Kutta iterations | 2000 Runge-Kutta iterations |
| Iterations included | All 2000 | All 2000 |
| File I/O | **Excluded** | **Excluded** |
| Domain / geometry setup | **Excluded** | **Excluded** |
| Initial H2D memory upload | N/A | **Excluded** (before timer) |
| Final D2H solution dump | **Excluded** | **Excluded** |
| Output unit | Total seconds for all 2000 iter | Seconds **per iteration** |

**Specified parsers (to be verified against actual stdout):**
- GPU regex: `r"([\d.]+)\s+seconds per iteration"` — source: `euler3d.cu:589`
- CPU regex: `r"Compute time:\s*([\d.]+)"` — source: `euler3d_cpu.cpp:494`

**Specified scaling rules (to be confirmed after experimental run):**
```python
gpu_total_compute_ms = float(gpu_match.group(1)) * 2000.0 * 1000.0
cpu_compute_ms       = float(cpu_match.group(1)) * 1000.0
```

**Definitional comparability (SPECIFIED, not yet experimentally confirmed):**
Both CPU and GPU nominally measure the same 2000 RK iterations excluding I/O and setup.
The GPU per-iteration value must be scaled by 2000 before comparison.
```
Speedup_CFD_candidate = cpu_compute_ms / gpu_compute_ms
```
This gate is **NOT closed**. `euler3d_cpu` must be executed, actual stdout captured, and
parsers verified against real output before CFD is admitted to Layer 2 and Layer 3.

### 9.5 BFS Transfer-to-Compute Ratio

**STATUS: UNAVAILABLE — pending BFS end-to-end validation (§9.2, §26 step 4).**

The conventional formula `transfer_ratio = (T_H2D + T_D2H) / T_compute` requires
a defensible pure-compute interval. For BFS, the GPU loop phase includes synchronous
internal control copies, and H2D instrumentation may include host-side initialization;
therefore a clean pure-compute denominator has not been confirmed.

- `actual_transfer_ratio_bfs`: UNAVAILABLE. Not reported. Not fabricated.
- `estimated_transfer_ratio_bfs`: May be derived post-validation for pre-execution
  scheduling feature estimation only (must not use post-execution measured values).

Qualitatively, BFS is expected to be transfer-heavy (H2D candidate ~14.12 ms >> compute
candidate ~3.83 ms), but no numerical ratio is asserted until the validation closes.

### 9.6 Timing Field Definitions

| Field | Meaning |
|---|---|
| `benchmark_reported_time_ms` | Workload's own printed timing |
| `cpu_process_wall_time_ms` | Full CPU wall time (diagnostic) |
| `gpu_phase_time_ms` | Workload-specific GPU phase; definition documented |
| `gpu_kernel_time_ms` | Only when kernel-only timing is instrumented and verified |
| `host_to_device_time_ms` | Measured H2D; instrumentation scope documented |
| `device_to_host_time_ms` | Measured D2H; instrumentation scope documented |
| `gpu_total_path_time_ms` | Set only after workload-specific validation |

### 9.7 Repeated Runs Protocol

- 1 warm-up run (not timed, not stored as a timed record).
- ≥ 6 timed runs per workload per device.
- Primary summary: median. Also: mean, std, min, max.
- All raw run values retained in Layer 1.
- **Repeated runs are NOT independent ML samples.**

---

## 10. Three-Layer Dataset Design

### Layer 1 — Raw Execution Dataset

One row = one actual run. Failures preserved with full metadata.

| Field | Type | Notes |
|---|---|---|
| `workload` | str | |
| `input_file` | str | Actual filename |
| `input_size_bytes` | int | `os.path.getsize()` |
| `input_size_mib` | float | `input_size_bytes / (1024 * 1024)` (binary MiB throughout) |
| `device` | str | `"cpu"` or `"gpu"` |
| `run_id` | int | 1-indexed |
| `is_warmup` | bool | |
| `benchmark_reported_time_ms` | float\|null | |
| `cpu_process_wall_time_ms` | float\|null | Diagnostic |
| `gpu_phase_time_ms` | float\|null | Workload-specific |
| `gpu_kernel_time_ms` | float\|null | Only if instrumented |
| `host_to_device_time_ms` | float\|null | If instrumented |
| `device_to_host_time_ms` | float\|null | If instrumented |
| `gpu_total_path_time_ms` | float\|null | Post-validation only |
| `gpu_total_path_time_status` | str | `"pending_validation"` / `"validated"` / `"blocked"` |
| `timing_type` | str | Semantic label |
| `build_status` | str | `"success"` / `"failed"` |
| `return_code` | int | |
| `timing_parse_ok` | bool | |
| `failure_reason` | str\|null | |
| `stdout` | str | |
| `stderr` | str | |
| `gpu_device_index` | int | 0 (primary controlled GPU) |
| `gpu_type` | str | `"Tesla T4"` |
| `cpu_cores_available` | int | 4 (verified runtime) |

Failures: stored with failure metadata. Never receive zero or imputed timing values.
Workload failure counts reported separately.

### Layer 2 — Aggregated Performance Dataset

One row = one workload/input configuration. Built from **valid Layer 1 rows only**.

| Field | Notes |
|---|---|
| `workload`, `input_file`, `input_size_mib` | `input_size_mib = input_size_bytes / (1024 * 1024)` |
| `cpu_median_ms`, `cpu_mean_ms`, `cpu_std_ms`, `cpu_n_valid` | |
| `cpu_timing_type` | Documented semantic |
| `gpu_median_ms`, `gpu_mean_ms`, `gpu_std_ms`, `gpu_n_valid` | |
| `gpu_timing_type` | Documented semantic |
| `h2d_median_ms`, `d2h_median_ms` | null if not instrumented |
| `gpu_total_path_ms` | null until validation experiment confirmed |
| `gpu_total_path_time_status` | |
| `comparable_timing_ok` | True only when CPU/GPU timings are definitionally compatible |
| `preferred_device` | `"cpu"` / `"gpu"` / null |
| `cpu_failure_count`, `gpu_failure_count` | |
| `profiling_notes` | Source-level timing summary |

### Layer 3 — ML / Scheduler Dataset

**Construction rule (explicit):**

> Layer 3 is constructed **only** from workload/input configurations where:
> `comparable_timing_ok = True` AND `gpu_total_path_time_status = "validated"` AND
> `preferred_device ∈ {"cpu", "gpu"}` (not null).
>
> BFS and CFD rows are excluded until their gating conditions are satisfied.
> It is an implementation error to assign BFS or CFD a `preferred_device` label
> from incomplete timing information.

One row = one pre-execution scheduling instance. Contains only pre-execution features
plus the validated ground-truth label. Post-execution measurements are strictly excluded
as classifier inputs.

---

## 11. ML Data Leakage Prevention

The classifier predicts device assignment before either device is executed.
Post-execution quantities must not be used as classifier inputs.

**Strictly excluded from classifier inputs:**
- Any measured CPU or GPU execution time.
- Measured H2D or D2H transfer time.
- Total measured transfer time.
- Measured CPU/GPU speedup or ratio.
- Any other post-run timing or energy value.

These are used for: evaluation, feedback, ground-truth labels, retraining — not prediction.

---

## 12. Feature Engineering

### 12.1 Two-Stage Architecture

```
Pre-execution workload features
    → ML device suitability predictor (Random Forest)
    → candidate devices: CPU / GPU / Hybrid

Runtime scheduler (separate component):
    - ready_task_count (scheduler-managed queue)
    - dependency / DAG state
    - transfer state
    - cpu_slots_free
    - gpu_free (single GPU resource, §5.1)
    - critical-path position
    - dynamic objective weights
    → final assignment
```

### 12.2 Pre-Execution Workload Features (ML Classifier Inputs)

| Feature | Derivation |
|---|---|
| `input_size_mib` | `os.path.getsize(input_path) / (1024 * 1024)` — binary MiB; 1 MiB = 1024² bytes |
| `workload_type` | Categorical label from source analysis (see ablation, §12.6) |
| `workload_domain` | e.g., `"graph"`, `"fluid_dynamics"` (see ablation, §12.6) |
| `estimated_parallelism` | Rubric score 1–5 (§12.4) |
| `estimated_memory_boundness` | Rubric score 1–5 (§12.4) |
| `estimated_compute_intensity` | Rubric score 1–5 (§12.4) |
| `has_irregular_memory` | Boolean (§12.4) |
| `has_strong_serial_dependency` | Boolean (§12.4) |
| `estimated_transfer_size_mb` | From static buffer analysis (§12.5) |
| `transfer_estimation_method` | Method label (§12.5) |
| `cpu_cores_available` | 4 (verified Kaggle runtime) |
| `gpu_type_encoded` | Encoded GPU identifier |
| `gpu_memory_gb` | 16 (Tesla T4 nominal) |

### 12.3 Runtime Scheduler State (Scheduler Inputs — NOT Classifier Inputs)

| Feature | Source |
|---|---|
| `cpu_slots_free` | Scheduler-managed count |
| `gpu_free` | Boolean: single controlled GPU currently unoccupied |
| `ready_task_count` | Scheduler-managed ready queue length |
| `dependency_count` | DAG: predecessor count for this task |
| `critical_path_flag` | Boolean: on longest estimated remaining path |

**Naming consistency**: `ready_task_count` is used throughout the plan
(not "queue_length" or "queue_pressure_proxy"). Dynamic weight triggers also use
`ready_task_count` (§18.1).

### 12.4 Complete Expert-Feature Rubrics (Rubric v1.0)

The rubric version is stored with the dataset. Scores are assigned from concrete
source-based criteria before model training. The feature-generation code is frozen
before Layer 3 is populated (see implementation order, §26, step 8a).

> **Scope of all rubric thresholds:** Arithmetic-intensity cut-points and rubric score
> boundaries are **project-defined heuristic thresholds** for this study. They are not
> presented as universal scientific definitions of memory-bound or compute-bound workloads.
> Their use makes feature engineering reproducible and auditable within this project.

---

**`estimated_parallelism` (1–5):**

| Score | Criterion |
|---|---|
| 5 | Embarrassingly parallel; all work units independent (e.g., dense matrix-matrix multiply) |
| 4 | Mostly parallel; small global reduction or barrier synchronisation only |
| 3 | Alternating parallel and serial phases; inter-step dependencies |
| 2 | Significant serial sections; most iterations depend on previous results |
| 1 | Inherently sequential; near-zero exploitable parallelism |

---

**`estimated_memory_boundness` (1–5):**

> **Naming note**: previously called `estimated_memory_intensity`. Renamed to
> `estimated_memory_boundness` because the rubric is based on an
> **arithmetic-intensity approximation** (FLOP/byte ratio derived from source-level
> analysis), not a directly measured memory-bandwidth figure. When a reviewer asks
> "how did you know BFS has 0.4 FLOP/byte?", the answer is: source-level inspection
> of loop structure, per-element operations, and data sizes — documented per workload.

| Score | Criterion (approximate arithmetic intensity) |
|---|---|
| 5 | Memory-bound; AI < 0.5 FLOP/byte; pointer-chasing or purely streaming access |
| 4 | Memory-heavy; AI 0.5–2 FLOP/byte; few operations per loaded element |
| 3 | Balanced; AI 2–10 FLOP/byte |
| 2 | Compute-leaning; AI 10–50 FLOP/byte |
| 1 | Compute-bound; AI > 50 FLOP/byte; rarely bottlenecked by bandwidth |

Source documentation per workload: loop count, per-element FLOP count, buffer sizes,
and access pattern class (sequential / strided / irregular).

---

**`estimated_compute_intensity` (1–5):**

| Score | Criterion |
|---|---|
| 5 | Very high arithmetic work per input byte (e.g., dense LU, iterative solvers) |
| 4 | High; significant floating-point per element |
| 3 | Moderate; mixed computation and data movement |
| 2 | Low; simple arithmetic; throughput dominated by data movement |
| 1 | Minimal arithmetic; primarily data transfer or indexing |

---

**`has_irregular_memory` (Boolean):**
`True` if the workload uses pointer-chasing, sparse structures, graph adjacency lists,
or access patterns that defeat hardware prefetching (e.g., BFS). `False` otherwise.

**`has_strong_serial_dependency` (Boolean):**
`True` if correctness of iteration N depends on completed output of iteration N−1,
making fine-grained parallel decomposition unsafe. `False` if iterations are independent.

---

### 12.5 `estimated_transfer_size_mb` — Operational Definition

Estimated from **static input structure and source-level buffer analysis**.

> **Unit standard:** All size quantities use binary MiB (1 MiB = 1024² = 1 048 576 bytes).
> `input_size_mib` and `estimated_transfer_size_mib` use this same unit throughout.
> Transfer size and input size are **separate concepts**:
> - `input_size_mib` = size of the workload's input file(s) on disk.
> - `estimated_transfer_size_mib` = estimated bytes moved between host and GPU device memory.
> These differ whenever GPU buffers are a subset or superset of the raw input file.

```python
estimated_transfer_size_bytes = sum(
    element_count_i × element_size_bytes_i
    for each buffer_i crossing the host/device boundary
)
estimated_transfer_size_mib = estimated_transfer_size_bytes / (1024 * 1024)
```

Buffer list is derived from: GPU allocation calls in the source
(`cudaMalloc`, `cudaMemcpy` destinations) and input array dimensions and types.

If a workload's transfer size cannot be reliably characterised: stored as `null`.

**`transfer_estimation_method`** field (required alongside `estimated_transfer_size_mib`):

| Value | Meaning |
|---|---|
| `"source_static_analysis"` | Buffer sizes read from source code buffer declarations |
| `"input_structure_analysis"` | Derived from input file element counts and layout |
| `"profiling_lookup"` | Taken from measured H2D + D2H (post-execution; used for calibration only) |
| `"unavailable"` | Cannot be reliably estimated; field is null |

Measured H2D/D2H times are kept exclusively in post-execution telemetry and are never
used as the source of `estimated_transfer_size_mb`.

### 12.6 Workload-Identity Ablation Study

`workload_type` and `workload_domain` are categorical features encoding workload identity.
Under LOWO validation, a model trained on BFS/CFD/LUD/Hotspot and tested on K-means may
treat K-means's unseen category as out-of-distribution, effectively making the workload
label a memorisation proxy rather than a generalisation signal.

**Ablation procedure:**

| Model | Features included |
|---|---|
| **Model A** | All pre-execution features + `workload_type` + `workload_domain` |
| **Model B** | All pre-execution features excluding `workload_type` and `workload_domain` |

Both models are trained and evaluated with LOWO.

**Reported separately:** LOWO accuracy, precision, recall, F1 for Model A and Model B.

This analysis demonstrates **reduced explicit workload-identity dependence** in Model B
relative to Model A. It does NOT make Model B fully workload-independent: workload-specific
characteristics remain indirectly encoded through expert features such as
`has_irregular_memory`, `estimated_parallelism`, and `estimated_compute_intensity`.
The ablation isolates the contribution of the categorical identity label, not all
workload-specific signal.

### 12.7 Random Forest Encoding

- Categorical features are **numerically or one-hot encoded**.
- Encoding pipeline is fit on training data only.
- No feature scaling is required for Random Forest.
- Encoding method and fitted values are documented per feature.

### 12.8 Post-Execution Telemetry (Not Classifier Features)

| Field | Description |
|---|---|
| `actual_cpu_time_ms` | Measured CPU execution time |
| `actual_gpu_time_ms` | Measured GPU phase/execution time |
| `actual_h2d_ms` | Measured H2D (if instrumented) |
| `actual_d2h_ms` | Measured D2H (if instrumented) |
| `actual_gpu_total_path_ms` | Validated GPU total path (post-validation only) |
| `actual_transfer_ratio` | Post-execution ratio for evaluation only |
| `energy_proxy_cpu_j` | Assumed power × execution time |
| `energy_proxy_gpu_j` | Assumed power × execution time |
| `cost_modelled_cpu` | Modelled equivalent cost (§22) |
| `cost_modelled_gpu` | Modelled equivalent cost (§22) |
| `cost_modelled_hybrid` | Modelled equivalent cost (§22) |
| `waiting_time_ms` | Time in ready queue |
| `scheduler_decision_time_ms` | Scheduler overhead (§25, required metric) |
| `empirical_preferred_device` | Device with lower measured time (post-execution label) |
| `ml_prediction` | Classifier output (pre-execution) |
| `ml_matches_empirical_preference` | Whether ML matched empirical label |
| `scheduler_assignment` | Final scheduler decision |
| `scheduler_override` | Whether scheduler differed from ML prediction |
| `override_reason` | Reason for scheduler override |

---

## 13. Ground-Truth Label Construction

### 13.1 Definition

```
preferred_device ∈ { "cpu", "gpu", null }

If comparable_timing_ok = True
   AND gpu_total_path_time_status = "validated":
    If cpu_median_ms < gpu_total_path_ms:
        preferred_device = "cpu"
    Else:
        preferred_device = "gpu"
Else:
    preferred_device = null   # label withheld until gating conditions met
```

CPU wall time is **never** compared against GPU kernel-only time.

> **Precise meaning of `preferred_device`:** This label means *empirically preferred under
> the specific measured hardware, software configuration, input file, and timing definition
> used in this experiment*. It does NOT mean the device is intrinsically superior for this
> workload class, universally optimal for this benchmark, or correctly labelled under a
> different hardware/software configuration.

**Distinct concepts that must not be conflated:**

| Field | Meaning |
|---|---|
| `empirical_preferred_device` | Device with lower measured time under this experiment's conditions |
| `ml_prediction` | Random Forest output before scheduling decision |
| `scheduler_assignment` | Final device selected by scheduler (may differ from ML prediction) |
| `ml_matches_empirical_preference` | Whether `ml_prediction == empirical_preferred_device` |
| `scheduler_override` | Whether `scheduler_assignment != ml_prediction` |
| `override_reason` | Why scheduler differed (resource pressure, dependency, transfer cost, etc.) |

Scheduler overrides are not ML errors. The scheduler may legitimately differ from the ML
prediction due to GPU availability, dependencies, transfer overhead, cost, energy,
dynamic weights, or hybrid execution constraints.

### 13.2 Current Label Status

| Workload | `empirical_preferred_device` | Reason |
|---|---|---|
| BFS | **null / blocked** | `gpu_total_path_time_status = "pending_validation"`; end-to-end validation experiment not yet performed |
| CFD | **null / blocked** | `status = "source_inspection_complete__experimental_closure_pending"`; CPU parser not yet run against real stdout |
| All others | **null / not started** | Not yet profiled |

**It is an implementation error to assign BFS = "GPU" or CFD = "GPU" before the gating
conditions are met.** Layer 3 is not populated with these workloads until they are cleared.

---

## 14. ML Model

### 14.1 Primary Task

Classify: `preferred_device ∈ {cpu, gpu}` using pre-execution workload features only.

**Primary model**: Random Forest classifier.

**Optional comparison**: XGBoost — added only after primary model is stable.

### 14.2 Execution-Time Estimates for the Scheduler (3-Tier Lookup + Fallback Policy)

The ML classifier outputs device suitability class; it does not directly output numerical
execution times. The multi-objective objective function (§18.1) requires numerical estimates.

**3-tier fallback policy:**

```
Tier 1: Exact Layer 2 Lookup
   │ (workload, input_file) match found in Layer 2?
   ├──► YES: Return T_CPU_est = cpu_median_ms
   │         Return T_GPU_est = gpu_total_path_ms   (post-validation only)
   │         Set estimate_source = "exact_lookup"
   │
   └──► NO: Tier 2: Empirical Size-Based Scaling
             │ Same workload profiled at a different input size in Layer 2?
             ├──► YES: Apply linear scaling (explicit baseline assumption):
             │         T_est = T_baseline × (input_size_mib / baseline_size_mib)
             │         HPC workloads do not necessarily scale linearly with input size.
             │         Accuracy must be evaluated, not assumed.
             │         Set estimate_source = "size_scaled_estimate"
             │
             └──► NO: Tier 3: Unprofiled Workload Fallback (Operational Safety Mechanism)
                       │ Completely unseen workload
                       ├──► Extract pre-execution rubric features from static analysis
                       ├──► Call ML classifier for device preference
                       ├──► T_CPU_Tier3 = median CPU time of valid Layer-2 records
                       │    in the same domain; if domain unseen: T_CPU_global_median
                       ├──► T_GPU_Tier3 = median GPU time (gpu_total_path_ms) of valid
                       │    Layer-2 records in the same domain; if domain unseen: T_GPU_global_median
                       ├──► CPU and GPU medians are computed SEPARATELY from valid runs only
                       ├──► Safeguard: If ML confidence < 0.70, assign CPU-safe fallback
                       └──► Set estimate_source = "domain_median_fallback" (or "global_fallback")
```

> **Tier-3 scope note:** The Tier-3 fallback is an **operational safety mechanism**.
> It is NOT evidence of validated ML generalization to unseen workloads. Its decisions
> are evaluated separately from LOWO classifier results. Arbitrary unseen application
> generalization is explicitly out of scope (§4).

**Metadata Tracking:**
For every scheduled task, the scheduler logs:
- `execution_time_estimate_source`: `"exact_lookup"`, `"size_scaled_estimate"`, `"domain_median_fallback"`, or `"global_fallback"`.
- `fallback_applied`: `True` / `False`.
- `fallback_reason`: null or diagnostic string.

### 14.3 Validation Strategy

**Preferred**: Leave-One-Workload-Out (LOWO).
- Train on all workloads except one; test on held-out; rotate.
- Group all repetitions of the same workload/input **in the same fold**.
- Do not split repeated runs of the same configuration across train and test.

If too small for stable LOWO: report instability explicitly.

Report: per-workload accuracy, precision, recall, F1, confusion matrix, macro-average,
and comparison against a majority-class baseline.

### 14.4 Dataset Size

The ML sample count = distinct valid workload/input configurations after aggregation.
Stated only after profiling is complete. No promise made before the profiling matrix
is finished.

---

## 15. Adaptive Scheduler

### 15.1 Decision Sequence

```
Step  1. Receive workload/task.
Step  2. Extract pre-execution workload features (§12.2).
Step  3. ML classifier predicts device suitability (Model A or B).
           Lookup or regression provides execution-time estimate (§14.2).
Step  4. Generate candidate assignments: CPU / GPU / Hybrid.
Step  5. Check dependency readiness (all predecessors complete?).
Step  6. Check cpu_slots_free.
Step  7. Check gpu_free (single controlled GPU, §5.1).
Step  8. Estimate transfer overhead (static model or profiling lookup).
Step  9. Evaluate current dynamic objective weights (§18).
Step 10. Compute normalised multi-objective score for each candidate (§17).
Step 11. Select execution mode with lowest objective score.
Step 12. Execute.
Step 13. Record scheduler_decision_time_ms (required metric, §25).
Step 14. Measure actual performance (Layer 1 telemetry).
Step 15. Compare actual vs predicted device and time.
Step 16. Record prediction_correct, scheduler_override, override_reason.
Step 17. Store telemetry record.
Step 18. Trigger weight adaptation (§18.1).
Step 19. Accumulate labeled samples; trigger retraining at stated threshold (§20.2).
```

### 15.2 Scheduler Evaluation Fields

| Field | Meaning |
|---|---|
| `empirical_preferred_device` | Device with lower measured time under this experiment's conditions |
| `ml_prediction` | What the Random Forest classifier predicted (pre-execution) |
| `scheduler_assignment` | Final device selected by the adaptive scheduler |
| `ml_matches_empirical_preference` | Whether `ml_prediction == empirical_preferred_device` (ML evaluation) |
| `scheduler_override` | Whether `scheduler_assignment != ml_prediction` |
| `override_reason` | Why scheduler differed from ML (e.g., `"gpu_free=False"`, `"transfer_ratio_high"`) |

> **Important:** Scheduler overrides are NOT automatically ML errors. The scheduler
> may legitimately differ from the ML prediction due to GPU availability, dependencies,
> transfer overhead, cost, energy, dynamic objective weights, or hybrid execution
> constraints. ML accuracy is evaluated against `empirical_preferred_device`,
> not against `scheduler_assignment`.

---

## 16. Dependency-Aware Scheduling

### 16.1 Project-Created DAG — Task Semantics

#### Definitions (authoritative, frozen)

| Concept | Definition |
|---|---|
| **Rodinia benchmark** | A Rodinia 3.1 application (e.g., BFS, CFD). Source of executable programs and input data. Does NOT provide the project DAG. |
| **Scheduling task** | One schedulable execution unit managed by the project scheduler. In the primary implementation, one task = one Rodinia workload instance run on one device with one input file. |
| **DAG node** | Represents one scheduling task. Each node carries: `task_id`, `workload`, `input_file`, `device_candidate`, `estimated_duration`, `dependency_count`. |
| **DAG edge A → B** | B cannot begin until A has completed execution AND its required output/data is available to B. |
| **dependency_count** | Number of direct predecessor nodes of a task in the project DAG. This is a scheduler/project-context feature, NOT an intrinsic property of any Rodinia benchmark. |
| **Ready task** | A task whose `dependency_count` predecessors have all reached `status = "completed"`. Only ready tasks may be dispatched. |
| **Critical-path proxy** | The longest dependency-constrained path through the DAG under the candidate execution-time estimates from §14.2. |

> **Rodinia does NOT provide the project scheduling DAG.** The DAG is constructed by
> the project for controlled scheduling experiments. Task-level decomposition is a
> project design choice, not a property of the Rodinia benchmark suite.

#### Concrete Example (Minimal Reference DAG)

```
Task A (BFS,  graph1MW_6.txt)  ─┬─► Task D (LUD, matrix512.dat)
                                │
Task B (CFD,  fvcorr.097K)    ─┘
Task C (CFD,  fvcorr.193K)    ───► [independent; no successor]
```

Operational meaning:
- Task A and Task B must both complete before Task D can begin.
- Task C has no predecessors and no successors in this example; it runs independently.
- `dependency_count(A) = 0`, `dependency_count(B) = 0`, `dependency_count(C) = 0`, `dependency_count(D) = 2`.
- When A and B are both `status="completed"`, D becomes a ready task.
- The critical-path proxy from the entry nodes to D = `max(estimated_duration(A), estimated_duration(B)) + estimated_duration(D)`.

All DAG nodes use the same task semantics: one task = one workload instance. Phases within
a single Rodinia benchmark execution are NOT split into multiple DAG nodes unless the
project explicitly defines a multi-phase task decomposition (PLANNED, not current scope).

### 16.2 Critical-Path Analysis

```
longest_path(leaf_node)  = estimated_duration(leaf_node)      [from §14.2 3-tier lookup]
longest_path(node)       = estimated_duration(node) + max(longest_path(s) for s in successors(node))
```

`estimated_duration(node)` uses the task's pre-execution time estimate from the 3-tier
lookup (§14.2): Tier-1 Layer-2 lookup preferred; Tier-2 size-scaled estimate if Tier-1
unavailable; Tier-3 domain/global median if neither applies.

This is a **critical-path proxy** (dependency-based structural lower-bound estimate),
not a complete makespan model. It does not account for resource contention, CPU/GPU
assignment, transfer time, scheduling overhead, or concurrent execution.
Critical-path tasks are not automatically assigned to the GPU.

---

## 17. Transfer-Aware Scheduling

### 17.1 Static vs Measured Transfer

| Context | Data used | Label |
|---|---|---|
| Pre-execution scheduling decision | `estimated_transfer_size_mib` from static analysis (§12.5) | "estimated" |
| Post-execution feedback / evaluation | Measured `actual_h2d_ms`, `actual_d2h_ms` | "measured" |

### 17.2 Pre-Execution Transfer-Time Estimation — Operational Formula

All quantities below use **only information available before the current task executes**.
Measured H2D/D2H from the current task must NOT feed back into its own pre-execution decision.

```python
# Step 1: Estimate transfer bytes (from §12.5 method)
estimated_bytes_H2D = estimated_H2D_fraction × estimated_transfer_size_bytes
estimated_bytes_D2H = estimated_transfer_size_bytes - estimated_bytes_H2D
# estimated_H2D_fraction: workload-specific ratio from source analysis (default 0.8 if unknown)

# Step 2: Apply bandwidth model calibrated from historical profiling
# (calibration uses previous profiling runs, NOT the current task's execution)
estimated_T_H2D_ms = (estimated_bytes_H2D / calibrated_H2D_bandwidth_bytes_per_ms)
                     + H2D_fixed_overhead_ms
estimated_T_D2H_ms = (estimated_bytes_D2H / calibrated_D2H_bandwidth_bytes_per_ms)
                     + D2H_fixed_overhead_ms

# Step 3: Combine
estimated_T_transfer_ms = estimated_T_H2D_ms + estimated_T_D2H_ms
```

**Calibrated bandwidth and overhead sources** (in priority order):
1. `profiling_lookup`: historical `h2d_median_ms` / `actual_h2d_ms` from Layer-2 or
   prior profiling runs for the same or comparable workload/configuration.
   **Not the current task's own post-execution measurement.**
2. `source_static_analysis` / `input_structure_analysis`: buffer sizes from source;
   bandwidth estimated from T4 PCIe 3.0 theoretical peak (12 GB/s H2D, 12 GB/s D2H)
   with a conservative utilization factor (e.g., 0.6) as an upper-bound fallback.
3. If neither is possible: `estimated_T_H2D_ms = null`, `estimated_T_D2H_ms = null`,
   transfer objective excluded from scoring for this candidate.

**`estimated_T_compute_ms` definition:**
```
For CPU candidate:  estimated_T_compute_ms = T_CPU_est  (from §14.2 3-tier lookup)
For GPU candidate:  estimated_T_compute_ms = T_GPU_est  (GPU compute/phase time; excludes H2D/D2H)
Note: GPU total offload path = estimated_T_compute_ms + estimated_T_H2D_ms + estimated_T_D2H_ms
      ONLY when those components are non-overlapping per the validated timing definition.
      Do NOT add H2D/D2H again if T_GPU_est already includes them.
```

### 17.3 Transfer-to-Compute Ratio

```
estimated_transfer_ratio  = (estimated_T_H2D_ms + estimated_T_D2H_ms) / estimated_T_compute_ms
    — pre-execution; from bandwidth model; used for scheduling decisions and weight triggers
    — must NOT use current-task post-execution data

actual_transfer_ratio     = (actual_h2d_ms + actual_d2h_ms) / actual_gpu_phase_time_ms
    — post-execution only; for evaluation and feedback
    — must never feed back into pre-execution scheduling decision
```

Numerator and denominator must not overlap.
**BFS `actual_transfer_ratio`**: UNAVAILABLE (§9.5). Pure-compute denominator not isolated.
**CFD `actual_transfer_ratio`**: UNAVAILABLE pending experimental closure (§9.4).

---

## 18. Multi-Objective Optimization

### 18.1 Objective Function and Candidate-Level Definitions

The objective is minimized over all feasible candidate assignments:

```
J(c) = w1 × norm(T_candidate(c))
     + w2 × norm(T_transfer(c))
     + w3 × norm(E_candidate(c))
     + w4 × norm(R_candidate(c))
     + w5 × norm(C_candidate(c))

subject to: Σ wᵢ = 1,  all wᵢ ≥ 0
```

**Candidate-level quantity definitions:**

| Quantity | CPU candidate | GPU candidate | Hybrid candidate |
|---|---|---|---|
| `T_candidate` (completion time) | `T_CPU_est` (from §14.2) | `T_GPU_total_est = T_GPU_compute_est + estimated_T_H2D_ms + estimated_T_D2H_ms` — ONLY if compute estimate excludes H2D/D2H | `T_hybrid_wall` = max or sum depending on concurrency (see §19.1) |
| `T_transfer` (transfer overhead) | 0 (CPU has no H2D/D2H) | `estimated_T_H2D_ms + estimated_T_D2H_ms` (§17.2) | Transfer/sync overhead for the partition (§19.1) |
| `E_candidate` (energy proxy) | `P_cpu × T_CPU_est_s` | `P_gpu × T_GPU_total_est_s` | `P_cpu × T_CPU_active_s + P_gpu × T_GPU_active_s` |
| `C_candidate` (modelled cost) | `T_CPU_hours × R_CPU_VM` (§22) | `T_GPU_hours × (R_CPU_VM + R_T4)` (§22) | `T_CPU_active_hours × R_CPU_VM + T_GPU_active_hours × R_T4` (§22) |
| `R_candidate` (resource pressure) | `cpu_pressure` (§18.3) | 0.0 (GPU pressure = 0 in primary experiment; §18.3) | `cpu_pressure` for CPU portion |

> **Double-counting guard:** If `T_GPU_compute_est` already includes H2D and D2H
> (i.e., it is the validated `gpu_total_path_ms` from Layer 2), then `T_transfer` for that
> candidate must be set to 0 to avoid adding H2D/D2H twice. The transfer objective is
> meaningful only when GPU compute time and transfer time are separately estimated.
> Document per-workload which convention is in use.

If an objective is unavailable: set `wᵢ = 0`, renormalise remaining active weights,
record which objectives were active for the decision.

### 18.2 Frozen-Reference Normalisation and Out-of-Range Handling

Normalisation parameters are fit from training/profiling data and **frozen before evaluation begins**.
Test data must never be used to recompute `z_min` or `z_max`.

```python
# Training time: fit normalization range
z_min = min(objective_values_in_training_set)
z_max = max(objective_values_in_training_set)

# Scoring time: normalize and clip
if z_max != z_min:
    z_norm = (z - z_min) / (z_max - z_min)
else:
    z_norm = 0.0   # degenerate range; objective contributes zero

# Out-of-range clipping: applied after normalization
z_norm_clipped = min(1.0, max(0.0, z_norm))
# Values below training minimum map to 0; values above training maximum map to 1.
# Clipping is a normalization policy; it does not change the raw measured/estimated value.
```

Candidate-set-dependent ranges are not used. The degenerate zero-range case is handled
deterministically: objective is zeroed and its weight is redistributed to active objectives.

### 18.3 Resource Penalty and Hard Constraints

**Hard constraints (candidate eliminated before scoring — feasibility check):**
- GPU candidate: eliminated if `gpu_free == False`.
- Any candidate: eliminated if any required predecessor task is not `status="completed"`.
- Any candidate: eliminated if the required execution mode is unsupported for this workload.

**Soft pressure (applied ONLY to feasible candidates remaining after hard-constraint filtering):**
```python
cpu_pressure = cpu_slots_in_use / total_cpu_slots   # ∈ [0, 1]; continuous metric
```

**GPU soft pressure — primary experiment:**
```
gpu_pressure = 0.0  (fixed; primary controlled experiment uses one binary-state GPU)
```
In the primary controlled experiment, the GPU is represented by the binary indicator
`gpu_free ∈ {0, 1}`. There is no continuous GPU utilization metric in the primary
experiment. GPU unavailability (`gpu_free == 0`) is a hard constraint, not a soft penalty.

> **No contradiction:** The GPU pressure trigger row in §20.1 (which activates the hard
> constraint) is consistent with this. The `w4` weight (resource penalty) therefore
> affects CPU pressure scoring only in the primary experiment. If a future extension
> adds measurable GPU utilization, it will be labelled **OPTIONAL / FUTURE** and
> evaluated separately.

Labelled "scheduler-state pressure proxy" — not a production GPU queue.

### 18.4 Primary vs Advanced Optimization

**Primary**: weighted-sum scalarisation.
**Optional**: NSGA-II after weighted-sum is stable. Alternative approach, not combined.

---

## 19. Hybrid CPU/GPU Execution

### 19.1 Empirical Partition-Ratio Model

`f(r)` (fraction of work executed at partition ratio `r`) is **not assumed to be linear**.
For the proof-of-concept partitionable workload:

**Empirical characterisation procedure:**

1. Run the workload at discrete partition ratios: `r ∈ {0.0, 0.25, 0.50, 0.75, 1.0}`.
2. Measure for each `r`:
   - CPU portion execution time: `T_CPU(r)`
   - GPU portion execution time: `T_GPU(r)`
   - Synchronisation overhead
   - Transfer overhead for the partition
3. Construct empirical tables:
   - `r → T_CPU_portion(r)`
   - `r → T_GPU_portion(r)`
   - `r → T_transfer(r)`
4. Compute hybrid completion time for each `r`:
   ```
   T_hybrid(r) = max(T_CPU_portion(r), T_GPU_portion(r))
                 + T_sync(r)
                 + T_transfer(r)
   ```
   **Concurrency assumption (MUST be stated explicitly):** This formula assumes the CPU
   and GPU portions execute *concurrently* after partition and setup. If the implementation
   executes the portions *serially*, the formula becomes:
   ```
   T_hybrid_serial(r) = T_CPU_portion(r) + T_GPU_portion(r) + T_sync(r) + T_transfer(r)
   ```
   The implementation must record which mode was used; results must not be mixed.
   Hybrid results are not reported as validated until the experiment is actually run.
5. Select `r*` that minimises `T_hybrid(r)` over the measured points.
6. Validate `T_hybrid(r*)` against CPU-only and GPU-only baselines.

This is substantially stronger than assuming linear scaling from a single throughput ratio.

### 19.2 Scope Constraint

- One partitionable workload is the proof-of-concept. Not forced onto BFS or CFD.
- Non-partitionable workloads: CPU-only or GPU-only only.

---

## 20. Dynamic Objective Weighting

### 20.1 Rule-Based Weight Adaptation — Full Specification

**Initial weights (frozen before evaluation):**

```
Initial weights:
    w1_init (time)     = 0.40
    w2_init (transfer) = 0.20
    w3_init (energy)   = 0.15
    w4_init (resource) = 0.15
    w5_init (cost)     = 0.10

Bounds:
    w_min = 0.05   (no objective fully excluded)
    w_max = 0.60   (no single objective dominates completely)

Step sizes (δ):
    δ_r = 0.10  (GPU pressure trigger)
    δ_t = 0.10  (transfer-ratio trigger)
    δ_e = 0.10  (energy trigger)
    δ_q = 0.10  (queue pressure trigger)
```

**Trigger conditions:**

| Trigger | Condition | Action | Frozen Parameter |
|---|---|---|---|
| GPU pressure | GPU candidate required and `gpu_free == False` — hard infeasibility; GPU candidate eliminated before scoring | GPU candidate eliminated; no weight change | — |
| Transfer-heavy | `estimated_transfer_ratio > θ_t` (pre-execution estimate only) | `w2 += δ_t` | `θ_t = 0.30`, `δ_t = 0.10` |
| Energy priority | `energy_mode == "energy_priority"` (explicitly set flag) | `w3 += δ_e` | `δ_e = 0.10` |
| Queue pressure | `ready_task_count > θ_q` | `w1 += δ_q` | `θ_q = 3`, `δ_q = 0.10` |

**`energy_mode` definition:**
```python
energy_mode ∈ {"normal", "energy_priority"}
# Set by an external flag or configuration parameter before scheduling begins.
# Default: "normal". Set to "energy_priority" to activate energy weight boost.
```

Thresholds `θ_t = 0.30` and `θ_q = 3` are frozen before test evaluation begins.
Multiple triggers may fire simultaneously; all increments are applied before renormalisation.

**After each update:**
```python
for i in range(len(w)):
    w[i] = max(w_min, min(w_max, w[i]))
total = sum(w)
w = [wi / total for wi in w]
```

Weights are not tuned on test workloads. Initial values and thresholds are stated
and frozen before the evaluation phase.

**No deadline proximity term** — deadlines are not part of this project's task model.

### 20.2 Bayesian Online Learning

Not the primary mechanism. Optional future extension only.

---

## 21. Energy Model

```
energy_proxy_j = assumed_average_power_w × execution_time_s
```

Labelled **"simplified energy proxy based on assumed average device power"** throughout.
These are sensitivity model parameters, NOT measured actual device power.

| Device | Assumed average power | Clarification |
|---|---|---|
| CPU | 65 W (central) | Assumed average device power for sensitivity model; actual CPU package power varies with workload and utilisation |
| NVIDIA Tesla T4 | 70 W (central) | Assumed average device power for sensitivity model; not measured TDP |

> If `pynvml` or `nvidia-smi` power sampling is used, measured values are reported
> separately and labeled as **measured GPU power**, distinct from assumed sensitivity values.

**Sensitivity analysis (3×3 Grid = 9 Evaluated Configurations):**
Because power is assumed rather than directly measured with hardware power meters,
energy rankings could theoretically depend on assumed power figures.
A sensitivity analysis systematically tests conclusions across a 9-point parameter grid:

```
P_cpu ∈ { 50 W, 65 W, 80 W }
P_gpu ∈ { 60 W, 70 W, 80 W }
Combinations: (50,60), (50,70), (50,80), (65,60), (65,70), (65,80), (80,60), (80,70), (80,80)
```

For every workload and scheduling decision, the analysis records whether the relative
energy ranking between CPU and GPU assignments changes under any of the 9 configurations.
If relative rankings are invariant across the entire grid, the energy proxy conclusion
is verified to be robust against assumed power variance.

The execution interval used for the energy calculation is specified per workload
(benchmark phase time vs end-to-end path time).

If NVIDIA power sampling via `pynvml` or `nvidia-smi` is used: sampling interval (100 ms),
aggregation rule (mean power over active task window), and limitations are documented.

---

## 22. Cloud Cost Model

All cost values are **modelled equivalent cloud resource costs**, NOT actual Kaggle billing.

**Resource-active-time cost formulas:**
```
C_CPU    = T_CPU_active_hours × R_CPU_VM

C_GPU    = T_GPU_active_hours × R_T4
           + T_CPU_host_active_hours × R_CPU_VM
         = T_GPU_active_hours × (R_CPU_VM + R_T4)   [if CPU host allocated for full GPU task duration]

C_hybrid = T_CPU_active_hours × R_CPU_VM
           + T_GPU_active_hours × R_T4
```

**Primary experiment assumption:** CPU host remains allocated for the entire GPU task
duration (VM not released during GPU work). Therefore `T_CPU_host_active = T_GPU_active`
and the simplified combined rate applies:
```
C_GPU_simplified = T_GPU_active_hours × $0.5400
```

**Hybrid cost:** If CPU and GPU execute concurrently for the full hybrid wall-clock time:
```
T_CPU_active = T_hybrid_wall
T_GPU_active = T_hybrid_wall
C_hybrid_simplified = T_hybrid_wall_hours × $0.5400
```
If CPU and GPU are active for different durations (measured separately), use:
```
C_hybrid = T_CPU_active_hours × $0.1900 + T_GPU_active_hours × $0.3500
```
The implementation must record which formula was applied and why.

**GCP configuration (frozen before cost evaluation):**

| Parameter | Value |
|---|---|
| Cloud Provider | Google Cloud Platform (GCP) Compute Engine |
| CPU Instance | `n1-standard-4` (4 vCPUs, 15 GB RAM) |
| GPU Instance | `n1-standard-4` + 1× NVIDIA Tesla T4 (16 GB GDDR6) |
| Region | `us-central1` |
| Pricing Tier | On-demand, billed per second |
| CPU VM Rate | $0.1900 /instance-hour |
| T4 GPU Rate | $0.3500 /GPU-hour |
| Combined GPU Node Rate | $0.5400 /hour (used when both resources allocated concurrently) |
| Price Access Date | September 25, 2026 |
| Pricing Source | `https://cloud.google.com/compute/all-pricing` |

---

## 23. Baseline Methods

All baselines execute the **same task graph, inputs, dependency constraints,
and measurement protocol**. No baseline receives information unavailable to the adaptive scheduler.

| Baseline | Definition |
|---|---|
| **CPU-only** | All eligible tasks assigned to CPU. |
| **GPU-only** | All GPU-supported tasks assigned to GPU; tasks with no GPU implementation assigned to CPU (documented). |
| **Rule-based** | GPU assigned if predeclared speedup threshold is met; uses same 3-tier timing estimate as scheduler (see below). |
| **Adaptive ML scheduler** | ML suitability predictor + dependency/transfer/resource context + normalised multi-objective. |

**Rule-based threshold — exact formula with fallback handling:**

```
# Step 1: Obtain time estimates using the same 3-tier lookup as the scheduler (§14.2)
T_CPU_est, T_GPU_est, estimate_source = tier_lookup(workload, input_file)

# Step 2: Compute speedup estimate
estimated_speedup = T_CPU_est / T_GPU_est

# Step 3: Apply frozen threshold
if estimated_speedup > θ_speedup:
    assign GPU
else:
    assign CPU

# Step 4: Log estimate provenance
rule_estimate_source = estimate_source  # one of: exact_lookup, size_scaled_estimate,
                                        #         domain_median_fallback, global_fallback
```

`θ_speedup` is selected from training/profiling data (e.g., ROC-based threshold on
training workloads) and **frozen before test evaluation begins**.
It is not chosen by inspecting test results.

The rule baseline uses the identical 3-tier estimation hierarchy as the adaptive scheduler,
so it never has access to information the ML scheduler does not also have.

**"Best-known assignment"**: best measured assignment among all evaluated strategies
for a given benchmark configuration. Not called globally optimal without formal proof.

---

## 24. Evaluation Metrics

### 24.1 Timing Metrics

```
CPU-to-GPU speedup (per workload):
    Speedup_CPU→GPU = cpu_median_ms / gpu_total_path_ms
    Requires compatible timing definitions.
    BFS: excluded until end-to-end validation closes (§9.2).
    CFD: excluded until experimental closure (§9.4).

Scheduler speedup:
    Speedup_CPU→Scheduler = T_CPU_baseline_makespan / T_scheduler_makespan

Transfer overhead ratio:
    estimated_transfer_ratio = (estimated_T_H2D + estimated_T_D2H) / estimated_T_compute
        (pre-execution; from static analysis; used for scheduling decisions)
    actual_transfer_ratio    = (measured_T_H2D + measured_T_D2H) / measured_T_compute
        (post-execution; for evaluation only)
    BFS actual_transfer_ratio: UNAVAILABLE (§9.5). Not reported until validation closes.
    CFD actual_transfer_ratio: UNAVAILABLE until experimental closure.
```

### 24.2 ML Classification Metrics

- `ml_matches_empirical_preference`: accuracy of ML prediction vs `empirical_preferred_device`
- Precision, Recall, F1 (per class + macro)
- Confusion matrix
- Per-workload behaviour
- Feature importance
- Model A vs Model B ablation (§12.6)
- Comparison against majority-class baseline
- `scheduler_override` rate and `override_reason` distribution (separate from ML accuracy)

### 24.3 System Metrics

- Makespan, per-workload execution time, speedup
- Transfer overhead ratio
- Resource utilisation (if measurable)
- Energy proxy (with sensitivity analysis)
- Modelled cost
- Waiting time (if implemented)
- **Scheduler decision overhead (required): `scheduler_decision_time_ms`**
- ML vs scheduler override analysis

### 24.4 Relative Gap

```
Gap = (Makespan_scheduler - Makespan_best_known) / Makespan_best_known
```

"Optimal" is not used without formal proof.

### 24.5 Pareto Analysis (Conditional)

Used only if NSGA-II is implemented, objectives are normalised, and a non-trivial
Pareto front is generated. Not required for the weighted-sum baseline.

---

## 25. Scheduler Decision Overhead

`scheduler_decision_time_ms` is a **required metric**, measured for every scheduling decision.

**Exact boundary — included in the measurement:**

```
Feature extraction from task record
+ ML inference call (predict())
+ candidate device list generation
+ DAG readiness check + dependency analysis
+ transfer overhead estimation
+ objective score normalization (all candidates)
+ dynamic weight adaptation step
+ final device selection
= scheduler_decision_time_ms
```

**Excluded from the measurement:**

```
Actual workload execution time
Data movement to/from profiling storage
Post-execution telemetry recording
One-time model loading: joblib.load(...) or equivalent
One-time scheduler initialization
```

> **Model loading note:** `joblib.load()` and one-time initialization are performed
> before the scheduling loop begins. They are **reported separately** as one-time startup
> cost, NOT included in `scheduler_decision_time_ms`. Including them would inflate the
> first-decision measurement and misrepresent steady-state per-decision overhead.

`scheduler_decision_time_ms` is measured for every scheduling decision and summarised
(median, range). This overhead is compared against scheduling benefit to assess justification.

---

## 26. Frozen Implementation Order

**Do not begin ML training or scheduler optimization before timing gates are closed.**

```
 1.  Inspect euler3d.cu; document CFD GPU timing semantics.
 2.  Capture actual CFD CPU stdout; implement and verify CFD CPU parser from actual output.
 3.  Verify CFD GPU parser (value-first pattern) against captured output.
 4.  Conduct BFS total-path validation experiment (§9.2); finalise formula.
 5.  Freeze Layer 1 per-run schema and workload-specific parser design.
 6.  Profile additional workloads one by one: workload-specific commands and parsers.
 7.  Preserve all raw successes and failures.
 8.  Aggregate valid runs into Layer 2 with medians, spread, and comparability flags.
 8a. Freeze workload metadata schema and feature-generation code (Rubric v1.0, §12.4).
     Assign expert feature scores programmatically from the frozen rubric before
     Layer 3 is populated. Do not assign scores manually after feature code is frozen.
 9.  Construct Layer 3 using only valid-label configurations (§10 Layer 3 rule).
10.  Generate preferred_device only for comparable workload/input configurations (§13).
11.  Train Random Forest with LOWO evaluation; run Model A and Model B ablation (§12.6).
12.  Report per-workload and macro metrics; compare against majority-class baseline.
13.  Implement rule-based baseline; freeze θ_speedup from training data.
14.  Add validated Layer 2 lookup execution-time estimates with fallback policy (§14.2).
15.  Implement adaptive scheduler candidate-generation layer (§15).
16.  Add dependency awareness and project-created DAG (§16).
17.  Add transfer-aware cost calculations (§17).
18.  Add frozen-reference objective normalisation (§18.2).
19.  Add measurable dynamic objective weighting; freeze all initial values and
     thresholds before evaluation (§20.1).
20.  Add scheduler_decision_time_ms measurement with defined boundary (§25).
21.  Implement hybrid execution empirical experiment for one partitionable workload (§19).
22.  Add feedback/retraining loop (§27.2).
23.  Freeze cloud cost provider/configuration/rates; record price_access_date (§22).
24.  Run CPU-only, GPU-only, rule-based, and adaptive scheduler baselines.
25.  Analyse ML predictions versus scheduler overrides (§15.2).
26.  Run energy sensitivity analysis (§21).
27.  Only then consider optional NSGA-II, richer regression, or additional ML models.
```

---

## 27. Feedback and Retraining

### 27.1 Objective Weight Adaptation (Fast Loop)

After every scheduled task: observe system state → apply rule-based weight update
(§20.1) → clip to bounds → renormalise → use updated weights for next decision.

### 27.2 ML Retraining (Slow Loop)

```
Store new labeled record (pre-execution features, empirical_preferred_device)
      ↓
Append to dataset
      ↓
When N_retrain new samples accumulated
      ↓
Retrain Random Forest from expanded dataset (full refit, not partial_fit)
      ↓
Evaluate on held-out fold
      ↓
Deploy updated model if evaluation passes
```

> **Implementation note:** Standard `sklearn.RandomForestClassifier` does not support
> genuine online incremental learning via `partial_fit`. This is **periodic retraining
> from an expanded dataset**, not incremental Random Forest updating. Any claim of
> "incrementally update Random Forest" is incorrect and must not appear.

> **N_retrain parameter:** `N_retrain` is a declared experimental parameter that will
> be fixed before evaluation begins. Its value is not pre-justified theoretically;
> the chosen value, rationale, and effect on retraining frequency will be documented
> before final evaluation runs.

Two separate loops. Not merged into a vague "adaptive feedback" mechanism.

---

## 28. Research Hypotheses

1. **H1**: The adaptive scheduler may reduce makespan relative to fixed CPU-only / GPU-only
   policies on workloads where device suitability varies across the mix.
2. **H2**: Transfer-aware scheduling may reduce poor GPU assignments for transfer-heavy tasks.
3. **H3**: Dynamic objective weighting may change device assignments when `gpu_free`,
   `ready_task_count`, or `transfer_ratio` cross their thresholds.
4. **H4**: Hybrid execution may improve performance for the chosen partitionable workload;
   not expected to benefit workloads with strong serial dependencies.
5. **H5**: A Random Forest on pre-execution features may predict preferred device above a
   majority-class baseline; generalisation claims beyond the measured workload set require
   careful interpretation given small dataset size.
6. **H6 (ablation)**: Removing `workload_type` and `workload_domain` (Model B) may reduce
   accuracy while improving cross-workload generalisation relative to Model A.

---

## 29. Limitations

1. Kaggle is a single-node environment; multi-node HPC is out of scope.
2. One GPU is used as the controlled scheduling resource; multi-GPU scheduling is out of scope.
3. Initial ML dataset is small; results must be interpreted with appropriate caution.
4. Rodinia workloads may not represent all real-world cloud HPC applications.
5. BFS `gpu_total_path_time_status = "pending_validation"`; BFS `empirical_preferred_device` is blocked.
6. CFD `status = "source_inspection_complete__experimental_closure_pending"`; CFD `empirical_preferred_device` is blocked.
7. BFS `actual_transfer_ratio` is unavailable until end-to-end validation closes.
8. Some workloads may not be technically partitionable; hybrid scope is limited.
9. Energy values are simplified proxies (assumed average power × time), not direct measurements;
   sensitivity analysis assesses robustness.
10. Cloud cost is modelled from documented pricing assumptions, not actual Kaggle billing.
11. Generalisation beyond the measured Rodinia workload/feature space is not claimed.
12. LOWO cross-validation may not give stable estimates for very small workload sets.
13. Scheduler DAG is project-constructed; not inherent to Rodinia.
14. Execution-time lookup fallback (Tier-2 size-scaling) assumes linear scaling, which
    may not hold for all workloads; accuracy must be evaluated.
15. `preferred_device` labels are empirical under this experiment's conditions; they are
    not intrinsic workload properties and may differ under other hardware/software configurations.
16. Hybrid execution results are not validated until the empirical experiment is run.
17. Scheduler weight adaptation uses rule-based triggers, not ML-learned weights.

---

## 30. Novelty and Contribution

The contribution is the **integrated framework**:

1. Workload-specific profiling with source-validated timing definitions and a
   documented validation experiment for resolving double-counting.
2. A three-layer dataset with an explicit valid-label gate preventing label leakage.
3. ML device suitability prediction from pre-execution features with workload-identity
   ablation to separate memorisation from genuine generalisation.
4. A two-stage architecture separating the ML predictor from the runtime scheduler.
5. Dependency-aware scheduling over a project-created DAG.
6. Transfer-aware scheduling with static/measured transfer distinction.
7. Normalised multi-objective optimization with frozen reference and measurable weights.
8. Hybrid execution characterised through empirical partition-ratio experiments.
9. Feedback loop for retraining and weight adaptation.

---

## 31. Module Structure

```
project/
├── main.py
├── requirements.txt
│
├── profiler/
│   ├── workload_configs.py      ← per-workload configs + timing notes
│   ├── workload_profiler.py     ← runs CPU/GPU; stores complete per-run records
│   ├── parsers.py               ← per-workload stdout timing parsers
│   └── measurement_utils.py    ← file-size, failure-handling, encoding utilities
│
├── dataset/
│   ├── raw_data/                ← Layer 1 CSVs
│   ├── aggregate_measurements.py  ← Layer 1 → Layer 2
│   └── build_ml_dataset.py     ← Layer 2 → Layer 3 (valid-label gate enforced)
│
├── ml/
│   ├── feature_schema.py        ← frozen Rubric v1.0 + feature-generation code
│   ├── train_classifier.py      ← Random Forest + LOWO + Model A/B ablation
│   ├── evaluate_classifier.py   ← metrics, confusion matrix, feature importance
│   └── predict.py               ← inference: pre-execution features → device
│
├── scheduler/
│   ├── scheduler.py             ← 19-step decision orchestrator
│   ├── dag_builder.py           ← project-created DAG, longest-path analysis
│   ├── transfer_model.py        ← transfer overhead estimation + fallback
│   ├── objective_function.py    ← frozen-reference normalised weighted-sum scorer
│   └── adaptive_weights.py      ← rule-based weight adaptation + frozen params
│
├── executor/
│   ├── cpu_executor.py          ← OpenMP subprocess runner
│   ├── gpu_executor.py          ← CUDA subprocess runner (single GPU device)
│   └── hybrid_executor.py       ← empirical partition-ratio model; one workload
│
└── analysis/
    ├── performance_analysis.py  ← speedup, transfer ratio, energy sensitivity
    ├── scheduler_comparison.py  ← baseline comparisons + override analysis
    └── visualization.py         ← charts, optional Pareto, weight evolution
```

---

## 32. Near-Final Implementation-Gated Checklist

**Timing & Semantics Gates — Status uses: VERIFIED / PENDING / PLANNED**

| Gate | Status | Evidence |
|---|---|---|
| BFS source segment structure identified (H2D → Loop → D2H) | VERIFIED | Source inspection of `bfs.cu` / `bfs.cpp` |
| BFS end-to-end total-path validation experiment | **PENDING** | Validation experiment not yet performed |
| BFS `gpu_total_path_time_status = "validated"` | **PENDING** | Blocked on validation experiment |
| BFS `empirical_preferred_device` assigned | **PENDING** | Blocked on total-path validation |
| BFS `actual_transfer_ratio` available | **PENDING** | Blocked on validation + pure-compute interval |
| CFD GPU timing scope specified from source (`euler3d.cu:566–589`) | VERIFIED | Source inspection complete |
| CFD CPU timing scope specified from source (`euler3d_cpu.cpp:472–494`) | VERIFIED | Source inspection complete |
| CFD parsers verified against actual stdout | **PENDING** | `euler3d_cpu` not yet executed |
| CFD definitional comparability experimentally confirmed | **PENDING** | Blocked on parser execution |
| CFD `empirical_preferred_device` assigned | **PENDING** | Blocked on experimental closure |

**Specification & Design Gates (PLANNED = designed, not yet implemented)**

| Gate | Status |
|---|---|
| DAG node semantics frozen (one node = one scheduling task; §16.1) | DEFINED (specification) |
| DAG concrete example documented (§16.1) | DEFINED (specification) |
| `dependency_count` defined as scheduler-context feature, not Rodinia-intrinsic (§16.1) | DEFINED (specification) |
| Transfer-time operational formula defined with bandwidth model (§17.2) | DEFINED (specification) |
| Pre-execution `estimated_transfer_ratio` vs post-execution `actual_transfer_ratio` defined (§17.3) | DEFINED (specification) |
| Candidate-level scoring quantities defined per candidate type (§18.1) | DEFINED (specification) |
| Double-counting guard for GPU path time documented (§18.1) | DEFINED (specification) |
| Hybrid cost uses resource-active-time formulation (§22) | DEFINED (specification) |
| GPU pressure = 0.0 in primary experiment (binary gpu_free); future extension labelled OPTIONAL (§18.3) | DEFINED (specification) |
| Input size unit standardized to binary MiB throughout (§10, §12.2, §12.5, §14.2) | DEFINED (specification) |
| Tier-3 CPU and GPU medians computed separately (§14.2) | DEFINED (specification) |
| Rule baseline uses same 3-tier lookup as scheduler; `rule_estimate_source` logged (§23) | DEFINED (specification) |
| Normalization: out-of-range clipping (`z_norm_clipped = clip(z_norm, 0, 1)`) defined (§18.2) | DEFINED (specification) |
| Cloud cost formula uses resource-active-time; hybrid formula variants documented (§22) | DEFINED (specification) |

**Active Implementation Milestones (all PLANNED)**
- [ ] BFS end-to-end validation experiment; close BFS timing gate.
- [ ] CFD execution run; capture stdout; verify parsers; close CFD timing gate.
- [ ] Layer 1 raw execution profiling across all target Rodinia workloads (≥6 runs + 1 warmup).
- [ ] Layer 2 aggregated performance dataset with medians, spread, comparability flags.
- [ ] Layer 3 ML dataset enforcing `empirical_preferred_device` gating only.
- [ ] Model A vs Model B Random Forest training and LOWO evaluation.
- [ ] Multi-objective adaptive scheduler execution across project DAG.
- [ ] Comparative makespan, cost, energy, and decision overhead evaluation against all baselines.

---

## 33. Guiding Principle

> *Measure first. Define semantics second. Freeze metadata and rubrics third.*
> *Aggregate fourth. Then build the predictor and scheduler from information*
> *genuinely available before execution.*
>
> *Do not add advanced algorithms before the measurement and dataset foundations are stable.*
> *Do not tune any parameter — weights, thresholds, cost rates — on the test set.*

---

## 34. Error-Correction Audit

| # | Error / Issue | Correction Made | Status |
|---|---|---|---|
| 1 | BFS `gpu_total_path_time_status = "validated_non_overlapping_formula"` contradicted §13.2/§29 | Changed to `"pending_validation"` throughout §8.2, §9.2, §9.3, §13.2, §29, §32 | CORRECTED |
| 2 | §9.2 presented candidate sum as validated formula | Retitled "Candidate Total GPU Offload Path (PENDING)"; end-to-end experiment explicitly stated as not performed | CORRECTED |
| 3 | §9.3 table said BFS cleared for Layer 3 | Table changed to PENDING; clearing statement removed | CORRECTED |
| 4 | §9.5 reported `transfer_ratio_bfs ≈ 3.89` as a fact | Replaced with UNAVAILABLE; no numerical ratio stated | CORRECTED |
| 5 | CFD config `status = "profiling_verified__source_semantics_validated"` | Changed to `"source_inspection_complete__experimental_closure_pending"` | CORRECTED |
| 6 | §9.4 titled "Verified Semantics" despite parsers not run | Retitled PENDING; what-is-included table added | CORRECTED |
| 7 | `preferred_device` conflated with intrinsic workload property | Empirical definition added; four distinct concepts tabulated | CORRECTED |
| 8 | §15.2 `prediction_correct` compared ML vs scheduler, not ML vs empirical label | `ml_matches_empirical_preference` defined; override ≠ ML error stated | CORRECTED |
| 9 | Tier-2 called "Scaled Interpolation" | Renamed "Empirical Size-Based Scaling"; linear-scaling assumption stated | CORRECTED |
| 10 | Tier-3 fallback not distinguished from ML generalization | Explicit callout added; evaluated separately from LOWO | CORRECTED |
| 11 | Model B described as achieving generalization from intrinsic characteristics | Corrected to "reduced explicit workload-identity dependence" | CORRECTED |
| 12 | Rubric thresholds without scope note | Project-defined heuristic note added | CORRECTED |
| 13 | §17.2 did not distinguish pre/post transfer ratio | `estimated_transfer_ratio` vs `actual_transfer_ratio` split; leakage rule stated | CORRECTED |
| 14 | Normalization used epsilon padding without deterministic degenerate rule | Explicit if/else with `z_norm = 0.0` degenerate case | CORRECTED |
| 15 | GPU candidate penalized even when already eliminated | Hard constraint / soft penalty separation added | CORRECTED |
| 16 | §19.1 hybrid formula assumed concurrency without stating it | Concurrency assumption + serial alternative formula added | CORRECTED |
| 17 | §20.1 trigger used ambiguous `transfer_ratio` | Changed to `estimated_transfer_ratio` (pre-execution only) | CORRECTED |
| 18 | Energy trigger undefined external flag | `energy_mode ∈ {"normal","energy_priority"}` defined | CORRECTED |
| 19 | Energy power labels misleading | Relabelled "Assumed average device power for sensitivity model" | CORRECTED |
| 20 | Cost formula double-counted CPU+GPU rates | Resource-active-time formulation with explicit hybrid variants | CORRECTED |
| 21 | Model loading included in `scheduler_decision_time_ms` | One-time initialization explicitly excluded; reported separately | CORRECTED |
| 22 | `partial_fit` / incremental RF stated; `N=5` example presented as fixed | "Periodic retraining from expanded dataset"; `N_retrain` declared as parameter | CORRECTED |
| 23 | §32 checklist said "Formally Resolved" for pending items | Replaced with VERIFIED/PENDING/PLANNED table | CORRECTED |
| 24 | §29 limitations used vague wording | Field-level status strings used | CORRECTED |
| 25 | Critical-path not distinguished from full makespan | "Critical-path proxy" label; limitations stated | CORRECTED |
| 26 | §12.8 telemetry table missing key fields | Eight fields added | CORRECTED |
| 27 | §24.1 used single `transfer_ratio`; §24.2 accuracy not specified | Split into estimated/actual; `ml_matches_empirical_preference` defined | CORRECTED |
| 28 | BFS/CFD configs labelled "verified" | Corrected to pending-status strings | CORRECTED |
| 29 | DAG node semantics not defined | §16.1 definitions table + concrete example added | DEFINED |
| 30 | Transfer-time formula missing operational definition | §17.2 bandwidth model formula added; calibration sources specified | DEFINED |
| 31 | Candidate-level scoring quantities not defined per candidate | §18.1 candidate table: T, T_transfer, E, C, R per CPU/GPU/Hybrid | DEFINED |
| 32 | Hybrid cost formula did not define resource-active time | §22 resource-active-time formulas; concurrent vs serial variants | DEFINED |
| 33 | GPU pressure contradiction (binary `gpu_free` vs continuous pressure) | §18.3 primary experiment: `gpu_pressure = 0.0`; future extension labelled OPTIONAL | CORRECTED |
| 34 | Input-size units mixed (MB vs MiB, 1024 vs 1e6) | `input_size_mib` standardized throughout §10, §12.2, §12.5, §14.2 | CORRECTED |
| 35 | Tier-3 used single combined median for CPU and GPU | CPU and GPU medians computed separately from valid Layer-2 records | DEFINED |
| 36 | Rule baseline fallback behavior undefined for unprofiled workloads | §23 rule baseline uses same 3-tier lookup; `rule_estimate_source` logged | DEFINED |
| 37 | Normalization out-of-range behavior undefined | §18.2 clipping: `z_norm_clipped = clip(z_norm, 0, 1)` after training-derived normalization | DEFINED |

---

## 35. ERROR-CORRECTION VERIFICATION REPORT

| Issue | Status | Where corrected |
|---|---|---|
| DAG task semantics | DEFINED | §16.1: definitions table + concrete DAG example |
| Transfer-time formula | DEFINED | §17.2: operational formula with bandwidth model |
| Candidate scoring | DEFINED | §18.1: candidate-level quantity table (T, T_transfer, E, C, R) |
| Hybrid cost | DEFINED | §22: resource-active-time formulas; concurrent vs serial variants |
| GPU pressure contradiction | CORRECTED | §18.3: primary experiment `gpu_pressure = 0.0`; §16.1 hard constraint separated |
| Input-size units | CORRECTED | §10 Layer 1, §12.2 features, §12.5, §14.2: `input_size_mib` = bytes/(1024×1024) |
| Tier-3 estimates | DEFINED | §14.2: `T_CPU_Tier3` and `T_GPU_Tier3` computed separately from valid Layer-2 records |
| Rule baseline fallback | DEFINED | §23: same 3-tier lookup as scheduler; `rule_estimate_source` logged |
| Normalization out-of-range | DEFINED | §18.2: `z_norm_clipped = clip(z_norm, 0, 1)` |
| Error-correction audit | UPDATED | §34: all 37 items; CORRECTED / DEFINED / PENDING EXPERIMENTAL VALIDATION |

---

## 36. REMAINING EXPERIMENTAL GATES

These items require **actual implementation and/or experimental measurement**.
They are NOT completed by the definitions above.

| Gate | What is Required |
|---|---|
| BFS end-to-end validation | Run instrumented BFS; confirm H2D + Loop + D2H covers the complete offload path; close `gpu_total_path_time_status` |
| BFS `empirical_preferred_device` | Requires closed BFS validation gate above |
| BFS `actual_transfer_ratio` | Requires isolated pure-compute interval from validated timing |
| CFD parser execution | Execute `euler3d_cpu` and `euler3d`; capture actual stdout; verify regex against real output |
| CFD `empirical_preferred_device` | Requires closed CFD experimental closure gate above |
| Bandwidth calibration for transfer model | Measure actual PCIe H2D/D2H bandwidth on Kaggle T4; derive `calibrated_H2D_bandwidth_bytes_per_ms` and `calibrated_D2H_bandwidth_bytes_per_ms` |
| Layer 1 profiling | ≥6 timed runs per workload per device for all target Rodinia workloads |
| Layer 2 aggregation | Build aggregated performance dataset from valid Layer-1 rows |
| Layer 3 ML dataset | Enforce `empirical_preferred_device` gating; construct ML samples |
| Model A / B training and LOWO | Train Random Forest; run LOWO cross-validation; report metrics |
| `θ_speedup` selection | Fit ROC-based threshold from training workloads; freeze before evaluation |
| Normalization range fitting | Fit `z_min` / `z_max` from training/profiling data; freeze before evaluation |
| `N_retrain` parameter selection | Fix retraining threshold value before evaluation begins; document rationale |
| Hybrid experiment | Run one partitionable workload at `r ∈ {0, 0.25, 0.5, 0.75, 1.0}`; measure T_hybrid empirically |
| Scheduler evaluation | Execute adaptive scheduler across project DAG; measure makespan, cost, energy, decision overhead |
| Energy sensitivity analysis | Run 9-point power grid analysis (§21) |
| Baseline comparisons | Run CPU-only, GPU-only, rule-based baselines under identical task graph |
| Feedback loop | Accumulate labeled records; trigger `N_retrain` retraining cycle; deploy updated model |
