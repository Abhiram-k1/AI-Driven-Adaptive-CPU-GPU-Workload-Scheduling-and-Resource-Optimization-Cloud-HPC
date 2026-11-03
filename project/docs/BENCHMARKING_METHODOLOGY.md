# BENCHMARKING METHODOLOGY & DATASET CURATION
## Rodinia 3.1 Workload Profiling, Timing Validation, and Repeated Runs Protocol

---

## 1. Workload Selection and Domain Coverage

The system benchmarks the Rodinia 3.1 heterogeneous benchmark suite across diverse computational dwarfs and application domains.

| Workload | Domain | Algorithmic Dwarf | Primary Memory Pattern | Compute Phase Semantics |
|---|---|---|---|---|
| **BFS** | Graph Traversal | Graph / Breadth-First | Irregular pointer indirection | Level-by-level frontier queue update |
| **CFD** | Fluid Dynamics | Structured Grid / Solver | Contiguous 3D element arrays | 2000 Runge-Kutta time-integration steps |
| **Hotspot** | Physics Simulation | Structured 2D Stencil | Nearest-neighbor 2D stencil | Iterative differential heat transfer equation |
| **LUD** | Linear Algebra | Dense Matrix Decomposition | Blocked contiguous 2D tiles | In-place diagonal elimination & panel update |
| **KMeans** | Data Mining | Dense Clustering | Contiguous feature vectors | Euclidean distance accumulation to K centers |
| **SRAD** | Image Processing | Structured 2D Grid | 2D pixel raster matrix | Anisotropic non-linear diffusion equations |
| **NN** | Data Mining | Distance Calculation | Sequential database record reads | Euclidean distance evaluation across 2D points |
| **NW** | Bioinformatics | Dynamic Programming | Anti-diagonal wavefront matrix | Scoring matrix anti-diagonal dependency |
| **Backprop** | Machine Learning | Neural Network / Dense Math | Contiguous layer weight matrices | Feedforward & error backpropagation matrix mults |
| **Pathfinder** | Dynamic Programming | 2D Grid Traversal | Rectangular array rows | Min-path calculation across preceding row |
| **Particlefilter** | Tracking | Sequential Monte Carlo | Contiguous particle state arrays | Particle state transition & exponential weights |
| **Heartwall** | Medical Imaging | Template Tracking | Multi-frame video buffer | Cross-correlation template matching |
| **B-Tree** | Data Structures | Tree Indexing | Irregular pointer-linked nodes | Tree traversal & binary search |
| **Leukocyte** | Medical Imaging | Cell Detection | Video frame buffers | GVF gradient matrix & snake optimization |
| **Streamcluster** | Data Mining | Streaming Clustering | Streaming point buffers | Median-based center gain evaluations |
| **Hybridsort** | Parallel Sorting | Sorting | Buckets & sub-arrays | Parallel bucket sort + merge sort |
| **LavaMD** | Molecular Dynamics | N-body Simulation | 3D space sub-boxes | Pairwise short-range potential calculation |
| **Gaussian** | Linear Algebra | Linear System Elimination | 2D matrix rows | Sequential pivot row forward elimination |
| **MummerGPU** | Bioinformatics | Sequence Alignment | Suffix tree pointer lookups | Exact string matching across genome suffixes |

---

## 2. Repeated Measurement Protocol & Statistical Reliability

To eliminate operating system scheduling jitter, instruction cache misses, and thermal throttling transients, every benchmark execution follows strict statistical gating:

1. **Warm-up Run (Run ID 0)**:
   - Always executed prior to data acquisition.
   - Initialized device memory allocations and populates hardware instruction/data caches.
   - Tagged as `is_warmup = True` in Layer 1 raw dataset.
   - **Strictly excluded from all summary statistics**.

2. **Timed Repetitions (Runs 1 to 6+)**:
   - Minimum of 6 independent runs collected per configuration.
   - Every individual run timestamp and execution output is preserved.

3. **Summary Statistics**:
   - **Median**: Primary statistic (robust against tail latency and OS interrupts).
   - **Mean, Std Deviation, Min, Max**: Supporting dispersion metrics.

$$\text{Median} = \text{Q}_{50}(T_1, T_2, \dots, T_n)$$

---

## 3. Source-Level Timing Instrumentation vs Wall-Clock Time

Generic parsers and process wall-clock timers (\`time.time()\`, \`/usr/bin/time\`) conflate file I/O, dataset parsing, and JVM/CUDA initialization with pure algorithmic computation.

The framework enforces source-level regex timing parsers:

| Workload | Platform | Timing Source Location | Extracted Metric | Unit |
|---|---|---|---|---|
| **BFS** | CPU (OpenMP) | `openmp/bfs/bfs.cpp:173` | `Compute time: <val>` | Seconds |
| **BFS** | GPU (CUDA) | `cuda/bfs/bfs.cu` | `H2D` + `Kernel` + `D2H` | Milliseconds |
| **CFD** | CPU (OpenMP) | `openmp/cfd/euler3d_cpu.cpp:494` | `Compute time: <val>` (2000 RK iter) | Seconds |
| **CFD** | GPU (CUDA) | `cuda/cfd/euler3d.cu:589` | `<val> seconds per iteration` (x2000) | Seconds/iter |
| **Hotspot** | CPU & GPU | Source timer loop | `Compute time: <val>` / `Time: <val>` | Seconds |
| **LUD** | CPU & GPU | Matrix decomposition loop | `Time: <val>` | Seconds |
| **KMeans** | CPU & GPU | Clustering loop | `Compute time: <val>` / `Time for kMeans:` | Seconds |
| **SRAD** | CPU & GPU | Diffusion solver loop | `Computation time: <val>` | Seconds |
| **NN** | CPU & GPU | Distance loop | `Time: <val>` | Seconds |

### End-to-End GPU Offload Path Time Definition
For discrete accelerators, offload benefit must account for interconnect overhead:

$$T_{\text{GPU\_total}} = T_{\text{Host-to-Device (H2D)}} + T_{\text{Kernel\_Execution}} + T_{\text{Device-to-Host (D2H)}}$$

---

## 4. Multi-Layer Dataset Pipeline

```
Raw Execution Runs (Layer 1)
  [335 raw records across 7 profiled workloads on Tesla T4]
             │
             ▼
Aggregated Measurements (Layer 2)
  [8 validated configurations, median timings, failure tracking]
             │
             ▼
Pre-Execution ML Feature Gating (Layer 3)
  [Strict Anti-Leakage Assertion: FORBIDDEN_LEAKAGE_COLUMNS excluded]
             │
             ▼
Expanded Multi-Scale Dataset (Layer 3 Expanded)
  [57 configurations across 19 workloads and S/M/L input profiles]
```

---

## 5. Empirical Hardware Validation Summary (Tesla T4)

Measurements collected on dedicated NVIDIA Tesla T4 (16GB GDDR6, PCIe 3.0 x16, 70W TDP) and 4-core host CPU:

| Workload | Input File | Input Size | CPU Median | GPU Total Path | Speedup | Preferred Device |
|---|---|---|---|---|---|---|
| **BFS** | `graph1MW_6.txt` | 61.22 MiB | 36.12 ms | 18.73 ms | 1.93x | **GPU** |
| **BFS** | `graph4096.txt` | 0.05 MiB | 1.25 ms | 8.85 ms | 0.14x | **CPU** |
| **CFD** | `fvcorr.domn.193K` | 43.54 MiB | 77,537.30 ms | 1,125.05 ms | 68.92x | **GPU** |
| **Hotspot** | `temp_512` | 2.75 MiB | 1,486.75 ms | 418.60 ms | 3.55x | **GPU** |
| **KMeans** | `819200.txt` | 96.98 MiB | 3,216.70 ms | 1,103.80 ms | 2.91x | **GPU** |
| **LUD** | `512.dat` | 2.48 MiB | 2,048.85 ms | 279.65 ms | 7.33x | **GPU** |
| **NN** | `cane4_0.db` | 0.50 MiB | 18.45 ms | 28.55 ms | 0.65x | **CPU** |
| **SRAD** | `image.pgm` | 0.75 MiB | 1,783.80 ms | 553.05 ms | 3.23x | **GPU** |

