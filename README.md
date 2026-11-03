# AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduling and Resource Optimization for Cloud HPC

**B.Tech AI & Data Science Capstone Project — SEM-5 HPC**  
**Final Production Release & Academic Evaluation Repository**

---

## 1. Project Overview & Abstract

Heterogeneous Cloud High-Performance Computing (HPC) nodes pairing multi-core CPUs with discrete accelerators (NVIDIA GPUs) offer tremendous parallel compute density. However, naive scheduling leads to severe pipeline stalls, PCIe bus contention, and excessive energy consumption.

This repository provides a complete, experimentally validated, presentation-ready **AI-Driven Adaptive Multi-Objective CPU–GPU Workload Scheduling System**. The system:
- Characterizes scientific workloads prior to execution using a frozen, zero-leakage algorithmic rubric.
- Employs Random Forest and Gradient Boosted decision trees to predict optimal device placement (CPU vs GPU) with 87.50% cross-validation accuracy.
- Builds a task dependency Directed Acyclic Graph (DAG) with bottom-up critical path prioritization.
- Samples dynamic multi-objective weights using Bayesian Thompson Sampling over a Dirichlet posterior distribution.
- Optimizes assignments across concurrent multi-slot CPU and GPU worker pools to balance workflow makespan, TDP energy proxies, and host-device transfer overheads.
- Features a live Plotly Dash web dashboard with real-time host telemetry, benchmark inspection, and interactive scheduling.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INTERACTIVE LIVE DASHBOARD                      │
│      (Plotly Dash Web UI: Real-Time CPU/RAM, Benchmarks, Scheduler)     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
┌────────────────────────────────────▼───────────────────────────────────┐
│                    SCHEDULING & OPTIMIZATION LAYER                     │
│  - WorkloadDAG (Critical-Path Analysis & Dependency Ordering)          │
│  - AdaptiveWeightController (Dirichlet Thompson Sampling Optimization) │
│  - WeightedSumScheduler & NSGAScheduler (Multi-Objective Solvers)      │
│  - ConcurrentSchedulerQueue (4 CPU Slots, 1 Dedicated GPU Slot)        │
└───────────────────▲────────────────────────────────┬───────────────────┘
                    │                                │
┌───────────────────┴──────────────┐   ┌─────────────▼───────────────────┐
│     PREDICTIVE MACHINE LEARNING  │   │  HARDWARE EXECUTION & TELEMETRY │
│  - Frozen Rubric v1.0 (19 Workloads) │ - CPUExecutor & GPUExecutor     │
│  - Model A (RF Full, 13 Features)│   │  - SimulatedExecutor (T4 Replay)│
│  - Model B (RF Ablated, 11 Feat.)│   │  - ResourceMonitor (psutil/NVML)│
│  - Model C (Gradient Boosting)   │   │  - EnergyEstimator (TDP Model)  │
│  - LOWO Holdout Evaluation       │   │  - 36 Automated Unit/Integ Tests│
└──────────────────────────────────┘   └─────────────────────────────────┘
```

---

## 3. Technology Stack

- **Core Runtime**: Python 3.12+ (Compatible with Linux and Windows x86_64)
- **Machine Learning**: `scikit-learn>=1.3`, `numpy>=1.24`, `pandas>=2.0`
- **Graph & Optimization**: `networkx>=3.1`, `deap>=1.4` (NSGA-II)
- **System Monitoring**: `psutil>=5.9`, Subprocess NVML query probe
- **Visualization & Dashboard**: `dash>=4.5`, `plotly>=5.15`, `matplotlib>=3.7`, `seaborn>=0.12`
- **Automated Testing**: `pytest>=9.0`

---

## 4. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/Abhiram-k1/AI-Driven-Adaptive-CPU-GPU-Workload-Scheduling-and-Resource-Optimization-Cloud-HPC.git
cd AI-Driven-Adaptive-CPU-GPU-Workload-Scheduling-and-Resource-Optimization-Cloud-HPC/project

# Install dependencies
pip install -r requirements.txt
pip install dash
```

---

## 5. Execution Instructions

### A. Run Objective 1 & 2 Master Pipeline
Executes dataset validation, Random Forest training, Leave-One-Workload-Out (LOWO) cross-validation, and visualization generation:
```bash
python main.py
```

### B. Run the Multi-Objective AI Scheduler
Runs the DAG builder, predictive placement, Thompson Sampling, and multi-slot execution:
```bash
python scheduler/scheduler.py --workloads bfs cfd hotspot lud kmeans nn srad --mode balanced
```

### C. Run Multi-Strategy Scheduling Benchmark Comparison
Evaluates AI-Adaptive against FIFO-CPU, FIFO-GPU, Round-Robin, Greedy-Speedup, and Random baselines:
```bash
python scheduler/strategy_comparison.py
```

### D. Launch the Interactive Live Dashboard
Launches the Plotly Dash application with live 2-second telemetry:
```bash
python dashboard/run_dashboard.py
# Open your browser and navigate to: http://localhost:8050
```

### E. Run the Complete Automated Test Suite
Executes all 36 unit and integration tests:
```bash
python -m pytest tests/ -v
```

---

## 6. Key Experimental Findings

### A. Scheduling Benchmark Summary (7 Workload Workflow on Tesla T4)
| Scheduling Strategy | Workflow Makespan (s) | Total Energy (J) | GPU Tasks | CPU Tasks | Speedup vs Pure CPU |
|---|---|---|---|---|---|
| **AI-Adaptive (Proposed)** | **2.229** | **429.70** | 2 | 5 | **34.79×** |
| **Greedy-Speedup** | 3.499 | 209.46 | 6 | 1 | 22.16× |
| **FIFO-GPU** | 3.527 | 210.14 | 7 | 0 | 21.98× |
| **FIFO-CPU** | 77.537 | 4,758.57 | 0 | 7 | 1.00× |
| **Round-Robin** | 77.537 | 4,522.96 | 4 | 3 | 1.00× |
| **Random Baseline** | 77.537 | 4,539.01 | 5 | 2 | 1.00× |

*Key Insight*: In policies that blindly schedule tasks (FIFO-CPU, Round-Robin, Random), assigning CFD to CPU creates a 77.5-second bottleneck. Conversely, assigning all tasks to the GPU serializes execution in the single GPU slot. AI-Adaptive achieves the fastest makespan (2.229s, 34.79× speedup) by overlapping CPU-suited tasks while isolating CFD on the GPU.

### B. Machine Learning LOWO Generalization
- **Model A (Full Features)**: Accuracy: **87.50%**, Macro F1-Score: **0.7949**
- **Model B (Identity-Free Ablated)**: Accuracy: **87.50%**, Macro F1-Score: **0.7949**
- **Model C (Gradient Boosting)**: Accuracy: 75.00%, Macro F1-Score: 0.6667
- **Majority-Class Baseline**: Accuracy: 75.00%, Macro F1-Score: 0.4286

---

## 7. Documentation Index

Comprehensive technical documentation is provided in `project/docs/`:
1. [`PROJECT_ARCHITECTURE.md`](project/docs/PROJECT_ARCHITECTURE.md) — 5-layer architectural model and component interactions.
2. [`BENCHMARKING_METHODOLOGY.md`](project/docs/BENCHMARKING_METHODOLOGY.md) — Rodinia 3.1 profiling, regex timing, repeated runs protocol.
3. [`ML_MODEL_EVALUATION.md`](project/docs/ML_MODEL_EVALUATION.md) — Pre-execution features, anti-leakage guarantees, and LOWO holdout results.
4. [`OPTIMIZATION_METHODOLOGY.md`](project/docs/OPTIMIZATION_METHODOLOGY.md) — Multi-objective scalarization, NSGA-II, and Bayesian Thompson Sampling.
5. [`TESTING_AND_VALIDATION.md`](project/docs/TESTING_AND_VALIDATION.md) — Detailed test cases and automated regression suite.
6. [`PROJECT_COMPLETION_REPORT.md`](project/docs/PROJECT_COMPLETION_REPORT.md) — Final completion audit and 19-requirement tracking matrix.
7. [`IEEE_FINAL_REPORT.md`](project/docs/IEEE_FINAL_REPORT.md) — Academic research paper formatted to strict IEEE conference standards.

---

## 8. Hardware & Environmental Transparency

- **Empirical Hardware Grounding**: Real GPU measurements were captured on an NVIDIA Tesla T4 (16 GB GDDR6, 70W TDP) and 4-core Intel Xeon host hosted in Kaggle Cloud.
- **Local Host Execution**: On local development machines without NVIDIA CUDA GPUs, the framework routes to `SimulatedExecutor`, replaying empirical Tesla T4 distributions with verified Gaussian variance.
- **No Fabricated Telemetry**: If a physical GPU is absent, `ResourceMonitor` explicitly reports `GPU_MONITORING_UNAVAILABLE`. All energy metrics are calibrated TDP proxies explicitly tagged as `ESTIMATED_NOT_MEASURED`.
