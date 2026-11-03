# MULTI-OBJECTIVE OPTIMIZATION & ADAPTIVE WEIGHT METHODOLOGY
## Formulation, Pareto Analysis, Dirichlet Thompson Sampling, and Energy Proxies

---

## 1. Problem Formulation

In heterogeneous Cloud HPC nodes, optimizing solely for speed (makespan) often leads to severe power spikes, high electricity costs, and PCIe bus contention. Conversely, optimizing solely for energy may violate Service Level Agreements (SLAs) by leaving discrete accelerators idle.

We formalize workload placement as a **Constrained Multi-Objective Optimization Problem**:

Given a set of $N$ dependent or independent workload tasks $\{T_1, T_2, \dots, T_N\}$, find an assignment vector $\mathbf{x} = [x_1, x_2, \dots, x_N]$ where $x_i \in \{\text{CPU}, \text{GPU}\}$ that simultaneously minimizes three conflicting objectives:

$$\min_{\mathbf{x}} \mathbf{F}(\mathbf{x}) = \Big[ f_1(\mathbf{x}), f_2(\mathbf{x}), f_3(\mathbf{x}) \Big]^T$$

---

## 2. Objective Function Definitions

### 2.1 Objective 1: Workflow Makespan ($f_1$)
Makespan represents the total elapsed time from workflow initiation to completion across parallel CPU slots ($S_{\text{CPU}} = 4$) and dedicated GPU slots ($S_{\text{GPU}} = 1$):

$$f_1(\mathbf{x}) = \max \Big( \text{Makespan}(\mathcal{Q}_{\text{CPU}}, S_{\text{CPU}}), \; \text{Makespan}(\mathcal{Q}_{\text{GPU}}, S_{\text{GPU}}) \Big)$$

where task durations are derived from the pre-execution runtime estimator:

$$\text{Makespan}(\mathcal{Q}, S) = \max_{j \in \{1 \dots S\}} \sum_{T_i \in \text{Slot}_j} \text{Duration}(T_i, x_i)$$

### 2.2 Objective 2: Total Energy Consumption ($f_2$)
Total energy is modeled using calibrated Thermal Design Power (TDP) proxies and active execution time:

$$f_2(\mathbf{x}) = \sum_{i=1}^N \Big[ \text{Runtime}(T_i, x_i) \cdot P_{\text{TDP}}(x_i) \cdot \eta_i + \text{Interconnect\_Energy}(T_i, x_i) \Big]$$

where:
- $P_{\text{TDP}}(\text{CPU}) = 65.0\text{ W}$ (Generic Server CPU Socket)
- $P_{\text{TDP}}(\text{GPU}) = 70.0\text{ W}$ (NVIDIA Tesla T4 Accelerator)
- $\eta_i \in [0.80, 0.95]$ is the active core utilization factor.

### 2.3 Objective 3: Host-Device Interconnect Transfer Overhead ($f_3$)
Measures the cumulative time spent saturating the PCIe interconnect:

$$f_3(\mathbf{x}) = \sum_{i=1}^N \mathbb{I}(x_i = \text{GPU}) \cdot \frac{\text{Transfer\_Size\_MB}(T_i)}{B_{\text{PCIe}}}$$

where $B_{\text{PCIe}} = 12,000.0\text{ MB/s}$ (PCIe 3.0 x16 practical bandwidth).

---

## 3. Optimization Algorithms

### 3.1 Strategy 1: Weighted-Sum Scalarization
Scalarizes the 3 objectives using normalized adaptive weights $\mathbf{w} = [w_1, w_2, w_3]$:

$$\mathcal{S}(\mathbf{x}, \mathbf{w}) = w_1 \cdot f_1(\mathbf{x}) + w_2 \cdot \left(\frac{f_2(\mathbf{x})}{10,000}\right) + w_3 \cdot f_3(\mathbf{x})$$

subject to:
$$w_1 + w_2 + w_3 = 1, \quad w_i \ge 0$$

- For $N \le 20$ tasks: **Exhaustive search** over all $2^N$ assignment combinations guarantees the global minimum.
- For $N > 20$ tasks: **Greedy marginal cost minimization** assigns each task to the device minimizing marginal scalarized cost.

### 3.2 Strategy 2: NSGA-II Multi-Objective Evolutionary Optimization
When the user requests the full Pareto frontier, the framework invokes the Non-dominated Sorting Genetic Algorithm II (NSGA-II) via DEAP:
- Population size: 50 individuals
- Generations: 100
- Crossover: Uniform crossover ($p_c = 0.70$)
- Mutation: Bit-flip mutation ($p_m = 1/N$)
- Selection: Non-dominated sorting with crowding distance metric.

A solution $\mathbf{x}_A$ dominates $\mathbf{x}_B$ ($\mathbf{x}_A \prec \mathbf{x}_B$) if:

$$\forall k \in \{1, 2, 3\}, \; f_k(\mathbf{x}_A) \le f_k(\mathbf{x}_B) \quad \land \quad \exists k \in \{1, 2, 3\}, \; f_k(\mathbf{x}_A) < f_k(\mathbf{x}_B)$$

---

## 4. Bayesian Online Weight Adaptation (Thompson Sampling)

Rather than fixing static objective weights upfront, the scheduler dynamically adapts priorities using **Bayesian Online Learning**:

### 4.1 Dirichlet Distribution Model
The weight vector $\mathbf{w} \sim \text{Dirichlet}(\boldsymbol{\alpha})$ where $\boldsymbol{\alpha} = [\alpha_1, \alpha_2, \alpha_3]$.

1. **Weight Sampling (Thompson Sampling)**:
   $$\mathbf{w}^{(t)} \sim \text{Dirichlet}(\alpha_1^{(t)}, \alpha_2^{(t)}, \alpha_3^{(t)})$$
   Naturally balances exploration and exploitation while guaranteeing $\sum w_i = 1$ and $w_i \ge 0$.

2. **Observed Cost Feedback**:
   Upon task completion, observed metrics $[c_1, c_2, c_3]$ (runtime, energy, transfer time) are normalized:
   $$\hat{c}_k = \frac{c_k}{\sum_{j=1}^3 c_j}$$

3. **Posterior Parameter Reinforcement Update**:
   $$\alpha_k^{(t+1)} = \alpha_k^{(t)} + \lambda \cdot \hat{c}_k \cdot 3$$
   where $\lambda = 0.3$ is the learning rate. High costs in an objective increase its concentration parameter $\alpha_k$, elevating its scheduling priority in subsequent rounds.

4. **System Pressure Adjustments**:
   - Queue length $> 10 \implies$ increase $\alpha_1$ (Makespan) by 5% per task.
   - GPU memory usage $> 85\% \implies$ increase $\alpha_3$ (Transfer / PCIe avoidance) by $1.5\times$.
   - Imminent deadline proximity $\implies$ double $\alpha_1$.

