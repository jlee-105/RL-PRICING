## Mathematical Formulations: Integrated MIP, Logic-Based Benders, Column Generation

### Notation (shared)
- **Sets**
  - Tasks: J; Renewable resources: R; Parts: M; Time periods: T = {0,…,H}
- **Parameters**
  - p_j: duration of task j
  - k_{jr}: units of renewable r used by task j when active
  - a_{jm}: units of part m consumed at the start of task j
  - K_r: capacity of renewable r
  - L_m: lead time of part m
  - I0_m: initial inventory of part m at t = 0
  - c_m, f_m, h_m: unit, setup, holding cost for part m
- **Derived demand from starts**
  - D_m[t] = Σ_j a_{jm} z_{j,t}
- **Decision variables (used across models)**
  - z_{j,t} ∈ {0,1}: 1 if task j starts at time t
  - y_{j,t} ∈ {0,1}: 1 if task j is active at time t
  - s_j ∈ Z_+: start time of task j
  - C_max ∈ Z_+: makespan
  - x_{m,t} ∈ Z_+: order quantity of part m at time t
  - δ_{m,t} ∈ {0,1}: setup indicator of an order for part m at time t
  - inv_{m,t} ∈ Z_+: end-of-period inventory of part m at time t

## Original Integrated MIP (time-indexed, consumption at start)

### Objective (example: procurement cost + optional makespan weight λ)
\[
\min \sum_{m \in M}\sum_{t \in T}\big(c_m x_{m,t} + f_m \, \delta_{m,t} + h_m\,inv_{m,t}\big)\;+\;\lambda\,C_{\max}
\]

### Task starts and activity span
\[
\sum_{t \in T} z_{j,t} = 1 \quad \forall j\in J
\]
\[
s_j = \sum_{t \in T} t\,z_{j,t} \quad \forall j\in J
\]
\[
y_{j,t} = \sum_{\tau \in T:\, \tau \le t < \tau + p_j} z_{j,\tau} \quad \forall j\in J,\; t\in T
\]

### Precedence
\[
s_j \;\ge\; s_i + p_i \quad \forall (i\to j)
\]

### Renewable capacities
\[
\sum_{j \in J} k_{jr}\, y_{j,t} \;\le\; K_r \quad \forall r\in R,\; t\in T
\]

### Makespan
\[
C_{\max} \;\ge\; s_j + p_j \quad \forall j\in J
\]

### Material demand at starts
\[
D_m[t] \;=\; \sum_{j\in J} a_{jm}\, z_{j,t} \quad \forall m\in M,\; t\in T
\]

### Inventory dynamics with lead time
\[
inv_{m,t} \;=\; inv_{m,t-1} + x_{m,t-L_m} - D_m[t] \quad \forall m\in M,\; t \in T
\]
Use base cases for t < 0 and t < L_m: set x_{m,\tau} = 0 for \(\tau<0\), and initialize \(inv_{m,-1} = I0_m\).

### Setup linking (one possible linearization)
\[
x_{m,t} \;\le\; U_m\, \delta_{m,t},\quad \delta_{m,t}\in\{0,1\},\; x_{m,t}\ge 0
\]
with a valid big-U bound \(U_m\).

---

## Logic-Based Benders Decomposition

### Master (Scheduling only; solved via CP/CP-SAT)
- Variables: \(s_j\) (and intervals), \(C_{\max}\)
- Constraints: precedence, cumulative renewable capacities, makespan
- Objective: minimize \(C_{\max}\) (or any scheduling objective)

### Subproblem (Procurement feasibility/cost)
- Given starts \(s_j\), compute \(D_m[t]\) and solve procurement (feasible? cost?).

### Cuts (iterative constraint generation)
- Practical logical delay cut used in code: if part m is infeasible at time \(t^*\), pick a culprit task j consuming m with \(s_j \le t^*\) and add
\[
s_{j} \;\ge\; t^* + 1
\]
- Stronger feasibility family (not used directly): for any \(t\),
\[
\sum_{t' \le t} D_m[t'] \;\le\; I0_m + \sum_{\tau \le t - L_m} x_{m,\tau}
\]
Iterate: solve master → subproblem → add cut if infeasible → repeat until feasible.

---

## Column Generation (restricted master + pricing)

### Columns
Each column k represents a full schedule pattern (starts \(s^k\)) with induced demands \(D_m^k[t]\) and an associated cost \(\mathrm{cost}_k\) (procurement cost, optionally plus weighted makespan).

### Restricted Master Problem (RMP)
- Decision vars: \(\lambda_k \ge 0\) for k in pool \(\mathcal{K}\)
- Convexity:
\[
\sum_{k\in \mathcal{K}} \lambda_k = 1
\]
- Early-demand protection (as in code) with optional slack \(s_{m,t}\ge 0\) and penalty M:
\[
\sum_{k\in \mathcal{K}} \lambda_k\, D_m^k[t] \;\le\; I0_m + s_{m,t}\quad \forall m\in M,\; t < L_m
\]
- Objective:
\[
\min \sum_{k\in \mathcal{K}} \lambda_k\, \mathrm{cost}_k \;+\; M \sum_{m,t} s_{m,t}
\]

### Duals of RMP
- Let \(\mu\) be the dual of convexity; \(\alpha_{m,t} \ge 0\) be duals of the early-demand constraints.

### Pricing subproblem
Find a new schedule (column) k minimizing reduced cost:
\[
\mathrm{rc}(k) \;=\; \mathrm{cost}_k \;-\; \sum_{m,t} \alpha_{m,t}\, D_m^k[t] \;-\; \mu
\]
Heuristic generation (as implemented): solve CP-SAT with a modified objective that rewards placing starts at times with large \(\alpha_{m,t}\), then compute \(\mathrm{cost}_k\) via procurement for that schedule. If \(\mathrm{rc}(k) < -\varepsilon\), add k to \(\mathcal{K}\) and re-solve the master; otherwise, stop.



