# CG + RL for Integrated RCPSP + Procurement

## Problem
Schedule tasks with precedence and resource constraints while simultaneously optimizing part procurement (lead times, setup costs, holding costs). Monolithic MIP doesn't scale.

## Architecture
```
LP Master (exact)  ←→  RL Pricing (learned)
   ↓                        ↓
Solves convex combo     GNN policy generates
of schedule columns,    new schedule given
returns duals α,μ       dual prices as input
```

## How it works
1. **LP Master** (keep exact): convex combination of schedule columns, minimizes procurement cost, returns dual prices α_{m,t} (shadow prices on early-demand constraints) and μ (convexity dual)
2. **RL Pricing** (replace CP-SAT): trained policy takes duals as extra input features, generates a schedule with negative reduced cost. Same GNN architecture as the litho paper but with dual-price-augmented state
3. **Loop**: Master → get duals → RL generates column → add to pool → re-solve master → repeat until no improving column

## Why CG + RL
- **Builds on litho paper**: same RL-for-scheduling core, reuses GNN architecture
- **Mathematically grounded**: LP duality provides principled training signal (reduced cost = reward)
- **Scalable**: RL pricing runs in milliseconds vs CP-SAT seconds/minutes
- **Publishable**: "Learning to Price" for scheduling is a clean, novel contribution
- **Dual prices as features**: the RL policy naturally adapts to procurement constraints without encoding them explicitly

## Key design decisions
- **State**: task/resource node features + dual prices α_{m,t} appended to part-related features + μ in global context
- **Reward**: negative reduced cost -(cost_k - Σ_{m,t} α_{m,t} D^k_{m,t} - μ)
- **Training**: generate diverse columns via HA-POMO style anchoring, train with REINFORCE against LP dual signals
- **Fallback**: if RL fails to find negative reduced cost column, run CP-SAT pricing once as safety net

## Existing code to reuse
- `ColumnGenerationHeuristic.py`: LP master (`solve_master_lp`), procurement cost computation (`plan_orders_greedy`), CG loop structure
- Litho project: GNN policy (`DRCSchedulingGNN`), HA-POMO trainer, batched episode runner

## Paper angle
"Learning to Price: Graph RL for Column Generation in Integrated Scheduling and Inventory Optimization" -- combines OR (CG) with ML (GNN+RL), shows RL-generated columns match or beat CP-SAT columns at 100x speed.

## Implementation steps
1. Define RCPSP+Procurement MDP environment (similar to `DRCSchedulingEnvGNN`)
2. Add dual price features to state representation
3. Build GNN policy for task dispatching with dual-augmented features
4. Wire RL pricing into existing CG loop (replace `schedule_with_cpsat` call)
5. Train on small instances, test generalization to large
6. Benchmark: CG+RL vs CG+CP-SAT vs monolithic MIP
