"""
Integrated RCPSP + Procurement (Lead-Time) — Extended Open-Source Demo
---------------------------------------------------------------------
Goal: Show a *scalable* alternating approach that avoids a huge monolithic MILP.
Includes:
  1) Cost objective (purchasing, holding, order setup).
  2) Greedy JIT procurement (per-item, with lead times) — easy to swap for a MILP.
  3) Logging of intermediate solutions for reproducibility.
  4) Initial-inventory usage reporting at task start times.

Tested with:
  - Python 3.10+
  - ortools >= 9.9  (pip install ortools)

Author: Your Name (JJ + ChatGPT scaffold)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

try:
    from ortools.sat.python import cp_model
except Exception as e:
    raise SystemExit(
        "OR-Tools is required. Please install with: pip install ortools\n"
        f"Original import error: {e}"
    )

# ------------------------------
# Data structures
# ------------------------------

@dataclass
class Activity:
    id: str
    duration: int
    demand_renewable: Dict[str, int]  # k_{jr}
    demand_parts: Dict[str, int]      # a_{jm} (consumed at START)

@dataclass
class Instance:
    activities: Dict[str, Activity]
    precedence: List[Tuple[str, str]]     # (i -> j)
    renewable_capacity: Dict[str, int]    # K_r (constant over time in this demo)
    parts: List[str]
    lead_time: Dict[str, int]             # L_m
    init_inventory: Dict[str, int]        # I0_m
    horizon: int
    unit_cost: Dict[str, float]           # c_m
    setup_cost: Dict[str, float]          # f_m
    holding_cost: Dict[str, float]        # h_m

# ------------------------------
# Scheduling with CP-SAT
# ------------------------------

def schedule_with_cpsat(inst: Instance,
                        earliest_start_lb: Optional[Dict[str, int]] = None,
                        time_limit_s: int = 10) -> Tuple[Dict[str, int], int]:
    """Solve RCPSP (renewable + precedence) minimizing makespan."""
    model = cp_model.CpModel()
    tasks = inst.activities
    H = inst.horizon

    # Variables: starts and intervals
    s: Dict[str, cp_model.IntVar] = {}
    intervals: Dict[str, cp_model.IntervalVar] = {}

    for j, act in tasks.items():
        latest_start = H - act.duration
        if latest_start < 0:
            raise RuntimeError("Horizon too short; an activity doesn't fit.")
        lb = earliest_start_lb.get(j, 0) if earliest_start_lb else 0
        s[j] = model.NewIntVar(lb, latest_start, f"start_{j}")
        end_j = model.NewIntVar(lb + act.duration, H, f"end_{j}")
        intervals[j] = model.NewIntervalVar(s[j], act.duration, end_j, f"iv_{j}")

    # Precedence
    for i, j in inst.precedence:
        model.Add(s[j] >= s[i] + tasks[i].duration)

    # Renewable capacities (cumulative)
    for r, cap in inst.renewable_capacity.items():
        model.AddCumulative(
            [intervals[j] for j in tasks],
            [tasks[j].demand_renewable.get(r, 0) for j in tasks],
            cap
        )

    # Makespan
    Cmax = model.NewIntVar(0, inst.horizon, "Cmax")
    for j, act in tasks.items():
        model.Add(Cmax >= s[j] + act.duration)
    model.Minimize(Cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Scheduling infeasible (or no solution within time limit).")

    starts = {j: int(solver.Value(s[j])) for j in tasks}
    makespan = int(solver.Value(Cmax))
    return starts, makespan

# ------------------------------
# Procurement: greedy JIT (baseline)
# ------------------------------

def derive_demands_by_time(inst: Instance, starts: Dict[str, int]) -> Dict[str, List[int]]:
    """D_m[t] = total units of part m consumed by tasks starting at t."""
    H = inst.horizon
    D: Dict[str, List[int]] = {m: [0]*(H+1) for m in inst.parts}
    for j, act in inst.activities.items():
        t = starts[j]
        for m, amt in act.demand_parts.items():
            if amt:
                D[m][t] += amt
    return D

@dataclass
class ProcurementPlan:
    feasible: bool
    orders: Dict[str, List[int]]      # q_m[order_time]
    arrivals: Dict[str, List[int]]    # arrivals at time t
    inventory: Dict[str, List[int]]   # end-of-period inventory
    cost: float = 0.0
    first_violation: Optional[Tuple[str, int]] = None
    # we'll attach: initial_used: Dict[str, List[int]] dynamically for reporting

def plan_orders_greedy(inst: Instance, D: Dict[str, List[int]]) -> ProcurementPlan:
    """
    Greedy JIT procurement per part (no capacity on orders; with unit/setup/holding costs).
    - Uses initial inventory first.
    - If demand at t cannot be covered and t - L_m < 0 -> violation (cannot arrive in time).
    """
    H = inst.horizon
    orders = {m: [0]*(H+1) for m in inst.parts}
    arrivals = {m: [0]*(H+1) for m in inst.parts}
    inv = {m: [0]*(H+1) for m in inst.parts}
    initial_used = {m: [0]*(H+1) for m in inst.parts}
    total_cost = 0.0

    for m in inst.parts:
        I = inst.init_inventory.get(m, 0)
        L = inst.lead_time[m]
        first_bad = None
        for t in range(H+1):
            # arrivals for t (from orders placed at t-L)
            if t - L >= 0:
                arrivals[m][t] += orders[m][t - L]

            # available and demand
            available_before = I
            available = I + arrivals[m][t]
            demand_t = D[m][t]

            # initial stock coverage (for reporting)
            use_init = min(available_before, demand_t)
            initial_used[m][t] = use_init

            if demand_t <= available:
                available -= demand_t
                I = available
            else:
                shortage = demand_t - available
                order_time = t - L
                if order_time < 0:
                    # impossible to arrive in time
                    first_bad = t if first_bad is None else first_bad
                    I = 0
                else:
                    orders[m][order_time] += shortage
                    arrivals[m][t] += shortage
                    I = 0
                    total_cost += shortage * inst.unit_cost[m] + inst.setup_cost[m]

            # holding cost on end-of-period inventory
            total_cost += I * inst.holding_cost[m]
            inv[m][t] = I

        if first_bad is not None:
            plan = ProcurementPlan(False, orders, arrivals, inv, total_cost, (m, first_bad))
            plan.initial_used = initial_used  # type: ignore[attr-defined]
            return plan

    plan = ProcurementPlan(True, orders, arrivals, inv, total_cost)
    plan.initial_used = initial_used  # type: ignore[attr-defined]
    return plan

# ------------------------------
# Prefix-cut enforcement (choose a culprit task to delay)
# ------------------------------

def choose_task_to_delay(inst: Instance,
                         starts: Dict[str, int],
                         violating_part: str,
                         t_star: int) -> Optional[str]:
    """Pick a task using violating_part that starts at/before t_star; delay it."""
    candidates = [
        j for j, act in inst.activities.items()
        if act.demand_parts.get(violating_part, 0) > 0 and starts[j] <= t_star
    ]
    return max(candidates, key=lambda j: starts[j]) if candidates else None

# ------------------------------
# Logging
# ------------------------------

def log_solution(iteration: int,
                 starts: Dict[str, int],
                 plan: ProcurementPlan,
                 filename: str = "solutions_log.json"):
    entry = {
        "iteration": iteration,
        "starts": starts,
        "cost": plan.cost,
        "feasible": plan.feasible,
        "violation": plan.first_violation
    }
    try:
        with open(filename, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging failure shouldn't crash the run

# ------------------------------
# Demo instance
# ------------------------------

def build_toy_instance() -> Instance:
    acts = {
        "A": Activity("A", 4, {"R1": 2}, {"M1": 1}),
        "B": Activity("B", 3, {"R1": 1}, {"M2": 2}),
        "C": Activity("C", 5, {"R1": 2}, {"M2": 1}),
        "D": Activity("D", 2, {"R1": 1}, {}),
        "E": Activity("E", 4, {"R1": 2}, {"M1": 1}),
    }
    precedence = [("A", "C"), ("B", "D"), ("C", "E"), ("D", "E")]
    parts = ["M1", "M2"]
    lead_time = {"M1": 3, "M2": 2}
    init_inventory = {"M1": 1, "M2": 0}
    unit_cost = {"M1": 5.0, "M2": 7.0}
    setup_cost = {"M1": 2.0, "M2": 2.0}
    holding_cost = {"M1": 0.5, "M2": 0.5}
    return Instance(
        activities=acts,
        precedence=precedence,
        renewable_capacity={"R1": 3},
        parts=parts,
        lead_time=lead_time,
        init_inventory=init_inventory,
        horizon=30,
        unit_cost=unit_cost,
        setup_cost=setup_cost,
        holding_cost=holding_cost,
    )

# ------------------------------
# Alternating algorithm with logging
# ------------------------------

def solve_alternating(inst: Instance,
                      max_iters: int = 20,
                      sched_time_limit_s: int = 5):
    earliest_lb: Dict[str, int] = {}

    for it in range(1, max_iters + 1):
        print(f"\n=== ITERATION {it} ===")
        # SCHEDULING (respect current earliest-start lower bounds)
        starts, Cmax = schedule_with_cpsat(inst, earliest_lb, sched_time_limit_s)
        print("Schedule:", starts, "Makespan=", Cmax)

        # DEMANDS (from realized starts)
        D = derive_demands_by_time(inst, starts)

        # PROCUREMENT (greedy JIT + costs)
        plan = plan_orders_greedy(inst, D)

        # REPORT which tasks used initial stock at their start time
        print("Initial-stock coverage (task -> {part: used_from_I0}):")
        coverage: Dict[str, Dict[str, int]] = {}
        for j, t in starts.items():
            usage = {}
            for m in inst.parts:
                used_series = getattr(plan, "initial_used", {}).get(m, None)
                if used_series is not None and 0 <= t <= inst.horizon:
                    used_at_t = used_series[t]
                    if used_at_t > 0 and inst.activities[j].demand_parts.get(m, 0) > 0:
                        # only report up to the task's actual demand for neatness
                        usage[m] = min(used_at_t, inst.activities[j].demand_parts[m])
            if usage:
                coverage[j] = usage
        for j in sorted(coverage, key=lambda x: starts[x]):
            print(f"  {j}: {coverage[j]}")

        print("Cost=", plan.cost, "Feasible=", plan.feasible)
        log_solution(it, starts, plan)

        if plan.feasible:
            print("Feasible integrated plan found! ✅")
            return {
                "starts": starts,
                "makespan": Cmax,
                "orders": plan.orders,
                "cost": plan.cost
            }
        else:
            # Add a prefix-cut–style delay to one culprit task and iterate
            part, t_bad = plan.first_violation  # type: ignore[assignment]
            culprit = choose_task_to_delay(inst, starts, part, t_bad)
            if culprit is None:
                print("No suitable culprit task found to delay; stopping.")
                return None
            earliest_lb[culprit] = max(earliest_lb.get(culprit, 0), t_bad + 1)
            print(f"Adding cut: delay task {culprit} >= {earliest_lb[culprit]}")

    print("Reached iteration cap without feasibility.")
    return None

# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    inst = build_toy_instance()
    result = solve_alternating(inst, max_iters=10, sched_time_limit_s=3)
    if result:
        print("\n=== FINAL SOLUTION ===")
        print(result)
    else:
        print("No feasible plan found.")