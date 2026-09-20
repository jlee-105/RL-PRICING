"""
Column-Generation style heuristic for integrated Scheduling + Procurement.

Master (LP): convex combination of candidate schedules with early-demand constraints
Pricing (Heuristic CP-SAT): generates a schedule guided by dual prices on early usage

This script is self-contained; it does not import other project files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
import json
import csv

try:
    from ortools.sat.python import cp_model
    from ortools.linear_solver import pywraplp
except Exception as e:
    raise SystemExit(
        "OR-Tools is required. Please install with: pip install ortools\n"
        f"Original import error: {e}"
    )


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class Mode:
    duration: int
    demand_renewable: Dict[str, int]  # k_{jrq}
    demand_parts: Dict[str, int]      # a_{jmq} (consumed at START)
    cost: float = 0.0                 # gamma_{jq}


@dataclass
class Activity:
    id: str
    modes: List[Mode]

    @property
    def duration(self) -> int:
        return self.modes[0].duration

    @property
    def demand_renewable(self) -> Dict[str, int]:
        return self.modes[0].demand_renewable

    @property
    def demand_parts(self) -> Dict[str, int]:
        return self.modes[0].demand_parts

    @property
    def num_modes(self) -> int:
        return len(self.modes)


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
# Helpers
# ------------------------------

def derive_demands_by_time(inst: Instance, starts: Dict[str, int],
                           mode_assignments: Optional[Dict[str, int]] = None) -> Dict[str, List[int]]:
    H = inst.horizon
    D: Dict[str, List[int]] = {m: [0] * (H + 1) for m in inst.parts}
    for j, act in inst.activities.items():
        t = starts[j]
        q = mode_assignments[j] if mode_assignments else 0
        mode = act.modes[q]
        for m, amt in mode.demand_parts.items():
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


def plan_orders_greedy(inst: Instance, D: Dict[str, List[int]]) -> ProcurementPlan:
    """Per-part greedy JIT: uses init inventory first, orders just-in-time with setup+unit costs."""
    H = inst.horizon
    orders = {m: [0] * (H + 1) for m in inst.parts}
    arrivals = {m: [0] * (H + 1) for m in inst.parts}
    inv = {m: [0] * (H + 1) for m in inst.parts}
    total_cost = 0.0
    first_violation: Optional[Tuple[str, int]] = None

    for m in inst.parts:
        I = inst.init_inventory.get(m, 0)
        L = inst.lead_time[m]
        first_bad = None
        for t in range(H + 1):
            if t - L >= 0:
                arrivals[m][t] += orders[m][t - L]

            available = I + arrivals[m][t]
            demand_t = D[m][t]

            if demand_t <= available:
                I = available - demand_t
            else:
                shortage = demand_t - available
                order_time = t - L
                if order_time < 0:
                    first_bad = t if first_bad is None else first_bad
                    I = 0
                else:
                    orders[m][order_time] += shortage
                    arrivals[m][t] += shortage
                    I = 0
                    total_cost += shortage * inst.unit_cost[m] + inst.setup_cost[m]

            total_cost += I * inst.holding_cost[m]
            inv[m][t] = I

        # Early (t < L_m) shortages are priced by the master's slack, not here. Keep
        # costing the remaining parts: returning early used to drop their cost entirely.
        if first_bad is not None and first_violation is None:
            first_violation = (m, first_bad)

    return ProcurementPlan(first_violation is None, orders, arrivals, inv, total_cost, first_violation)


# ------------------------------
# Scheduling with CP-SAT (pricing variant supports dual-based penalties)
# ------------------------------

def schedule_with_cpsat(inst: Instance,
                        penalty_alpha: Optional[Dict[Tuple[str, int], float]] = None,
                        time_limit_s: int = 5) -> Tuple[Dict[str, int], Dict[str, int], int, float]:
    """
    Solve MRCPSP minimizing makespan + mode costs minus dual-weighted early consumption penalties.
    Returns: (starts, mode_assignments, makespan, total_alpha_reward)
    """
    model = cp_model.CpModel()

    tasks = inst.activities
    H = inst.horizon
    starts: Dict[str, cp_model.IntVar] = {}
    mode_vars: Dict[str, cp_model.IntVar] = {}
    durations: Dict[str, cp_model.IntVar] = {}

    for j, act in tasks.items():
        num_modes = act.num_modes
        min_dur = min(mode.duration for mode in act.modes)
        max_dur = max(mode.duration for mode in act.modes)
        latest_start = H - min_dur
        if latest_start < 0:
            raise RuntimeError("Horizon too short; an activity doesn't fit.")

        s_j = model.NewIntVar(0, latest_start, f"start_{j}")
        starts[j] = s_j

        if num_modes > 1:
            q_j = model.NewIntVar(0, num_modes - 1, f"mode_{j}")
            mode_vars[j] = q_j
            dur_j = model.NewIntVar(min_dur, max_dur, f"dur_{j}")
            dur_list = [mode.duration for mode in act.modes]
            model.AddElement(q_j, dur_list, dur_j)
            durations[j] = dur_j
        else:
            mode_vars[j] = model.NewConstant(0)
            durations[j] = model.NewConstant(act.duration)

    # Precedence
    for i, j in inst.precedence:
        model.Add(starts[j] >= starts[i] + durations[i])

    # Renewable capacities — use optional intervals per mode
    for r, cap in inst.renewable_capacity.items():
        all_intervals = []
        all_demands = []
        for j, act in tasks.items():
            if act.num_modes == 1:
                e_j = model.NewIntVar(0, H, f"end_{j}_{r}")
                model.Add(e_j == starts[j] + durations[j])
                iv = model.NewIntervalVar(starts[j], durations[j], e_j, f"iv_{j}_{r}")
                all_intervals.append(iv)
                all_demands.append(act.modes[0].demand_renewable.get(r, 0))
            else:
                for qi, mode in enumerate(act.modes):
                    is_mode = model.NewBoolVar(f"is_{j}_mode{qi}")
                    model.Add(mode_vars[j] == qi).OnlyEnforceIf(is_mode)
                    model.Add(mode_vars[j] != qi).OnlyEnforceIf(is_mode.Not())
                    e_jq = model.NewIntVar(0, H, f"end_{j}_m{qi}_{r}")
                    model.Add(e_jq == starts[j] + mode.duration).OnlyEnforceIf(is_mode)
                    iv_jq = model.NewOptionalIntervalVar(
                        starts[j], mode.duration, e_jq, is_mode, f"iv_{j}_m{qi}_{r}"
                    )
                    all_intervals.append(iv_jq)
                    all_demands.append(mode.demand_renewable.get(r, 0))
        model.AddCumulative(all_intervals, all_demands, cap)

    # Makespan
    Cmax = model.NewIntVar(0, H, "Cmax")
    for j in tasks:
        model.Add(Cmax >= starts[j] + durations[j])

    # Mode costs (scaled to integer)
    mode_cost_vars = []
    for j, act in tasks.items():
        if act.num_modes > 1:
            costs = [int(round(mode.cost * 1000)) for mode in act.modes]
            mc_j = model.NewIntVar(min(costs), max(costs), f"mode_cost_{j}")
            model.AddElement(mode_vars[j], costs, mc_j)
            mode_cost_vars.append(mc_j)
        else:
            if act.modes[0].cost > 0:
                mode_cost_vars.append(model.NewConstant(int(round(act.modes[0].cost * 1000))))

    # Dual-based per-task start penalties using Element
    alpha_reward_vars: List[cp_model.IntVar] = []
    if penalty_alpha is not None:
        for j, act in tasks.items():
            if act.num_modes == 1:
                mode = act.modes[0]
                rewards: List[int] = []
                for t in range(H + 1):
                    reward_t = 0.0
                    for m, a_jm in mode.demand_parts.items():
                        if a_jm:
                            reward_t += penalty_alpha.get((m, t), 0.0) * a_jm
                    rewards.append(int(round(reward_t * 1000)))
                r_j = model.NewIntVar(min(rewards), max(rewards), f"alpha_reward_{j}")
                model.AddElement(starts[j], rewards, r_j)
                alpha_reward_vars.append(r_j)
            else:
                for qi, mode in enumerate(act.modes):
                    rewards_q: List[int] = []
                    for t in range(H + 1):
                        reward_t = 0.0
                        for m, a_jm in mode.demand_parts.items():
                            if a_jm:
                                reward_t += penalty_alpha.get((m, t), 0.0) * a_jm
                        rewards_q.append(int(round(reward_t * 1000)))
                    is_mode = model.NewBoolVar(f"is_{j}_m{qi}_alpha")
                    model.Add(mode_vars[j] == qi).OnlyEnforceIf(is_mode)
                    model.Add(mode_vars[j] != qi).OnlyEnforceIf(is_mode.Not())
                    r_jq = model.NewIntVar(min(rewards_q), max(rewards_q), f"alpha_reward_{j}_m{qi}")
                    model.AddElement(starts[j], rewards_q, r_jq)
                    contrib = model.NewIntVar(min(rewards_q), max(rewards_q), f"alpha_contrib_{j}_m{qi}")
                    model.Add(contrib == r_jq).OnlyEnforceIf(is_mode)
                    model.Add(contrib == 0).OnlyEnforceIf(is_mode.Not())
                    alpha_reward_vars.append(contrib)

    # Objective: minimize Cmax + mode_costs - sum(alpha_rewards)
    obj_terms = [Cmax]
    if mode_cost_vars:
        obj_terms.extend(mode_cost_vars)
    if alpha_reward_vars:
        obj_terms.append(-sum(alpha_reward_vars))
    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Pricing CP-SAT failed to find a schedule.")

    starts_sol = {j: int(solver.Value(starts[j])) for j in tasks}
    modes_sol = {j: int(solver.Value(mode_vars[j])) for j in tasks}
    cmax_val = int(solver.Value(Cmax))
    alpha_reward_sum = 0.0
    if alpha_reward_vars:
        alpha_reward_sum = sum(solver.Value(v) for v in alpha_reward_vars) / 1000.0
    return starts_sol, modes_sol, cmax_val, alpha_reward_sum


# ------------------------------
# Master LP over schedule pool
# ------------------------------

def solve_master_lp(inst: Instance,
                    pool_D: List[Dict[str, List[int]]],
                    pool_cost: List[float],
                    early_only: bool = True) -> Tuple[List[float], Dict[Tuple[str, int], float], float, float, float]:
    """
    Solve restricted master problem:
      min sum_k cost_k * lambda_k
      s.t. sum_k lambda_k = 1
           For (m,t in early window): sum_k lambda_k * D^k_{m,t} <= I0_m
    Returns: (lambda, alpha_duals, mu_dual)
    """
    n = len(pool_D)
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        raise RuntimeError("Failed to create LP solver (GLOP).")

    lam = [solver.NumVar(0.0, solver.infinity(), f"lam_{k}") for k in range(n)]

    # Convexity
    conv = solver.Add(sum(lam) == 1.0)

    # Early constraints (with slack to maintain feasibility)
    alpha_duals: Dict[Tuple[str, int], float] = {}
    cons_map: Dict[Tuple[str, int], pywraplp.Constraint] = {}
    slack_vars = {}
    for m in inst.parts:
        I0 = inst.init_inventory.get(m, 0)
        t_max = inst.lead_time[m] - 1 if early_only else inst.horizon
        for t in range(max(t_max + 1, 0)):
            s_mt = solver.NumVar(0.0, solver.infinity(), f"slack_{m}_{t}")
            slack_vars[(m, t)] = s_mt
            ct = solver.Add(sum(lam[k] * pool_D[k][m][t] for k in range(n)) <= float(I0) + s_mt)
            cons_map[(m, t)] = ct

    # Objective
    solver.Minimize(
        sum(pool_cost[k] * lam[k] for k in range(n))
        + PENALTY_SLACK * sum(slack_vars.values())
    )

    result = solver.Solve()
    if result != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Master LP did not solve to optimality.")

    lam_val = [lam[k].solution_value() for k in range(n)]
    mu = conv.dual_value()
    for key, ct in cons_map.items():
        alpha_duals[key] = ct.dual_value()
    # Objective tracing
    obj_val = solver.Objective().Value()
    slack_sum = 0.0
    # Recover slack vars by scanning variables named with prefix "slack_"
    for v in solver.variables():
        name = v.name()
        if name.startswith("slack_"):
            slack_sum += v.solution_value()
    return lam_val, alpha_duals, mu, obj_val, slack_sum


# ------------------------------
# Demo instance
# ------------------------------

def build_toy_instance() -> Instance:
    acts = {
        "A": Activity("A", [Mode(4, {"R1": 2}, {"M1": 1})]),
        "B": Activity("B", [Mode(3, {"R1": 1}, {"M2": 2})]),
        "C": Activity("C", [Mode(5, {"R1": 2}, {"M2": 1})]),
        "D": Activity("D", [Mode(2, {"R1": 1}, {})]),
        "E": Activity("E", [Mode(4, {"R1": 2}, {"M1": 1})]),
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
# Column generation loop (heuristic pricing)
# ------------------------------

# Configuration
MAX_PRICING_ITERS = 10
SCHED_TIME_LIMIT_S = 3
MAKESPAN_WEIGHT = 0.0  # keep cost as procurement only for clarity
IMPROVEMENT_TOL = 1e-6
PENALTY_SLACK = 1000.0  # penalty per unit of early unmet demand in the master


def schedule_cost(inst: Instance, starts: Dict[str, int],
                   mode_assignments: Optional[Dict[str, int]] = None) -> Tuple[float, int, Dict[str, List[int]]]:
    if mode_assignments is None:
        mode_assignments = {j: 0 for j in inst.activities}
    D = derive_demands_by_time(inst, starts, mode_assignments)
    plan = plan_orders_greedy(inst, D)
    mode_cost = sum(inst.activities[j].modes[mode_assignments[j]].cost for j in inst.activities)
    cmax = max(starts[j] + inst.activities[j].modes[mode_assignments[j]].duration for j in inst.activities)
    cost = mode_cost + MAKESPAN_WEIGHT * cmax + plan.cost
    return cost, cmax, D


def run_cg_cpsat(inst: Instance = None, max_iters: int = MAX_PRICING_ITERS):
    """Run the full CG loop with CP-SAT pricing. Returns final results dict."""
    if inst is None:
        inst = build_toy_instance()

    env_seed = os.getenv("CG_SEED_STARTS", "").strip()
    if env_seed:
        try:
            parsed = json.loads(env_seed)
            assert isinstance(parsed, dict)
            for j in inst.activities.keys():
                assert j in parsed
                assert isinstance(parsed[j], int)
                assert 0 <= parsed[j] <= inst.horizon
            starts0 = {str(k): int(v) for k, v in parsed.items()}
            cost0, cmax0_chk, D0 = schedule_cost(inst, starts0)
            print("Seed schedule (from env):", starts0, "Cmax=", cmax0_chk, "Cost=", cost0)
        except Exception as e:
            print("Failed to parse CG_SEED_STARTS, falling back to CP-SAT seed:", e)
            starts0, modes0, cmax0, _ = schedule_with_cpsat(inst, penalty_alpha=None, time_limit_s=SCHED_TIME_LIMIT_S)
            cost0, cmax0_chk, D0 = schedule_cost(inst, starts0, modes0)
            print("Seed schedule:", starts0, "Cmax=", cmax0, "Cost=", cost0)
    else:
        starts0, modes0, cmax0, _ = schedule_with_cpsat(inst, penalty_alpha=None, time_limit_s=SCHED_TIME_LIMIT_S)
        cost0, cmax0_chk, D0 = schedule_cost(inst, starts0, modes0)
        print("Seed schedule:", starts0, "Cmax=", cmax0, "Cost=", cost0)

    pool_starts: List[Dict[str, int]] = [starts0]
    pool_cost: List[float] = [cost0]
    pool_cmax: List[int] = [cmax0_chk]
    pool_D: List[Dict[str, List[int]]] = [D0]

    trace_rows: List[Dict[str, object]] = []
    for it in range(1, max_iters + 1):
        lam, alpha_duals, mu, obj_val, slack_sum = solve_master_lp(inst, pool_D, pool_cost, early_only=True)
        print(f"\n=== CG ITER {it} ===")
        print("Master lambdas:", [round(x, 3) for x in lam], "mu:", round(mu, 6))
        print("Master objective:", round(obj_val, 3), "Slack sum:", round(slack_sum, 6))

        starts_p, modes_p, cmax_p, alpha_reward = schedule_with_cpsat(inst, penalty_alpha=alpha_duals, time_limit_s=SCHED_TIME_LIMIT_S)
        cost_p, _, Dp = schedule_cost(inst, starts_p, modes_p)

        alpha_dot_D = 0.0
        for m in inst.parts:
            for t, d_mt in enumerate(Dp[m]):
                if d_mt:
                    alpha_dot_D += alpha_duals.get((m, t), 0.0) * d_mt
        rc = cost_p - alpha_dot_D - mu

        print("Pricing schedule:", starts_p, "Cmax=", cmax_p, "Cost=", round(cost_p, 3))
        print("Alpha*D=", round(alpha_dot_D, 6), "Reduced cost=", round(rc, 6))
        trace_rows.append({
            "iter": it, "mu": mu, "master_obj": obj_val, "slack_sum": slack_sum,
            "pricing_cost": cost_p, "alpha_dot": alpha_dot_D, "reduced_cost": rc,
        })

        if rc < -IMPROVEMENT_TOL:
            print("Adding column with negative reduced cost.")
            pool_starts.append(starts_p)
            pool_cost.append(cost_p)
            pool_cmax.append(cmax_p)
            pool_D.append(Dp)
        else:
            print("No improving column found. Stopping.")
            break

    lam, _, _, obj_val, slack_sum = solve_master_lp(inst, pool_D, pool_cost, early_only=True)
    best_k = max(range(len(lam)), key=lambda k: lam[k])
    print("\n=== FINAL MASTER ===")
    print("Lambdas:", [round(x, 3) for x in lam])
    print("Chosen schedule (argmax lam):", pool_starts[best_k])
    print("Cost:", round(pool_cost[best_k], 3))
    print("Master objective:", round(obj_val, 3), "Slack sum:", round(slack_sum, 6))

    csv_path = os.getenv("CG_CSV_LOG", "").strip()
    if csv_path:
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "iter", "mu", "master_obj", "slack_sum",
                    "pricing_cost", "alpha_dot", "reduced_cost",
                ])
                writer.writeheader()
                for row in trace_rows:
                    writer.writerow(row)
            print("Trace CSV written to:", csv_path)
        except Exception as e:
            print("Failed to write CSV:", e)

    return {
        "final_obj": obj_val,
        "final_cost": pool_cost[best_k],
        "final_starts": pool_starts[best_k],
        "num_columns": len(pool_starts),
        "trace": trace_rows,
    }


if __name__ == "__main__":
    run_cg_cpsat()


