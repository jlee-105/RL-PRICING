"""
Integrated time-indexed MIP for MRCPSP-P (scheduling + procurement), solved
monolithically (no decomposition). Follows document/FORMULATIONS.md
("Original Integrated MIP"), extended to multiple modes with mode costs.

    z[j,q,t] in {0,1}   task j starts at t in mode q (end <= H)
    o[m,t]   in Z+      order of part m placed at t, arrives at t + L_m
    d[m,t]   in {0,1}   setup indicator of that order
    inv[m,t] >= 0       end-of-period inventory (hard: no backlog)

    min  sum_{j,q} gamma_{jq} z + sum_{m,t} (c_m o + f_m d + h_m inv) + lambda * Cmax
    s.t. sum_{q,t} z[j,q,t] = 1
         precedence, renewable capacity (time-indexed)
         inv[m,t] = inv[m,t-1] + o[m,t-L_m] - D[m,t],  inv[m,-1] = I0_m
         o[m,t] <= U_m d[m,t]

Unlike the CG model, procurement is optimized (orders may be batched), and
early demand (t < L_m) must be covered by initial stock -- no slack.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from ortools.linear_solver import pywraplp

from ColumnGenerationHeuristic import Instance, MAKESPAN_WEIGHT, schedule_cost


def solve_integrated_mip(inst: Instance,
                         time_limit_s: float = 300.0,
                         solver_name: str = "SCIP",
                         makespan_weight: float = MAKESPAN_WEIGHT,
                         verbose: bool = False) -> dict:
    H = inst.horizon
    acts = inst.activities
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"MIP solver {solver_name} unavailable.")
    if verbose:
        solver.EnableOutput()
    solver.SetTimeLimit(int(time_limit_s * 1000))

    z = {(j, q, t): solver.BoolVar(f"z_{j}_{q}_{t}")
         for j, act in acts.items() for q, mode in enumerate(act.modes)
         for t in range(H - mode.duration + 1)}
    by_task = {j: [(q, t) for (jj, q, t) in z if jj == j] for j in acts}

    for j in acts:
        solver.Add(sum(z[j, q, t] for q, t in by_task[j]) == 1)

    start = {j: sum(t * z[j, q, t] for q, t in by_task[j]) for j in acts}
    end = {j: sum((t + acts[j].modes[q].duration) * z[j, q, t] for q, t in by_task[j]) for j in acts}

    for i, j in inst.precedence:
        solver.Add(start[j] >= end[i])

    for r, cap in inst.renewable_capacity.items():
        for u in range(H):
            terms = [acts[j].modes[q].demand_renewable.get(r, 0) * z[j, q, t]
                     for j in acts for q, t in by_task[j]
                     if acts[j].modes[q].demand_renewable.get(r, 0)
                     and t <= u < t + acts[j].modes[q].duration]
            if terms:
                solver.Add(sum(terms) <= cap)

    obj = [acts[j].modes[q].cost * z[j, q, t]
           for j in acts for q, t in by_task[j] if acts[j].modes[q].cost]

    if makespan_weight:
        cmax = solver.IntVar(0, H, "Cmax")
        for j in acts:
            solver.Add(cmax >= end[j])
        obj.append(makespan_weight * cmax)

    o, dlt, inv = {}, {}, {}
    for m in inst.parts:
        L, I0 = inst.lead_time[m], inst.init_inventory.get(m, 0)
        D = [sum(a * z[j, q, t] for j, act in acts.items() for q, mode in enumerate(act.modes)
                 if (j, q, t) in z and (a := mode.demand_parts.get(m, 0)))
             for t in range(H + 1)]
        U = sum(max(md.demand_parts.get(m, 0) for md in acts[j].modes) for j in acts)
        for t in range(0, H - L + 1):
            o[m, t] = solver.IntVar(0, U, f"o_{m}_{t}")
            dlt[m, t] = solver.BoolVar(f"d_{m}_{t}")
            solver.Add(o[m, t] <= U * dlt[m, t])
            obj += [inst.unit_cost[m] * o[m, t], inst.setup_cost[m] * dlt[m, t]]
        prev = I0
        for t in range(H + 1):
            inv[m, t] = solver.NumVar(0, solver.infinity(), f"inv_{m}_{t}")
            arrival = o[m, t - L] if t - L >= 0 else 0
            solver.Add(inv[m, t] == prev + arrival - D[t])
            obj.append(inst.holding_cost[m] * inv[m, t])
            prev = inv[m, t]

    solver.Minimize(sum(obj))
    t0 = time.time()
    status = solver.Solve()
    wall = time.time() - t0

    result = {"status": {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
                         pywraplp.Solver.INFEASIBLE: "INFEASIBLE"}.get(status, f"OTHER({status})"),
              "wall_time": wall, "num_vars": solver.NumVariables(), "num_cons": solver.NumConstraints()}
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return result

    starts, modes = {}, {}
    for (j, q, t), v in z.items():
        if v.solution_value() > 0.5:
            starts[j], modes[j] = t, q
    orders = {m: [0] * (H + 1) for m in inst.parts}
    for (m, t), v in o.items():
        orders[m][t] = int(round(v.solution_value()))

    result.update({
        "objective": solver.Objective().Value(),
        "bound": solver.Objective().BestBound(),
        "starts": starts, "modes": modes, "orders": orders,
        "check_cost": evaluate_plan(inst, starts, modes, orders, makespan_weight),
        # cost of the same schedule if procurement were done greedy-JIT, as in CG columns
        "greedy_jit_cost": schedule_cost(inst, starts, modes)[0],
    })
    r = result
    r["gap"] = abs(r["objective"] - r["bound"]) / max(abs(r["objective"]), 1e-9)
    return result


def evaluate_plan(inst: Instance, starts: Dict[str, int], modes: Dict[str, int],
                  orders: Dict[str, list], makespan_weight: float = MAKESPAN_WEIGHT) -> Optional[float]:
    """Independent check: cost of (schedule, orders), or None if any constraint is violated."""
    H, acts = inst.horizon, inst.activities
    dur = {j: acts[j].modes[modes[j]].duration for j in acts}
    if any(starts[j] < 0 or starts[j] + dur[j] > H for j in acts):
        return None
    if any(starts[j] < starts[i] + dur[i] for i, j in inst.precedence):
        return None
    for r, cap in inst.renewable_capacity.items():
        for u in range(H):
            if sum(acts[j].modes[modes[j]].demand_renewable.get(r, 0)
                   for j in acts if starts[j] <= u < starts[j] + dur[j]) > cap:
                return None
    cost = sum(acts[j].modes[modes[j]].cost for j in acts)
    cost += makespan_weight * max(starts[j] + dur[j] for j in acts)
    for m in inst.parts:
        L, I = inst.lead_time[m], inst.init_inventory.get(m, 0)
        for t in range(H + 1):
            q = orders[m][t]
            if q:
                if t + L > H:
                    return None
                cost += inst.unit_cost[m] * q + inst.setup_cost[m]
            arr = orders[m][t - L] if t - L >= 0 else 0
            dem = sum(acts[j].modes[modes[j]].demand_parts.get(m, 0) for j in acts if starts[j] == t)
            I = I + arr - dem
            if I < 0:
                return None
            cost += inst.holding_cost[m] * I
    return cost
