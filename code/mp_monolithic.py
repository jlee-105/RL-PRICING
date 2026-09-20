"""
Monolithic time-indexed MIP for the multi-project problem (ground truth for small P),
and an independent solution evaluator shared by every method.

Periods are t = 0..H-1. A task of project p starts no earlier than r_p and ends
no later than the project deadline. Objective, all in money:

    sum_p w_p * max(0, C_p - d_p)                 tardiness (cost of delay)
  + sum_{j,q} gamma_{jq} z                        mode costs
  + sum_{m,t} (c_m o + f_m delta + h_m inv)       procurement (shared inventory)
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from ortools.linear_solver import pywraplp

from multiproject_instance import MultiProjectInstance


def task_windows(inst: MultiProjectInstance):
    """(project index, task id, activity, release, deadline) for every task."""
    for p, pr in enumerate(inst.projects):
        for j, act in pr.activities.items():
            yield p, j, act, pr.release, pr.deadline


def evaluate_solution(inst: MultiProjectInstance, starts: Dict[str, int], modes: Dict[str, int],
                      orders: Optional[Dict[str, List[int]]] = None,
                      tool_orders: Optional[Dict[str, int]] = None) -> Optional[dict]:
    """Cost breakdown of a full solution, or None if infeasible. If `orders` is None the
    procurement is planned optimally for the given schedule (per part, Wagner-Whitin DP).
    With tool purchase, `tool_orders` gives the period each tool is ordered; it must respect the
    capex budget, and a project cannot start before its tool arrives (order + tool lead). If it is
    None, every tool is assumed ordered as early as possible (period 0)."""
    H = inst.horizon
    buys = inst.capex_budget is not None
    arrival = {}
    for pr in inst.projects:
        o = 0 if tool_orders is None else tool_orders.get(pr.id, 0)
        if buys and (o < 0 or o > pr.order_latest):
            return None
        arrival[pr.id] = (o + pr.tool_lead) if buys else pr.release
    if buys:
        spend = [0.0] * H
        for pr in inst.projects:
            spend[(tool_orders or {}).get(pr.id, 0)] += pr.tool_cost
        if any(x > b + 1e-9 for x, b in zip(spend, inst.capex_budget)):
            return None
    dur = {}
    for p, j, act, rel, dl in task_windows(inst):
        dur[j] = act.modes[modes[j]].duration
        if starts[j] < max(rel, arrival[inst.projects[p].id]) or starts[j] + dur[j] > dl:
            return None
    for pr in inst.projects:
        if any(starts[j] < starts[i] + dur[i] for i, j in pr.precedence):
            return None
    for r, cap in inst.capacity.items():
        use = [0] * H
        for p, j, act, *_ in task_windows(inst):
            k = act.modes[modes[j]].demand_renewable.get(r, 0)
            for t in range(starts[j], starts[j] + dur[j]):
                use[t] += k
        if any(u > c for u, c in zip(use, cap)):
            return None

    tard = sum(pr.tardiness_weight * max(0, max(starts[j] + dur[j] for j in pr.activities) - pr.due)
               for pr in inst.projects)
    mode_cost = sum(act.modes[modes[j]].cost for _, j, act, *_ in task_windows(inst))
    capital = sum(pr.tool_cost for pr in inst.projects) if buys else 0.0

    demand = {m: [0] * H for m in inst.parts}
    for _, j, act, *_ in task_windows(inst):
        for m, a in act.modes[modes[j]].demand_parts.items():
            demand[m][starts[j]] += a

    proc = 0.0
    for m in inst.parts:
        if orders is None:
            c = lot_sizing_cost(inst, m, demand[m])
            if c is None:
                return None
            proc += c
        else:
            L, I = inst.lead_time[m], inst.init_inventory.get(m, 0)
            for t in range(H):
                if orders[m][t]:
                    if t + L > H - 1:
                        return None
                    proc += inst.unit_cost[m] * orders[m][t] + inst.setup_cost[m]
                I += (orders[m][t - L] if t >= L else 0) - demand[m][t]
                if I < 0:
                    return None
                proc += inst.holding_cost[m] * I
    out = {"total": tard + mode_cost + proc + capital, "tardiness": tard, "mode": mode_cost,
           "procurement": proc}
    if buys:
        out["capital"] = capital
    return out


def lot_sizing_cost(inst: MultiProjectInstance, m: str, D: List[int]) -> Optional[float]:
    """Optimal single-item uncapacitated lot sizing with lead time and initial stock (Wagner-Whitin).
    Returns None if demand before the lead time exceeds initial stock."""
    H, L, I0 = len(D), inst.lead_time[m], inst.init_inventory.get(m, 0)
    c, f, h = inst.unit_cost[m], inst.setup_cost[m], inst.holding_cost[m]
    # Initial stock covers demand first-come; the remainder (net demand) must be ordered.
    net, stock, hold0 = [0] * H, I0, 0.0
    for t in range(H):
        use = min(stock, D[t])
        stock -= use
        net[t] = D[t] - use
        hold0 += h * stock
        if net[t] and t < L:
            return None
    # WW over net demand: an order arriving at s (s >= L) covers net demand of s..e.
    INF = float("inf")
    best = [INF] * (H + 1)
    best[H] = 0.0
    for s in range(H - 1, -1, -1):
        if net[s] == 0:
            best[s] = best[s + 1]
            continue
        if s < L:
            continue
        qty, hold = 0, 0.0
        for e in range(s, H):
            qty += net[e]
            hold += h * net[e] * (e - s)      # units for period e are held from s to e
            if best[e + 1] < INF:
                best[s] = min(best[s], f + c * qty + hold + best[e + 1])
    return hold0 + best[0] if best[0] < INF else None


def solve_monolithic(inst: MultiProjectInstance, time_limit_s: float = 600.0,
                     solver_name: str = "SCIP", verbose: bool = False) -> dict:
    H = inst.horizon
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if verbose:
        solver.EnableOutput()
    solver.SetTimeLimit(int(time_limit_s * 1000))

    z, by_task, acts = {}, {}, {}
    for p, j, act, rel, dl in task_windows(inst):
        acts[j] = act
        by_task[j] = []
        for q, md in enumerate(act.modes):
            for t in range(rel, dl - md.duration + 1):
                z[j, q, t] = solver.BoolVar(f"z_{j}_{q}_{t}")
                by_task[j].append((q, t))
        solver.Add(sum(z[j, q, t] for q, t in by_task[j]) == 1)

    start = {j: sum(t * z[j, q, t] for q, t in by_task[j]) for j in acts}
    end = {j: sum((t + acts[j].modes[q].duration) * z[j, q, t] for q, t in by_task[j]) for j in acts}

    obj = []
    for pr in inst.projects:
        for i, j in pr.precedence:
            solver.Add(start[j] >= end[i])
        T = solver.NumVar(0, solver.infinity(), f"T_{pr.id}")
        for j in pr.activities:
            solver.Add(T >= end[j] - pr.due)
        obj.append(pr.tardiness_weight * T)

    for r, cap in inst.capacity.items():
        use = [[] for _ in range(H)]
        for (j, q, t), v in z.items():
            k = acts[j].modes[q].demand_renewable.get(r, 0)
            if k:
                for u in range(t, t + acts[j].modes[q].duration):
                    use[u].append(k * v)
        for u in range(H):
            if use[u]:
                solver.Add(sum(use[u]) <= cap[u])

    obj += [acts[j].modes[q].cost * v for (j, q, t), v in z.items() if acts[j].modes[q].cost]

    demand = {m: [[] for _ in range(H)] for m in inst.parts}
    for (j, q, t), v in z.items():
        for m, a in acts[j].modes[q].demand_parts.items():
            demand[m][t].append(a * v)
    o = {}
    for m in inst.parts:
        L, prev = inst.lead_time[m], inst.init_inventory.get(m, 0)
        U = sum(max(md.demand_parts.get(m, 0) for md in acts[j].modes) for j in acts)
        for t in range(H - L):
            o[m, t] = solver.IntVar(0, U, f"o_{m}_{t}")
            d = solver.BoolVar(f"d_{m}_{t}")
            solver.Add(o[m, t] <= U * d)
            obj += [inst.unit_cost[m] * o[m, t], inst.setup_cost[m] * d]
        for t in range(H):
            inv = solver.NumVar(0, solver.infinity(), f"inv_{m}_{t}")
            solver.Add(inv == prev + (o[m, t - L] if t >= L else 0) - sum(demand[m][t]))
            obj.append(inst.holding_cost[m] * inv)
            prev = inv

    solver.Minimize(sum(obj))
    t0 = time.time()
    status = solver.Solve()
    res = {"status": {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
                      pywraplp.Solver.INFEASIBLE: "INFEASIBLE"}.get(status, f"OTHER({status})"),
           "wall_time": time.time() - t0, "num_vars": solver.NumVariables()}
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return res
    starts, modes = {}, {}
    for (j, q, t), v in z.items():
        if v.solution_value() > 0.5:
            starts[j], modes[j] = t, q
    orders = {m: [0] * H for m in inst.parts}
    for (m, t), v in o.items():
        orders[m][t] = int(round(v.solution_value()))
    res.update(objective=solver.Objective().Value(), bound=solver.Objective().BestBound(),
               starts=starts, modes=modes, orders=orders,
               check=evaluate_solution(inst, starts, modes, orders))
    return res
