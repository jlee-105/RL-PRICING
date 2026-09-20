"""
Compact (monolithic) formulations of the multi-project problem, in variants, for
bound-strength comparisons in the style of Kolter, Grunow & Kolisch (2025).

    precedence  "agg"  S_j >= C_i                      (PDT, as in mp_monolithic)
                "dis"  sum_q sum_{s >= t-p_iq+1} z_iqs + sum_q sum_{s <= t} z_jqs <= 1   (PDDT)
    procurement "bigm" inventory balance, x <= U delta  (as in mp_monolithic)
                "fl"   facility location by project, y[m,s,e,p] <= Dmax_p[m,e] delta[m,s]
                       (the same procurement model as the CG master)
    mode        "lp"   linear relaxation (GLOP)
                "root" SCIP root node with its cuts (node limit 1): bound after the root
                "mip"  solve to optimality / time limit
"""
from __future__ import annotations

import time
from typing import Dict

from ortools.linear_solver import pywraplp

from multiproject_instance import MultiProjectInstance
from mp_monolithic import task_windows, evaluate_solution


def _max_demand(inst: MultiProjectInstance, p: int, m: str) -> Dict[int, int]:
    pr = inst.projects[p]
    out: Dict[int, int] = {}
    for act in pr.activities.values():
        a = max(md.demand_parts.get(m, 0) for md in act.modes)
        if a:
            for t in range(pr.release, pr.deadline - min(md.duration for md in act.modes) + 1):
                out[t] = out.get(t, 0) + a
    return out


def solve_compact(inst: MultiProjectInstance, precedence: str = "dis", procurement: str = "fl",
                  mode: str = "lp", time_limit_s: float = 600.0) -> dict:
    H = inst.horizon
    relax = mode == "lp"
    solver = pywraplp.Solver.CreateSolver("GLOP" if relax else "SCIP")
    inf = solver.infinity()
    if not relax:
        solver.SetTimeLimit(int(time_limit_s * 1000))
        if mode == "root":
            solver.SetSolverSpecificParametersAsString("limits/nodes = 1\n")
    binvar = (lambda n: solver.NumVar(0, 1, n)) if relax else solver.BoolVar
    intvar = (lambda lo, hi, n: solver.NumVar(lo, hi, n)) if relax else solver.IntVar

    z, by_task, acts, proj_of = {}, {}, {}, {}
    for p, j, act, rel, dl in task_windows(inst):
        acts[j], proj_of[j] = act, p
        by_task[j] = []
        for q, md in enumerate(act.modes):
            for t in range(rel, dl - md.duration + 1):
                z[j, q, t] = binvar(f"z_{j}_{q}_{t}")
                by_task[j].append((q, t))
        solver.Add(sum(z[j, q, t] for q, t in by_task[j]) == 1)

    end = {j: sum((t + acts[j].modes[q].duration) * z[j, q, t] for q, t in by_task[j]) for j in acts}
    start = {j: sum(t * z[j, q, t] for q, t in by_task[j]) for j in acts}

    obj = []
    for p, pr in enumerate(inst.projects):
        for i, j in pr.precedence:
            if precedence == "agg":
                solver.Add(start[j] >= end[i])
            else:
                for t in range(pr.release, pr.deadline):
                    lhs = [z[i, q, s] for q, s in by_task[i] if s >= t - acts[i].modes[q].duration + 1]
                    lhs += [z[j, q, s] for q, s in by_task[j] if s <= t]
                    if lhs:
                        solver.Add(sum(lhs) <= 1)
        T = solver.NumVar(0, inf, f"T_{pr.id}")
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

    order = {}
    if inst.capex_budget is not None:
        # the tool itself is bought: ordering project p in period o makes it arrive at o + lead, and
        # no task of p may start before that; the capital is committed in the order period
        for p, pr in enumerate(inst.projects):
            ovars = {t: binvar(f"o_{pr.id}_{t}") for t in range(pr.order_latest + 1)}
            order[p] = ovars
            solver.Add(sum(ovars.values()) == 1)
            obj.append(pr.tool_cost * sum(ovars.values()))
            for j in pr.activities:
                for q, t in by_task[j]:
                    ready = [ovars[o] for o in ovars if o + pr.tool_lead <= t]
                    solver.Add(z[j, q, t] <= (sum(ready) if ready else 0))
        for t in range(H):
            spend = [inst.projects[p].tool_cost * ov[t] for p, ov in order.items() if t in ov]
            if spend:
                solver.Add(sum(spend) <= inst.capex_budget[t])

    if procurement == "bigm":
        demand = {m: [[] for _ in range(H)] for m in inst.parts}
        for (j, q, t), v in z.items():
            for m, a in acts[j].modes[q].demand_parts.items():
                demand[m][t].append(a * v)
        for m in inst.parts:
            L, prev = inst.lead_time[m], inst.init_inventory.get(m, 0)
            U = sum(max(md.demand_parts.get(m, 0) for md in acts[j].modes) for j in acts)
            o = {}
            for t in range(H - L):
                o[t] = intvar(0, U, f"o_{m}_{t}")
                d = binvar(f"d_{m}_{t}")
                solver.Add(o[t] <= U * d)
                obj += [inst.unit_cost[m] * o[t], inst.setup_cost[m] * d]
            for t in range(H):
                inv = solver.NumVar(0, inf, f"inv_{m}_{t}")
                solver.Add(inv == prev + (o[t - L] if t >= L else 0) - sum(demand[m][t]))
                obj.append(inst.holding_cost[m] * inv)
                prev = inv
    else:
        for m in inst.parts:
            L, h, I0 = inst.lead_time[m], inst.holding_cost[m], inst.init_inventory.get(m, 0)
            delta = {s: binvar(f"d_{m}_{s}") for s in range(L, H)}
            obj += [inst.setup_cost[m] * d for d in delta.values()]
            obj.append(h * I0 * H)
            stock = []
            for p in range(len(inst.projects)):
                for e, cap in _max_demand(inst, p, m).items():
                    D = [a * z[j, q, e] for j in inst.projects[p].activities
                         for q, md in enumerate(acts[j].modes)
                         if (a := md.demand_parts.get(m, 0)) and (j, q, e) in z]
                    w = solver.NumVar(0, inf, f"w_{m}_{e}_{p}")
                    stock.append(w)
                    obj.append(-h * (H - e) * w)
                    supply = [w]
                    for s in range(L, e + 1):
                        y = solver.NumVar(0, cap, f"y_{m}_{s}_{e}_{p}")
                        solver.Add(y <= cap * delta[s])
                        obj.append((inst.unit_cost[m] + h * (e - s)) * y)
                        supply.append(y)
                    solver.Add(sum(supply) == sum(D))
            if stock:
                solver.Add(sum(stock) <= I0)

    solver.Minimize(sum(obj))
    t0 = time.time()
    status = solver.Solve()
    wall = time.time() - t0
    ok = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    out = {"status": status, "time": wall, "vars": solver.NumVariables(), "cons": solver.NumConstraints()}
    if relax:
        out["bound"] = solver.Objective().Value() if ok else None
        return out
    out["bound"] = solver.Objective().BestBound()
    out["incumbent"] = solver.Objective().Value() if ok else None
    if ok:
        # return the schedule too, so the MIP can be scored by the same evaluator as CG
        # (schedule + optimal Wagner-Whitin procurement) instead of by SCIP's own objective
        starts, modes = {}, {}
        for (j, q, t), v in z.items():
            if v.solution_value() > 0.5:
                starts[j], modes[j] = t, q
        out["starts"], out["modes"] = starts, modes
        tool_orders = None
        if order:
            tool_orders = {inst.projects[p].id: next(t for t, v in ov.items() if v.solution_value() > 0.5)
                           for p, ov in order.items()}
            out["tool_orders"] = tool_orders
        ev = evaluate_solution(inst, starts, modes, orders=None, tool_orders=tool_orders)
        out["eval"] = ev
        out["cost"] = None if ev is None else ev["total"]
    return out
