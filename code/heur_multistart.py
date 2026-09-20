"""Prototype baseline: multi-start priority-rule SGS + greedy capex ordering.

Reuses cg_runner.sgs_repair as the decoder (with keep_timing=False it sorts by the priority
vector rather than using it as a lower bound) and mp_monolithic.evaluate_solution for scoring,
so this is scored exactly like MIP and every CG variant. Priorities are random topological
orders per project, because sgs_repair assumes the order is precedence-feasible.

Not yet: forward-backward improvement, a shared time limit, integration into exp_*.
"""
import random
import time

from cg_runner import sgs_repair
from mp_monolithic import evaluate_solution


def rand_priority(pr, rng, base=0):
    """Random topological order of one project; the rank is the priority."""
    succ, indeg = {j: [] for j in pr.activities}, {j: 0 for j in pr.activities}
    for i, k in pr.precedence:
        succ[i].append(k)
        indeg[k] += 1
    ready = [j for j in pr.activities if indeg[j] == 0]
    prio, r = {}, 0
    while ready:
        j = ready.pop(rng.randrange(len(ready)))
        prio[j] = base + r
        r += 1
        for k in succ[j]:
            indeg[k] -= 1
            if indeg[k] == 0:
                ready.append(k)
    return prio


def greedy_orders(inst):
    """Order long-lead, early-due tools first, into the first period the capex budget allows."""
    if inst.capex_budget is None:
        return None, {}
    spent = [0.0] * len(inst.capex_budget)
    orders, arrivals = {}, {}
    for _, pr in sorted(enumerate(inst.projects), key=lambda x: (-x[1].tool_lead, x[1].due)):
        for t in range(pr.order_latest + 1):
            if spent[t] + pr.tool_cost <= inst.capex_budget[t] + 1e-9:
                spent[t] += pr.tool_cost
                orders[pr.id], arrivals[pr.id] = t, t + pr.tool_lead
                break
        else:
            orders[pr.id], arrivals[pr.id] = pr.order_latest, pr.order_latest + pr.tool_lead
    return orders, arrivals


def multistart(inst, n_starts=2000, seed=0, time_limit_s=1e9):
    rng = random.Random(seed)
    orders, arrivals = greedy_orders(inst)
    best, t0, feas = None, time.time(), 0
    for _ in range(n_starts):
        if time.time() - t0 > time_limit_s:
            break
        prio, modes = {}, {}
        for pr in inst.projects:
            prio.update(rand_priority(pr, rng))
            for j, act in pr.activities.items():
                modes[j] = rng.randrange(len(act.modes))
        s = sgs_repair(inst, prio, modes, False, arrivals)
        if s is None:
            continue
        ev = evaluate_solution(inst, s, modes, orders=None, tool_orders=orders)
        if ev is not None:
            feas += 1
            if best is None or ev["total"] < best["total"]:
                best = ev
    return {"cost": None if best is None else best["total"], "breakdown": best,
            "feasible_starts": feas, "time": time.time() - t0}
