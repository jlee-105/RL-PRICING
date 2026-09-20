"""
Column generation as a large-scale heuristic with interchangeable pricers.

    pricer "exact"  CP-SAT per project (time-limited)
           "heur"   the seven anchor heuristics of rl_pricing_trainer, all run per project
           "rl"    the learned RL pricer: all projects x all anchors in one batched rollout

The master uses the compact inventory-balance procurement ("bigm"): its LP takes ~0.1 s at
P = 40, against ~17 s for the facility-location master, and it gives the same kind of duals
(pi per resource-period, beta per material-period). CG stops when no pricer column has negative
reduced cost or at the time limit; the final plan comes from the integer master over the column
pool, and its cost is re-evaluated with optimal (Wagner-Whitin) procurement for the chosen
schedule, so every method is scored by the same evaluator.
"""
from __future__ import annotations

import random
import time
from typing import Optional

import torch

from multiproject_instance import MultiProjectInstance
from mp_cg import (initial_columns, solve_master, price_exact, reduced_cost, make_column,
                   order_candidates, shift_project, RC_TOL)
from mp_monolithic import evaluate_solution
from mp_pricing_env import INFEASIBLE_RC
import rl_pricing_trainer as T


def _heur_columns(inst, p, m, rng):
    cols = []
    for o in order_candidates(inst, m)[p]:
        prob = {"inst": shift_project(inst, p, o), "p": p, "pi": m["pi"], "beta": m["beta"], "mu": m["mu"]}
        env = T.make_env(prob)
        for an, fn in T.ANCHORS.items():
            if fn is None:
                continue
            rc, col = T.heuristic_rollout(env, an, rng)
            if col is not None:
                cols.append(make_column(inst, p, col.starts, col.modes, order=o))
    return cols


@torch.no_grad()
def _rl_columns(inst, m, policy, S, rng):
    """One column per (project, anchor, candidate order period). A PricingGNNBatch policy prices
    every case in a single batched rollout on the GPU; the scalar policy falls back to loops."""
    cands = order_candidates(inst, m)
    probs, tags = [], []
    for p in range(len(inst.projects)):
        for o in cands[p]:
            probs.append({"inst": shift_project(inst, p, o), "p": p,
                          "pi": m["pi"], "beta": m["beta"], "mu": m["mu"]})
            tags.append((p, o))
    if policy.__class__.__name__.endswith("Batch"):
        import rl_pricing_trainer_batch as TB
        dev = next(policy.parameters()).device
        g = TB.make_generator(dev, rng.randrange(1 << 30))
        n_anchor = len(TB.ANCHOR_NAMES)
        env, _, _, _ = TB.run_batch(policy, probs, n_anchor, S, greedy=True, device=dev,
                                    rng_t=g, track_grad=False)
        out = [[] for _ in inst.projects]
        for k, col in enumerate(env.columns()):
            if col is not None:
                p, o = tags[k // n_anchor]
                out[p].append(make_column(inst, p, col["starts"], col["modes"], order=o))
        return out
    _, meta, recs = T.run_rollouts(policy, probs, list(T.ANCHORS), S, greedy=True, rng=rng, track_grad=False)
    out = [[] for _ in inst.projects]
    for (i, _), rc in zip(meta, recs):
        col = (rc["info"] or {}).get("column")
        if col is not None:
            p, o = tags[i]
            # the scalar env hands back a Column built on the shifted instance; rebuild it here so
            # the order period is recorded (the batch env returns plain dicts instead)
            out[p].append(make_column(inst, p, col.starts, col.modes, order=o))
    return out


def sgs_repair(inst: MultiProjectInstance, pref_starts, modes, keep_timing: bool, arrivals=None):
    """Serial SGS over all tasks in order of their preferred start (LP-chosen columns). Each task
    goes to the earliest period >= its predecessors' ends and >= the release (or >= its preferred
    start if keep_timing) where every resource of its mode, shared or tool, has room.
    Returns feasible starts or None if some task cannot finish by its deadline."""
    H = inst.horizon
    use = {r: [0] * H for r in inst.capacity}
    order = sorted(((pref_starts[j], p, j) for p, pr in enumerate(inst.projects) for j in pr.activities))
    starts = {}
    for _, p, j in order:
        pr = inst.projects[p]
        md = pr.activities[j].modes[modes[j]]
        preds = [i for i, k in pr.precedence if k == j]
        lb = max([starts[i] + pr.activities[i].modes[modes[i]].duration for i in preds]
                 + [(arrivals or {}).get(pr.id, pr.release)])
        if keep_timing:
            lb = max(lb, pref_starts[j])
        t = lb
        while t + md.duration <= pr.deadline:
            if all(use[r][u] + k <= inst.capacity[r][u]
                   for r, k in md.demand_renewable.items() for u in range(t, t + md.duration)):
                break
            t += 1
        else:
            return None
        starts[j] = t
        for r, k in md.demand_renewable.items():
            for u in range(t, t + md.duration):
                use[r][u] += k
    return starts


def dive(inst, pools, procurement: str, max_lp_s: float = 300.0):
    """LP-guided diving: solve the master LP, fix the project whose largest lambda is closest to 1
    to that column, re-solve, repeat. Rounding every project at once (price-and-branch) ignores how
    the remaining projects then have to share the resources; fixing one at a time lets the LP
    re-plan the others after each commitment."""
    fixed = {}
    for _ in range(len(pools)):
        sub = [[pl[fixed[p]]] if p in fixed else pl for p, pl in enumerate(pools)]
        try:
            m = solve_master(inst, sub, procurement=procurement, time_limit_s=max_lp_s)
        except RuntimeError:
            return None
        best_p, best_k, best_v = None, None, -1.0
        for p, lam in enumerate(m["lam"]):
            if p in fixed:
                continue
            k = max(range(len(lam)), key=lambda k: lam[k])
            if lam[k] > best_v:
                best_p, best_k, best_v = p, k, lam[k]
        if best_p is None:
            break
        fixed[best_p] = best_k
    if len(fixed) < len(pools):
        return None
    return [pools[p][fixed[p]] for p in range(len(pools))]


def _plan_from_columns(inst, chosen):
    """Evaluate a choice of one column per project (raw, and after both repairs)."""
    pref, modes, tool_orders, arrivals = {}, {}, {}, {}
    for p, col in enumerate(chosen):
        pref.update(col.starts)
        modes.update(col.modes)
        pr = inst.projects[p]
        tool_orders[pr.id] = col.order
        arrivals[pr.id] = col.order + pr.tool_lead if inst.capex_budget is not None else pr.release
    orders = tool_orders if inst.capex_budget is not None else None
    best = None
    for s in (pref, sgs_repair(inst, pref, modes, True, arrivals),
              sgs_repair(inst, pref, modes, False, arrivals)):
        if s is None:
            continue
        ev = evaluate_solution(inst, s, modes, orders=None, tool_orders=orders)
        if ev is not None and (best is None or ev["total"] < best[0]["total"]):
            best = (ev, s)
    return best


def _best_plan(inst, pools, lam):
    """Argmax-lambda column per project, repaired two ways; cheapest feasible plan."""
    chosen = [pl[max(range(len(pl)), key=lambda k: lam[p][k])] for p, pl in enumerate(pools)]
    return _plan_from_columns(inst, chosen)


def run(inst: MultiProjectInstance, pricer: str, time_limit_s: float = 300.0, max_iters: int = 500,
        policy=None, S: int = 1, exact_tl: float = 5.0, ip_time_s: float = 60.0, seed: int = 0,
        procurement: str = "bigm", verbose: bool = False) -> dict:
    rng = random.Random(seed)
    t0 = time.time()
    pools = initial_columns(inst)
    seen = [{c.key() for c in pl} for pl in pools]
    trace, it = [], 0
    t_price = 0.0
    while it < max_iters and time.time() - t0 < time_limit_s:
        it += 1
        m = solve_master(inst, pools, procurement=procurement)
        tp = time.time()
        if pricer == "exact":
            cands = order_candidates(inst, m)
            cols = []
            for p in range(len(pools)):
                cp = []
                for o in cands[p]:
                    col = price_exact(shift_project(inst, p, o), p, m["pi"], m["beta"], m["mu"], exact_tl)[0]
                    cp.append(make_column(inst, p, col.starts, col.modes, order=o))
                cols.append(cp)
        elif pricer == "heur":
            cols = [_heur_columns(inst, p, m, rng) for p in range(len(pools))]
        elif pricer == "rl":
            cols = _rl_columns(inst, m, policy, S, rng)
        else:
            raise ValueError(pricer)
        t_price += time.time() - tp
        added = 0
        for p, cl in enumerate(cols):
            for c in cl:
                if reduced_cost(c, m["pi"], m["beta"], m["mu"], m.get("eta")) < -RC_TOL                         and c.key() not in seen[p]:
                    pools[p].append(c)
                    seen[p].add(c.key())
                    added += 1
        trace.append((round(time.time() - t0, 2), m["obj"], m["slack"], added))
        if verbose:
            print(f"  [{pricer}] it {it} t={time.time() - t0:.1f}s rmp={m['obj']:.2f} slack={m['slack']:.2f} +{added}",
                  flush=True)
        if added == 0:
            break
    t_cg = time.time() - t0
    lp = solve_master(inst, pools, procurement=procurement)
    try:
        ip = solve_master(inst, pools, integer=True, time_limit_s=ip_time_s, procurement=procurement)
    except RuntimeError:
        ip = None
    res = {"pricer": pricer, "iters": it, "cols": sum(len(pl) for pl in pools), "lp": lp["obj"],
           "lp_slack": lp["slack"], "t_cg": t_cg, "t_pricing": t_price, "trace": trace}
    # candidate plans: the LP's argmax columns, the integer master's, and LP-guided diving,
    # each also passed through the SGS repair; the cheapest feasible one wins
    plans = [_best_plan(inst, pools, lp["lam"])]
    if ip is not None:
        plans.append(_best_plan(inst, pools, ip["lam"]))
    dived = dive(inst, pools, procurement)
    if dived is not None:
        plans.append(_plan_from_columns(inst, dived))
    plans = [pl for pl in plans if pl is not None]
    if not plans:
        res.update(cost=None, feasible=False, t_total=time.time() - t0)
        return res
    ev, starts = min(plans, key=lambda pl: pl[0]["total"])
    res.update(cost=ev["total"], breakdown=ev, feasible=True, t_total=time.time() - t0,
               ip_feasible=ip is not None and ip["slack"] <= 1e-6)
    return res
