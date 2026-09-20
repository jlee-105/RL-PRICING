"""
Branch-and-price for the multi-project problem, branching only on setups delta[m,s].

delta lives only in the master, so a branch fixes one master variable and leaves every
pricing problem unchanged (only the duals move). Consequences used here:
  * one global column pool serves every node (a column is valid everywhere);
  * any pricer -- exact or learned -- runs unmodified throughout the tree.

Node bound: z_RMP + sum_p min(0, rc*_p) with exact pricing (Lagrangian), which also
prunes a node before its CG has converged. Node selection is best-bound first.

If a node converges with every delta integral but lam fractional, delta branching alone
cannot close it; such nodes are reported (`unresolved`) and their bound stays in the
global lower bound, so the reported gap remains valid.
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from multiproject_instance import MultiProjectInstance
from mp_monolithic import evaluate_solution
from mp_cg import (Column, initial_columns, price_exact, reduced_cost, solve_master, RC_TOL)

INT_TOL = 1e-6
SLACK_TOL = 1e-6


@dataclass(order=True)
class Node:
    bound: float
    id: int
    fix: Dict[Tuple[str, int], int] = field(compare=False)
    depth: int = field(compare=False, default=0)


class BranchAndPrice:
    def __init__(self, inst: MultiProjectInstance, pricing_time_limit_s: float = 30.0,
                 integer_time_limit_s: float = 10.0, verbose: bool = True,
                 max_pricing_time_limit_s: float = 600.0, incumbent_every: int = 10):
        self.max_pricing_tl = max_pricing_time_limit_s
        self.incumbent_every = incumbent_every
        self.inst = inst
        self.P = len(inst.projects)
        self.pool: List[List[Column]] = initial_columns(inst)
        self.seen = [{c.key() for c in pl} for pl in self.pool]
        self.pricing_tl = pricing_time_limit_s
        self.integer_tl = integer_time_limit_s
        self.verbose = verbose
        self.incumbent = float("inf")
        self.best = None
        self.stats = {"nodes": 0, "pruned": 0, "infeasible": 0, "unresolved": 0,
                      "cg_iters": 0, "pricing_calls": 0, "t_pricing": 0.0, "t_master": 0.0}

    # -- column generation at one node -------------------------------------------------
    def node_cg(self, fix) -> Tuple[dict, float, bool]:
        """Returns (final master LP, node lower bound, pruned early)."""
        lb = -float("inf")
        while True:
            m = solve_master(self.inst, self.pool, delta_fix=fix)
            self.stats["t_master"] += m["time"]
            self.stats["cg_iters"] += 1
            added, lag, exact = 0, m["obj"], True
            t0 = time.time()
            for p in range(self.P):
                col, rc_lb, proven = price_exact(self.inst, p, m["pi"], m["beta"], m["mu"], self.pricing_tl)
                self.stats["pricing_calls"] += 1
                exact &= proven
                lag += min(0.0, rc_lb)
                if reduced_cost(col, m["pi"], m["beta"], m["mu"]) < -RC_TOL and col.key() not in self.seen[p]:
                    self.pool[p].append(col)
                    self.seen[p].add(col.key())
                    added += 1
            self.stats["t_pricing"] += time.time() - t0
            if exact:
                lb = max(lb, lag)
            if lb >= self.incumbent - 1e-6:
                return m, lb, True
            if added == 0:
                if exact:
                    return m, max(lb, m["obj"]), False
                # no improving column found, but some pricing problem was not proven optimal:
                # the LP value is not a valid bound; re-price with a longer limit before trusting it
                self.stats["pricing_retries"] = self.stats.get("pricing_retries", 0) + 1
                if self.pricing_tl >= self.max_pricing_tl:
                    return m, lb, False                          # keep only the certified bound
                self.pricing_tl = min(2 * self.pricing_tl, self.max_pricing_tl)

    # -- incumbent from the current pool -------------------------------------------------
    def try_incumbent(self, fix):
        try:
            ip = solve_master(self.inst, self.pool, integer=True, time_limit_s=self.integer_tl, delta_fix=fix)
        except RuntimeError:                     # no integer solution within the time limit
            return
        if ip["slack"] > SLACK_TOL:
            return
        starts, modes = {}, {}
        for p, pl in enumerate(self.pool):
            k = max(range(len(pl)), key=lambda k: ip["lam"][p][k])
            starts.update(pl[k].starts)
            modes.update(pl[k].modes)
        chk = evaluate_solution(self.inst, starts, modes, ip["orders"])
        if chk is not None and chk["total"] < self.incumbent - 1e-9:
            self.incumbent = chk["total"]
            self.best = {"starts": starts, "modes": modes, "orders": ip["orders"], "cost": chk}
            if self.verbose:
                print(f"    new incumbent {self.incumbent:.3f}", flush=True)

    # -- branching -------------------------------------------------------------------------
    def branch(self, fix, delta) -> List[dict]:
        """Branch on cumulative setup counts G_m(tau) = sum_{s <= tau} delta[m,s].
        Single-delta branching barely moves the bound (the LP shifts the order to s +- 1);
        fixing how many orders happen up to tau cuts off all those shifted solutions at once.
        First the total count per part (tau = last period), then the cumulative count whose
        fractional part is closest to 1/2, ties broken by the larger setup cost."""
        best, best_key = None, None
        for m in self.inst.parts:
            cum = 0.0
            periods = sorted(s for (mm, s) in delta if mm == m)
            for idx, s in enumerate(periods):
                cum += delta[m, s]
                f = cum - int(cum + INT_TOL)
                if INT_TOL < f < 1 - INT_TOL:
                    is_total = idx == len(periods) - 1
                    key = (is_total, 0.5 - abs(f - 0.5), self.inst.setup_cost[m])
                    if best_key is None or key > best_key:
                        best_key, best = key, (m, s, cum)
        m, tau, val = best
        children = []
        for sense, k in (("<=", int(val)), (">=", int(val) + 1)):
            cum_fix = dict(fix.get("cum") or {})
            old = cum_fix.get((m, tau, sense))
            cum_fix[m, tau, sense] = k if old is None else (min(old, k) if sense == "<=" else max(old, k))
            children.append({**fix, "cum": cum_fix})
        return children

    # -- tree ----------------------------------------------------------------------------
    def solve(self, time_limit_s: float = 3600.0, node_limit: int = 10_000) -> dict:
        t_start = time.time()
        counter = itertools.count()
        open_nodes: List[Node] = [Node(-float("inf"), next(counter), {})]
        unresolved_bounds: List[float] = []
        root_lb = None

        while open_nodes and self.stats["nodes"] < node_limit and time.time() - t_start < time_limit_s:
            node = heapq.heappop(open_nodes)
            if node.bound >= self.incumbent - 1e-6:
                self.stats["pruned"] += 1
                continue
            self.stats["nodes"] += 1
            m, lb, early = self.node_cg(node.fix)
            if root_lb is None:
                root_lb = lb
            if self.stats["nodes"] % self.incumbent_every == 1 or self.incumbent_every == 1:
                self.try_incumbent({})                         # price-and-branch over the global pool
            frac = {k: v for k, v in m["delta"].items() if INT_TOL < v < 1 - INT_TOL}
            if self.verbose:
                print(f"node {node.id:4d} depth {node.depth:2d}  LB={lb:10.3f}  inc={self.incumbent:10.3f}  "
                      f"frac delta={len(frac):3d}  slack={m['slack']:.2g}  open={len(open_nodes)}", flush=True)
            if early or lb >= self.incumbent - 1e-6:
                self.stats["pruned"] += 1
                continue
            if not frac:
                if m["slack"] > SLACK_TOL:
                    self.stats["infeasible"] += 1               # converged, setups fixed, still needs slack
                    continue
                self.try_incumbent(node.fix)
                if self.incumbent <= lb + 1e-6:
                    continue                                    # node solved to optimality
                self.stats["unresolved"] += 1                   # integral delta, fractional lam
                unresolved_bounds.append(lb)
                continue
            for child in self.branch(node.fix, m["delta"]):
                heapq.heappush(open_nodes, Node(lb, next(counter), child, node.depth + 1))

        global_lb = min([n.bound for n in open_nodes] + unresolved_bounds + [self.incumbent])
        return {"objective": self.incumbent, "lower_bound": global_lb, "root_lb": root_lb,
                "gap": (self.incumbent - global_lb) / abs(self.incumbent) if self.best else None,
                "solved": not open_nodes and not unresolved_bounds,
                "time": time.time() - t_start, "columns": sum(len(pl) for pl in self.pool),
                **self.stats, "solution": self.best}
