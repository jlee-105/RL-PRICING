"""
Dantzig-Wolfe column generation for the multi-project problem.

Column (p, k) = one schedule (starts, modes) of project p, within its window.
Procurement lives in the master, so a column only carries its own cost
(mode costs + tardiness), its renewable usage u[r][t], and its demand D[m][t].

Restricted master (LP, or MIP for the final integer solution):
  min  sum cost_pk lam_pk + sum (c o + f delta + h inv) + BIG * (slacks)
  conv_p : sum_k lam_pk                                 = 1          (mu_p)
  res_rt : sum_pk u_pk[r,t] lam_pk - s_rt              <= K[r][t]    (pi_rt <= 0)
  bal_mt : inv_mt - inv_m,t-1 - o_m,t-L - e_mt
           + sum_pk D_pk[m,t] lam_pk                    = I0 [t = 0]  (beta_mt)
  o_mt <= U delta_mt
Slacks s, e (penalty BIG) keep the RMP feasible with few columns.

Pricing for project p, linear in the time-indexed start binaries x[j,q,tau]:
  rc = gamma_jq + w_p T_p - sum_r k_jqr sum_{t in [tau, tau+d)} pi_rt
       - sum_m a_jqm beta_m,tau - mu_p
Because each conv_p row equals 1, z_RMP + sum_p min(0, rc*_p) is a valid lower
bound (Lagrangian bound) whenever the pricing problems are solved to optimality.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from multiproject_instance import MultiProjectInstance
from mp_monolithic import evaluate_solution

BIG = 1e4          # penalty on artificial slacks
RC_TOL = 1e-6
SCALE = 100_000    # CP-SAT integer scaling of pricing coefficients


@dataclass
class Column:
    p: int
    starts: Dict[str, int]
    modes: Dict[str, int]
    cost: float                         # mode costs + tardiness
    usage: Dict[str, Dict[int, int]]    # r -> {t: units}
    demand: Dict[str, Dict[int, int]]   # m -> {t: units}
    order: int = 0                      # period the project's tool is ordered (tool purchase only)
    capex: float = 0.0                  # capital committed in that period

    def key(self):
        return (self.order,) + tuple(sorted((j, self.starts[j], self.modes[j]) for j in self.starts))


def make_column(inst: MultiProjectInstance, p: int, starts: Dict[str, int], modes: Dict[str, int],
                order: int = 0) -> Column:
    pr = inst.projects[p]
    usage = {r: {} for r in inst.capacity}
    demand = {m: {} for m in inst.parts}
    cmax, mode_cost = 0, 0.0
    for j, act in pr.activities.items():
        md = act.modes[modes[j]]
        s = starts[j]
        cmax = max(cmax, s + md.duration)
        mode_cost += md.cost
        for r, k in md.demand_renewable.items():
            for t in range(s, s + md.duration):
                usage[r][t] = usage[r].get(t, 0) + k
        for m, a in md.demand_parts.items():
            demand[m][s] = demand[m].get(s, 0) + a
    cost = mode_cost + pr.tardiness_weight * max(0, cmax - pr.due)
    capex = pr.tool_cost if inst.capex_budget is not None else 0.0
    return Column(p, dict(starts), dict(modes), cost + capex, usage, demand, order, capex)


def reduced_cost(col: Column, pi, beta, mu, eta=None) -> float:
    """beta: list of per-project dicts (as returned by solve_master); eta: capex duals per period."""
    beta = beta[col.p]
    rc = col.cost - mu[col.p]
    if eta and col.capex:
        rc -= eta.get(col.order, 0.0) * col.capex
    for r, ut in col.usage.items():
        rc -= sum(pi.get((r, t), 0.0) * u for t, u in ut.items())
    for m, dt in col.demand.items():
        rc -= sum(beta.get((m, t), 0.0) * d for t, d in dt.items())
    return rc


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------

def solve_master(inst: MultiProjectInstance, pools: List[List[Column]], integer: bool = False,
                 time_limit_s: float = 300.0, solver_name: Optional[str] = None,
                 procurement: str = "fl", delta_fix: Optional[Dict[Tuple[str, int], int]] = None) -> dict:
    """procurement: "fl" (facility location by project, strong LP) or "bigm" (inventory balance,
    o <= U delta; its LP lets setups go nearly free). beta is returned per project.
    delta_fix {(part, arrival period): 0/1} fixes setups (branching); "fl" only. Its optional
    key "cum" maps (part, tau, "<=" | ">=") -> k: bounds on sum_{s <= tau} delta[part, s]."""
    H = inst.horizon
    solver = pywraplp.Solver.CreateSolver(solver_name or ("SCIP" if integer else "GLOP"))
    if integer:
        solver.SetTimeLimit(int(time_limit_s * 1000))
    inf = solver.infinity()

    lam = [[(solver.BoolVar if integer else lambda n: solver.NumVar(0, inf, n))(f"l_{p}_{k}")
            for k in range(len(pool))] for p, pool in enumerate(pools)]
    obj = [c.cost * lam[p][k] for p, pool in enumerate(pools) for k, c in enumerate(pool)]

    conv = [solver.Add(sum(lam[p]) == 1) for p in range(len(pools))]

    res_rows = {}
    for r, cap in inst.capacity.items():
        terms = [[] for _ in range(H)]
        for p, pool in enumerate(pools):
            for k, c in enumerate(pool):
                for t, u in c.usage[r].items():
                    terms[t].append(u * lam[p][k])
        for t in range(H):
            if terms[t]:
                s = solver.NumVar(0, inf, f"s_{r}_{t}")
                obj.append(BIG * s)
                res_rows[r, t] = solver.Add(sum(terms[t]) - s <= cap[t])

    capex_rows = _add_capex_rows(inst, pools, solver, lam, obj)

    if procurement == "fl":
        return _finish_master_fl(inst, pools, solver, lam, obj, conv, res_rows, capex_rows, integer,
                                 delta_fix or {})
    if delta_fix:
        raise ValueError("delta_fix requires procurement='fl'")

    bal_rows, o_vars = {}, {}
    U = {m: sum(max(md.demand_parts.get(m, 0) for md in a.modes)
                for pr in inst.projects for a in pr.activities.values()) for m in inst.parts}
    for m in inst.parts:
        L = inst.lead_time[m]
        for t in range(H - L):
            o = solver.IntVar(0, U[m], f"o_{m}_{t}") if integer else solver.NumVar(0, U[m], f"o_{m}_{t}")
            d = solver.BoolVar(f"d_{m}_{t}") if integer else solver.NumVar(0, 1, f"d_{m}_{t}")
            solver.Add(o <= U[m] * d)
            obj += [inst.unit_cost[m] * o, inst.setup_cost[m] * d]
            o_vars[m, t] = o
        terms = [[] for _ in range(H)]
        for p, pool in enumerate(pools):
            for k, c in enumerate(pool):
                for t, dq in c.demand[m].items():
                    terms[t].append(dq * lam[p][k])
        prev = None
        for t in range(H):
            inv = solver.NumVar(0, inf, f"inv_{m}_{t}")
            e = solver.NumVar(0, inf, f"e_{m}_{t}")
            obj += [inst.holding_cost[m] * inv, BIG * e]
            lhs = inv - e - (o_vars[m, t - L] if t >= L else 0) + sum(terms[t])
            if prev is not None:
                lhs = lhs - prev
            bal_rows[m, t] = solver.Add(lhs == (inst.init_inventory.get(m, 0) if t == 0 else 0))
            prev = inv

    solver.Minimize(sum(obj))
    t0 = time.time()
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"master status {status}")
    out = {"obj": solver.Objective().Value(), "time": time.time() - t0,
           "lam": [[v.solution_value() for v in row] for row in lam],
           "slack": sum(v.solution_value() for v in solver.variables()
                        if v.name().startswith(("s_", "e_")))}
    if integer:
        out["bound"] = solver.Objective().BestBound()
        out["optimal"] = status == pywraplp.Solver.OPTIMAL
        out["orders"] = {m: [int(round(o_vars[m, t].solution_value())) if (m, t) in o_vars else 0
                             for t in range(H)] for m in inst.parts}
    else:
        out["mu"] = [c.dual_value() for c in conv]
        out["pi"] = {k: c.dual_value() for k, c in res_rows.items()}
        beta = {k: c.dual_value() for k, c in bal_rows.items()}
        out["beta"] = [beta] * len(pools)
        out["eta"] = {t: r.dual_value() for t, r in capex_rows.items()}
    return out


def max_demand(inst: MultiProjectInstance, p: int, m: str) -> Dict[int, int]:
    """Largest possible demand of part m by project p at each period (all its tasks that can start then)."""
    pr = inst.projects[p]
    out: Dict[int, int] = {}
    for act in pr.activities.values():
        a = max(md.demand_parts.get(m, 0) for md in act.modes)
        if a:
            for t in range(pr.release, pr.deadline - min(md.duration for md in act.modes) + 1):
                out[t] = out.get(t, 0) + a
    return out


def _add_capex_rows(inst, pools, solver, lam, obj):
    """Tool purchase: capital committed in a period is limited by the capex budget. A column carries
    its own order period, so the row is sum over columns ordering in t of tool_cost * lambda <= B_t.
    Its dual eta_t prices the periods where the capital plan is tight."""
    if inst.capex_budget is None:
        return {}
    inf = solver.infinity()
    rows, terms = {}, {}
    for p, pool in enumerate(pools):
        for k, c in enumerate(pool):
            if c.capex:
                terms.setdefault(c.order, []).append(c.capex * lam[p][k])
    for t, ts in terms.items():
        s_t = solver.NumVar(0, inf, f"s_capex_{t}")
        obj.append(BIG * s_t)
        rows[t] = solver.Add(sum(ts) - s_t <= inst.capex_budget[t])
    return rows


def _finish_master_fl(inst, pools, solver, lam, obj, conv, res_rows, capex_rows, integer, delta_fix):
    """Procurement as a facility-location lot-sizing model, disaggregated by project:
        y[m,s,e,p]  units arriving at s (ordered at s-L) that serve project p's demand at e >= s
        w[m,e,p]    units of initial stock serving project p's demand at e
        dem_mep : sum_k D_pk[m,e] lam_pk - sum_s y[m,s,e,p] - w[m,e,p] - slack = 0     (beta_p[m,e])
        y[m,s,e,p] <= Dmax_p[m,e] * delta[m,s]
    Holding: an arrival at s used at e is held e-s periods; unused initial stock is held to H."""
    H = inst.horizon
    inf = solver.infinity()
    dem_rows: Dict[Tuple[int, str, int], object] = {}
    y_by_s: Dict[Tuple[str, int], list] = {}
    all_delta = {}
    for m in inst.parts:
        L, h, I0 = inst.lead_time[m], inst.holding_cost[m], inst.init_inventory.get(m, 0)
        delta = {s: (solver.BoolVar if integer else lambda n: solver.NumVar(0, 1, n))(f"d_{m}_{s}")
                 for s in range(L, H)}
        for s, d in delta.items():
            if (m, s) in delta_fix:
                d.SetBounds(delta_fix[m, s], delta_fix[m, s])
            all_delta[m, s] = d
        # branching rows on cumulative setup counts: sum_{s <= tau} delta[m,s] (<= or >=) k
        for (mm, tau, sense), k in (delta_fix.get("cum") or {}).items():
            if mm == m:
                expr = sum(d for s, d in delta.items() if s <= tau)
                solver.Add(expr <= k if sense == "<=" else expr >= k)
        obj += [inst.setup_cost[m] * d for d in delta.values()]
        stock_terms = []
        obj.append(h * I0 * H)                          # all initial stock held to H ...
        for p, pool in enumerate(pools):
            dmax = max_demand(inst, p, m)
            col_terms: Dict[int, list] = {}
            for k, c in enumerate(pool):
                for t, dq in c.demand[m].items():
                    col_terms.setdefault(t, []).append(dq * lam[p][k])
            for e, cap in dmax.items():
                supply = []
                w = solver.NumVar(0, inf, f"w_{m}_{e}_{p}")
                stock_terms.append(w)
                obj.append(-h * (H - e) * w)            # ... minus the periods after it is used
                supply.append(w)
                for s in range(L, e + 1):
                    y = solver.NumVar(0, cap, f"y_{m}_{s}_{e}_{p}")
                    solver.Add(y <= cap * delta[s])
                    obj.append((inst.unit_cost[m] + h * (e - s)) * y)
                    supply.append(y)
                    y_by_s.setdefault((m, s), []).append(y)
                sl = solver.NumVar(0, inf, f"e_{m}_{e}_{p}")
                obj.append(BIG * sl)
                dem_rows[p, m, e] = solver.Add(sum(col_terms.get(e, [])) - sum(supply) - sl == 0)
        if stock_terms:
            solver.Add(sum(stock_terms) <= I0)

    solver.Minimize(sum(obj))
    t0 = time.time()
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"master status {status}")
    out = {"obj": solver.Objective().Value(), "time": time.time() - t0,
           "lam": [[v.solution_value() for v in row] for row in lam],
           "slack": sum(v.solution_value() for v in solver.variables()
                        if v.name().startswith(("s_", "e_"))),
           "delta": {k: d.solution_value() for k, d in all_delta.items()}}
    if integer:
        out["bound"] = solver.Objective().BestBound()
        out["optimal"] = status == pywraplp.Solver.OPTIMAL
        orders = {m: [0] * H for m in inst.parts}
        for (m, s), ys in y_by_s.items():
            orders[m][s - inst.lead_time[m]] += int(round(sum(y.solution_value() for y in ys)))
        out["orders"] = orders
    else:
        out["mu"] = [c.dual_value() for c in conv]
        out["pi"] = {k: c.dual_value() for k, c in res_rows.items()}
        out["beta"] = [{(m, e): row.dual_value() for (pp, m, e), row in dem_rows.items() if pp == p}
                       for p in range(len(pools))]
        out["eta"] = {t: r.dual_value() for t, r in capex_rows.items()}
    return out


# ---------------------------------------------------------------------------
# Exact pricing (CP-SAT, one project)
# ---------------------------------------------------------------------------

def price_exact(inst: MultiProjectInstance, p: int, pi, beta, mu, time_limit_s: float = 30.0,
                num_workers: int = 8, hint: Optional[Column] = None,
                project_capacity: bool = True) -> Tuple[Column, float, bool]:
    """Min reduced-cost schedule of project p. Returns (column, lower bound on rc, proven optimal).
    project_capacity=False drops the project's own capacity constraints from pricing (the
    resource-unconstrained pricing that Kolter et al. 2025 show adds almost nothing to the bound)."""
    pr = inst.projects[p]
    acts = pr.activities
    beta = beta[p]
    model = cp_model.CpModel()
    x, coef = {}, {}
    for j, act in acts.items():
        for q, md in enumerate(act.modes):
            for tau in range(pr.release, pr.deadline - md.duration + 1):
                c = md.cost
                for r, k in md.demand_renewable.items():
                    c -= k * sum(pi.get((r, t), 0.0) for t in range(tau, tau + md.duration))
                for m, a in md.demand_parts.items():
                    c -= a * beta.get((m, tau), 0.0)
                x[j, q, tau] = model.NewBoolVar(f"x_{j}_{q}_{tau}")
                coef[j, q, tau] = c
        model.AddExactlyOne([v for (jj, _, _), v in x.items() if jj == j])

    s = {j: model.NewIntVar(pr.release, pr.deadline, f"s_{j}") for j in acts}
    e = {j: model.NewIntVar(pr.release, pr.deadline, f"e_{j}") for j in acts}
    for j, act in acts.items():
        model.Add(s[j] == sum(tau * v for (jj, q, tau), v in x.items() if jj == j))
        model.Add(e[j] == sum((tau + act.modes[q].duration) * v for (jj, q, tau), v in x.items() if jj == j))
    for i, j in pr.precedence:
        model.Add(s[j] >= e[i])

    # The project alone must also respect capacity (valid for every integer solution; tightens the bound)
    for r, cap in (inst.capacity.items() if project_capacity else ()):
        ivs, dem = [], []
        for (j, q, tau), v in x.items():
            md = acts[j].modes[q]
            k = md.demand_renewable.get(r, 0)
            if k:
                ivs.append(model.NewOptionalFixedSizeIntervalVar(tau, md.duration, v, f"iv_{j}_{q}_{tau}_{r}"))
                dem.append(k)
        if ivs:
            # time-varying capacity: block the unused part with fixed intervals
            maxcap = max(cap[pr.release:pr.deadline])
            for t in range(pr.release, pr.deadline):
                if cap[t] < maxcap:
                    ivs.append(model.NewFixedSizeIntervalVar(t, 1, f"blk_{r}_{t}"))
                    dem.append(maxcap - cap[t])
            model.AddCumulative(ivs, dem, maxcap)

    cmax = model.NewIntVar(pr.release, pr.deadline, "cmax")
    model.AddMaxEquality(cmax, list(e.values()))
    T = model.NewIntVar(0, max(0, pr.deadline - pr.due), "T")
    model.Add(T >= cmax - pr.due)

    model.Minimize(sum(int(round(SCALE * c)) * x[key] for key, c in coef.items())
                   + int(round(SCALE * pr.tardiness_weight)) * T)
    if hint is not None:
        for (j, q, tau), v in x.items():
            model.AddHint(v, int(hint.modes[j] == q and hint.starts[j] == tau))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"pricing for {pr.id}: {solver.StatusName(status)}")
    starts = {j: solver.Value(s[j]) for j in acts}
    modes = {j: next(q for q in range(len(acts[j].modes))
                     if any(solver.Value(v) for (jj, qq, _), v in x.items() if jj == j and qq == q))
             for j in acts}
    col = make_column(inst, p, starts, modes)
    # rounding: each of |J|+1 scaled terms is off by at most 0.5/SCALE
    bound = solver.BestObjectiveBound() / SCALE - mu[p] - (len(acts) + 1) * 0.5 / SCALE
    return col, bound, status == cp_model.OPTIMAL


# ---------------------------------------------------------------------------
# CG loop
# ---------------------------------------------------------------------------

Pricer = Callable[[MultiProjectInstance, int, dict, dict, list], List[Column]]


def exact_pricer(time_limit_s: float = 30.0, project_capacity: bool = True):
    def f(inst, p, pi, beta, mu, stats):
        col, lb, proven = price_exact(inst, p, pi, beta, mu, time_limit_s, project_capacity=project_capacity)
        stats["lb_parts"].append(lb if proven else None)
        return [col]
    return f


def order_candidates(inst: MultiProjectInstance, m, n: int = 5) -> List[List[int]]:
    """Candidate tool-order periods per project. With a capex budget the order period is part of
    the column, so pricing tries a few: the earliest, the latest, and the cheapest by the capex
    dual eta (the periods where the capital plan has room). Without a budget there is nothing to
    order and every column sits at period 0."""
    if inst.capex_budget is None:
        return [[0] for _ in inst.projects]
    eta = (m or {}).get("eta") or {}
    out = []
    for pr in inst.projects:
        span = pr.order_latest
        grid = sorted({0, span, span // 2, span // 4, (3 * span) // 4})
        cheap = sorted(range(span + 1), key=lambda t: -eta.get(t, 0.0))[:2]
        out.append(sorted(set(grid + cheap))[:n])
    return out


def shift_project(inst: MultiProjectInstance, p: int, order: int) -> MultiProjectInstance:
    """A copy of the instance where project p's window starts when its tool arrives. Without a
    capex budget there is no tool order, so the instance is returned untouched -- tool_lead is 0
    there and the arrival formula would otherwise reset every release to 0."""
    import copy
    if inst.capex_budget is None:
        return inst
    pr = inst.projects[p]
    arrival = order + pr.tool_lead
    if arrival == pr.release:
        return inst
    shifted = copy.copy(inst)
    shifted.projects = list(inst.projects)
    new_pr = copy.copy(pr)
    new_pr.release = arrival
    shifted.projects[p] = new_pr
    return shifted


def initial_columns(inst: MultiProjectInstance) -> List[List[Column]]:
    """One column per project: its own cheapest schedule ignoring the other projects (zero duals)."""
    P = len(inst.projects)
    zero_beta = [{} for _ in range(P)]
    return [[price_exact(inst, p, {}, zero_beta, [0.0] * P, time_limit_s=10)[0]] for p in range(P)]


def run_cg(inst: MultiProjectInstance, pricer=None, max_iters: int = 200, verbose: bool = True,
           integer_time_limit_s: float = 300.0, procurement: str = "fl") -> dict:
    pricer = pricer or exact_pricer()
    t_start = time.time()
    pools = initial_columns(inst)
    seen = [{c.key() for c in pool} for pool in pools]
    trace, best_lb = [], -float("inf")
    t_master = t_price = 0.0

    for it in range(1, max_iters + 1):
        m = solve_master(inst, pools, procurement=procurement)
        t_master += m["time"]
        stats = {"lb_parts": []}
        t0 = time.time()
        added, min_rcs = 0, []
        for p in range(len(inst.projects)):
            cols = pricer(inst, p, m["pi"], m["beta"], m["mu"], stats)
            rcs = [reduced_cost(c, m["pi"], m["beta"], m["mu"]) for c in cols]
            min_rcs.append(min(rcs))
            for c, rc in zip(cols, rcs):
                if rc < -RC_TOL and c.key() not in seen[p]:
                    pools[p].append(c)
                    seen[p].add(c.key())
                    added += 1
        t_price += time.time() - t0

        lb = None
        if stats["lb_parts"] and all(b is not None for b in stats["lb_parts"]):
            lb = m["obj"] + sum(min(0.0, b) for b in stats["lb_parts"])
            best_lb = max(best_lb, lb)
        trace.append({"iter": it, "rmp": m["obj"], "slack": m["slack"], "lb": lb,
                      "min_rc": min(min_rcs), "added": added})
        if verbose:
            print(f"it {it:3d}  RMP={m['obj']:12.3f}  slack={m['slack']:8.3f}  "
                  f"LB={'-' if lb is None else f'{lb:12.3f}'}  min rc={min(min_rcs):10.4f}  +{added}",
                  flush=True)
        if added == 0:
            break

    lp = solve_master(inst, pools, procurement=procurement)
    ip = solve_master(inst, pools, integer=True, time_limit_s=integer_time_limit_s, procurement=procurement)
    starts, modes = {}, {}
    for p, pool in enumerate(pools):
        k = max(range(len(pool)), key=lambda k: ip["lam"][p][k])
        starts.update(pool[k].starts)
        modes.update(pool[k].modes)
    check = evaluate_solution(inst, starts, modes, ip["orders"])
    return {"lp_obj": lp["obj"], "lp_slack": lp["slack"], "lower_bound": best_lb,
            "ip_obj": ip["obj"], "ip_optimal_over_pool": ip["optimal"], "ip_slack": ip["slack"],
            "check": check, "starts": starts, "modes": modes, "orders": ip["orders"],
            "columns": sum(len(pl) for pl in pools), "iters": len(trace), "trace": trace,
            "time_total": time.time() - t_start, "time_master": t_master, "time_pricing": t_price}
