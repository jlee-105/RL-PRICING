"""
Exact pricing oracle for the CG master in ColumnGenerationHeuristic.py.

Minimizes the true reduced cost of a column,

    rc(s, q) = schedule_cost(s, q) - sum_{m,t} alpha_{mt} D_{mt}(s, q) - mu,

where schedule_cost = mode costs + MAKESPAN_WEIGHT * Cmax + greedy-JIT procurement
cost (plan_orders_greedy). The surrogate CP-SAT pricer (schedule_with_cpsat)
omits the procurement term; this one models it exactly.

Procurement cost of plan_orders_greedy in closed form, per part m with
cumulative demand S_t, initial inventory I0, lead time L:

    E_t      = max(0, S_t - I0)                  cumulative shortage
    short_t  = E_t - E_{t-1}                     quantity ordered to arrive at t
    cost_m   = sum_{t >= L} [ c * short_t + f * 1(short_t > 0) ]
             + h * sum_t max(0, I0 - S_t)

Shortages at t < L cannot be ordered and are free here (the master's slack
prices them). Time-indexed start binaries make D_{mt} linear. CP-SAT needs
integer coefficients, so every cost coefficient is scaled by SCALE and
rounded; the returned reduced cost is always recomputed in floating point
with schedule_cost.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from ColumnGenerationHeuristic import Instance, schedule_cost, MAKESPAN_WEIGHT

SCALE = 10_000


def _rc(inst: Instance, starts, modes, alpha_duals, mu):
    cost, cmax, D = schedule_cost(inst, starts, modes)
    alpha_dot = sum(alpha_duals.get((m, t), 0.0) * d
                    for m in inst.parts for t, d in enumerate(D[m]) if d)
    return cost - alpha_dot - mu, cost, cmax, D


def price_exact(inst: Instance,
                alpha_duals: Dict[Tuple[str, int], float],
                mu: float,
                time_limit_s: float = 10.0,
                num_workers: int = 8,
                hint: Optional[Tuple[Dict[str, int], Dict[str, int]]] = None) -> dict:
    """Solve the pricing problem exactly (up to coefficient rounding).

    Returns a column dict in the same format as POMOTrainer.generate_columns
    plus solver info: `proven_optimal`, and `rc_lower_bound`, a valid lower
    bound on the minimum reduced cost (exact only when proven_optimal).
    """
    H = inst.horizon
    acts = inst.activities
    model = cp_model.CpModel()

    # x[j][q][t] = 1 iff task j starts at t in mode q (and finishes by H)
    x: Dict[str, List[Dict[int, cp_model.IntVar]]] = {}
    start: Dict[str, cp_model.LinearExpr] = {}
    dur: Dict[str, cp_model.LinearExpr] = {}
    is_mode: Dict[Tuple[str, int], cp_model.LinearExpr] = {}
    for j, act in acts.items():
        x[j] = []
        for q, mode in enumerate(act.modes):
            x[j].append({t: model.NewBoolVar(f"x_{j}_{q}_{t}") for t in range(H - mode.duration + 1)})
        all_x = [v for xq in x[j] for v in xq.values()]
        if not all_x:
            raise RuntimeError(f"Horizon too short; activity {j} doesn't fit.")
        model.AddExactlyOne(all_x)
        start[j] = sum(t * v for xq in x[j] for t, v in xq.items())
        for q in range(act.num_modes):
            is_mode[(j, q)] = sum(x[j][q].values())
        dur[j] = sum(mode.duration * is_mode[(j, q)] for q, mode in enumerate(act.modes))

    s_var = {j: model.NewIntVar(0, H, f"s_{j}") for j in acts}
    for j in acts:
        model.Add(s_var[j] == start[j])

    for i, j in inst.precedence:
        model.Add(s_var[j] >= s_var[i] + dur[i])

    # Renewable resources: one optional interval per (task, mode)
    for r, cap in inst.renewable_capacity.items():
        ivs, dem = [], []
        for j, act in acts.items():
            for q, mode in enumerate(act.modes):
                k = mode.demand_renewable.get(r, 0)
                if k:
                    pres = model.NewBoolVar(f"p_{j}_{q}_{r}")
                    model.Add(pres == is_mode[(j, q)])
                    ivs.append(model.NewOptionalFixedSizeIntervalVar(s_var[j], mode.duration, pres, f"iv_{j}_{q}_{r}"))
                    dem.append(k)
        if ivs:
            model.AddCumulative(ivs, dem, cap)

    def sc(v: float) -> int:
        return int(round(v * SCALE))

    obj = []

    # Mode costs
    for j, act in acts.items():
        for q, mode in enumerate(act.modes):
            if mode.cost:
                obj.append(sc(mode.cost) * is_mode[(j, q)])

    # Makespan (weight is 0 by default, kept for consistency with schedule_cost)
    if MAKESPAN_WEIGHT:
        cmax = model.NewIntVar(0, H, "Cmax")
        for j in acts:
            model.Add(cmax >= s_var[j] + dur[j])
        obj.append(sc(MAKESPAN_WEIGHT) * cmax)

    # Part demand per period, then the closed-form greedy-JIT cost and the dual term
    for m in inst.parts:
        users = [(j, q, a) for j, act in acts.items() for q, mode in enumerate(act.modes)
                 if (a := mode.demand_parts.get(m, 0))]
        if not users:
            # unused part: its initial stock is held all horizon (a constant, kept so bounds are exact)
            obj.append(sc(inst.holding_cost[m]) * inst.init_inventory.get(m, 0) * (H + 1))
            continue
        D_t = [sum(a * x[j][q][t] for j, q, a in users if t in x[j][q]) for t in range(H + 1)]
        total_max = sum(max(md.demand_parts.get(m, 0) for md in acts[j].modes) for j in acts)

        for t in range(H + 1):
            coef = alpha_duals.get((m, t), 0.0)
            if coef:
                obj.append(-sc(coef) * D_t[t])

        I0, L = inst.init_inventory.get(m, 0), inst.lead_time[m]
        c, f, h = inst.unit_cost[m], inst.setup_cost[m], inst.holding_cost[m]
        S = 0
        E_prev = None
        for t in range(H + 1):
            S = S + D_t[t]
            E = model.NewIntVar(0, max(total_max - I0, 0), f"E_{m}_{t}")
            model.AddMaxEquality(E, [0, S - I0])
            if h and I0 > 0:
                G = model.NewIntVar(0, I0, f"G_{m}_{t}")       # on-hand inventory
                model.Add(G >= I0 - S)                           # tight at optimum since h > 0
                obj.append(sc(h) * G)
            if t >= L:
                short = E - (E_prev if E_prev is not None else 0)
                obj.append(sc(c) * short)
                if f:
                    y = model.NewBoolVar(f"setup_{m}_{t}")
                    model.Add(short <= total_max * y)
                    obj.append(sc(f) * y)
            E_prev = E

    model.Minimize(sum(obj))

    if hint is not None:
        hs, hq = hint
        for j in acts:
            for q in range(acts[j].num_modes):
                for t, v in x[j][q].items():
                    model.AddHint(v, int(hq.get(j, 0) == q and hs.get(j) == t))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Exact pricing found no schedule (status={solver.StatusName(status)}).")

    starts = {j: int(solver.Value(s_var[j])) for j in acts}
    modes = {j: next(q for q in range(acts[j].num_modes)
                     if any(solver.Value(v) for v in x[j][q].values())) for j in acts}
    rc, cost, cmax_val, D = _rc(inst, starts, modes, alpha_duals, mu)
    return {
        "starts": starts, "modes": modes, "cost": cost, "makespan": cmax_val, "D": D,
        "reduced_cost": rc,
        "model_reduced_cost": solver.ObjectiveValue() / SCALE - mu,
        "rc_lower_bound": solver.BestObjectiveBound() / SCALE - mu,
        "proven_optimal": status == cp_model.OPTIMAL,
        "wall_time": solver.WallTime(),
    }
