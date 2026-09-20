"""
Verify exact_pricing.price_exact against brute-force enumeration on tiny instances.

For each instance: random early-window duals (alpha <= 0, as for the master's <=
rows) and random mu; enumerate every (start, mode) assignment that satisfies
precedence, renewable capacity, and end <= horizon; compare the true minimum
reduced cost with the exact pricer's. Also compares against the surrogate
CP-SAT pricer to show the gap it leaves.
"""
from __future__ import annotations

import io
import random
import contextlib
from itertools import product

from ColumnGenerationHeuristic import schedule_with_cpsat
from instance_generator import generate_random_instance
from exact_pricing import price_exact, _rc


def brute_force_min_rc(inst, alpha, mu):
    tasks = list(inst.activities)            # generator's DAG is index-ordered
    preds = {j: [i for i, k in inst.precedence if k == j] for j in tasks}
    H = inst.horizon
    best = [float("inf"), None]
    starts, modes = {}, {}
    usage = {r: [0] * H for r in inst.renewable_capacity}

    def rec(idx):
        if idx == len(tasks):
            rc = _rc(inst, starts, modes, alpha, mu)[0]
            if rc < best[0]:
                best[0], best[1] = rc, (dict(starts), dict(modes))
            return
        j = tasks[idx]
        for q, mode in enumerate(inst.activities[j].modes):
            lb = max([starts[i] + inst.activities[i].modes[modes[i]].duration for i in preds[j]], default=0)
            for t in range(lb, H - mode.duration + 1):
                span = range(t, t + mode.duration)
                if any(usage[r][u] + mode.demand_renewable.get(r, 0) > cap
                       for r, cap in inst.renewable_capacity.items() for u in span):
                    continue
                for r in inst.renewable_capacity:
                    for u in span:
                        usage[r][u] += mode.demand_renewable.get(r, 0)
                starts[j], modes[j] = t, q
                rec(idx + 1)
                for r in inst.renewable_capacity:
                    for u in span:
                        usage[r][u] -= mode.demand_renewable.get(r, 0)
        starts.pop(j, None)
        modes.pop(j, None)

    rec(0)
    return best[0], best[1]


def random_duals(inst, rng):
    alpha = {}
    for m in inst.parts:
        for t in range(inst.lead_time[m]):
            if rng.random() < 0.6:
                alpha[(m, t)] = -round(rng.uniform(0, 30), 3)
    return alpha, round(rng.uniform(-50, 50), 3)


def main(n_instances: int = 40):
    rng = random.Random(0)
    worst_gap = worst_model_err = 0.0
    n_match = n_proven = 0
    sur_gaps = []
    for k in range(n_instances):
        inst = generate_random_instance(num_tasks=4, num_parts=2, horizon=10, max_modes=2, seed=1000 + k)
        # make procurement matter: longer lead times, varied initial stock
        for m in inst.parts:
            inst.lead_time[m] = rng.randint(1, 6)
            inst.init_inventory[m] = rng.randint(0, 3)
        alpha, mu = random_duals(inst, rng)

        bf_rc, _ = brute_force_min_rc(inst, alpha, mu)
        col = price_exact(inst, alpha, mu, time_limit_s=30)
        with contextlib.redirect_stdout(io.StringIO()):
            s, q, _, _ = schedule_with_cpsat(inst, alpha, time_limit_s=5)
        sur_rc = _rc(inst, s, q, alpha, mu)[0]

        gap = col["reduced_cost"] - bf_rc
        worst_gap = max(worst_gap, abs(gap))
        worst_model_err = max(worst_model_err, abs(col["model_reduced_cost"] - col["reduced_cost"]))
        n_match += abs(gap) < 1e-6
        n_proven += col["proven_optimal"]
        sur_gaps.append(sur_rc - bf_rc)
        print(f"[{k:02d}] brute={bf_rc:10.4f}  exact={col['reduced_cost']:10.4f}  "
              f"(model {col['model_reduced_cost']:10.4f}, proven={col['proven_optimal']})  "
              f"surrogate={sur_rc:10.4f}")

    print(f"\nexact == brute force: {n_match}/{n_instances}   proven optimal: {n_proven}/{n_instances}"
          f"   worst |gap| = {worst_gap:.2e}")
    print(f"worst |model objective - true rc| = {worst_model_err:.2e}  (coefficient rounding only)")
    worse = [g for g in sur_gaps if g > 1e-6]
    print(f"surrogate CP-SAT worse than optimum on {len(worse)}/{n_instances} instances;"
          f" mean excess rc when worse = {sum(worse) / max(len(worse), 1):.3f}")


if __name__ == "__main__":
    main()
