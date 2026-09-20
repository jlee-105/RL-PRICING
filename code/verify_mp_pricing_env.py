"""
Checks for MPPricingEnv on real CG duals:
  (1) the undiscounted return of every completed episode equals -(reduced cost) of its column
      (also with a POMO anchor placed in reset);
  (2) every completed column is feasible for the project alone (window, precedence, own capacity);
  (3) how far random and "cheapest-next" rollouts are from the exact pricer (sanity, not a result).
"""
from __future__ import annotations

import io
import contextlib
import random

import numpy as np

from multiproject_instance import generate_multiproject_instance
from mp_cg import initial_columns, solve_master, price_exact
from mp_pricing_env import MPPricingEnv, INFEASIBLE_RC


def feasible_alone(inst, p, col):
    pr = inst.projects[p]
    dur = {j: pr.activities[j].modes[col.modes[j]].duration for j in pr.activities}
    if any(col.starts[j] < pr.release or col.starts[j] + dur[j] > pr.deadline for j in pr.activities):
        return False
    if any(col.starts[j] < col.starts[i] + dur[i] for i, j in pr.precedence):
        return False
    for r, ut in col.usage.items():
        if any(u > inst.capacity[r][t] for t, u in ut.items()):
            return False
    return True


def rollout(env, policy, anchor=None, rng=None):
    env.reset(anchor_task_idx=anchor)
    total, done, info = 0.0, False, {}
    while not done:
        mask = env.get_action_mask().cpu().numpy()
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return total, {"reduced_cost": INFEASIBLE_RC}
        a = policy(env, valid, rng)
        _, r, done, info = env.step(int(a))
        total += r
    return total, info


def random_policy(env, valid, rng):
    return rng.choice(list(valid))


def cheapest_policy(env, valid, rng):
    # myopic: lowest immediate dual cost among valid actions
    best, best_c = None, np.inf
    for a in valid:
        i, rem = divmod(int(a), env.Q * env.D)
        q, d = divmod(rem, env.D)
        c = env.coef[i][q][env.cand[i, q, d]]
        if c < best_c:
            best, best_c = a, c
    return best


def main():
    rng = random.Random(0)
    worst_err, n_done, n_infeasible, n_bad = 0.0, 0, 0, 0
    rows = []
    for tool in ("none", "qual"):
        for seed in range(3):
            inst = generate_multiproject_instance(num_projects=4, tasks_per_project=8, tool_unary=tool, seed=seed)
            pools = initial_columns(inst)
            for _ in range(3):                                   # a few CG iterations -> real duals
                m = solve_master(inst, pools)
                for p in range(len(pools)):
                    pools[p].append(price_exact(inst, p, m["pi"], m["beta"], m["mu"])[0])
            m = solve_master(inst, pools)
            for p in range(len(inst.projects)):
                env = MPPricingEnv(inst, p, m["pi"], m["beta"], m["mu"])
                exact_rc = price_exact(inst, p, m["pi"], m["beta"], m["mu"])[1]
                best = {"random": np.inf, "cheapest": np.inf}
                for k in range(40):
                    anchor = k % env.n if k % 2 else None
                    for name, pol in (("random", random_policy), ("cheapest", cheapest_policy)):
                        ret, info = rollout(env, pol, anchor, rng)
                        rc = info.get("reduced_cost", INFEASIBLE_RC)
                        if rc >= INFEASIBLE_RC:
                            n_infeasible += 1
                            continue
                        n_done += 1
                        worst_err = max(worst_err, abs(ret + rc))
                        n_bad += not feasible_alone(inst, p, info["column"])
                        best[name] = min(best[name], rc)
                rows.append((tool, seed, p, exact_rc, best["random"], best["cheapest"]))

    print(f"completed episodes: {n_done}, dead ends: {n_infeasible}")
    print(f"(1) max |return + reduced cost| = {worst_err:.2e}")
    print(f"(2) infeasible columns: {n_bad}")
    print("(3) best rc over 40 rollouts vs exact min rc")
    for tool, seed, p, ex, rnd, ch in rows:
        print(f"  tool={tool:4s} seed={seed} P{p}: exact={ex:10.3f}  random={rnd:10.3f}  cheapest={ch:10.3f}")


if __name__ == "__main__":
    main()
