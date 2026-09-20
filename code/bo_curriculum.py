"""
Bayesian optimization over the dual generator: find the dual regimes where the current pricing
policy is weakest, and train on them (an adversarial curriculum for column generation).

Objective at a parameter vector theta: draw duals from dual_generator, solve those pricing
problems both with the policy (anchored rollouts, greedy) and exactly with CP-SAT, and return
the mean gap (policy rc - exact rc) / dual scale. Larger gap = harder for the policy.

A Gaussian process (Matern 5/2, sklearn) models the objective and expected improvement picks
the next theta; the acquisition is maximized by random search over the box, which is enough in
seven dimensions. The exact solves make each evaluation cost a few seconds, which is why the
search is sample-efficient BO rather than a learned generator.
"""
from __future__ import annotations

import random
import time
from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from dual_generator import (BOUNDS, NAMES, PERTURB_BOUNDS, PERTURB_NAMES, make_problems,
                            make_problems_from_snapshot, random_theta, random_perturb_theta)
from mp_cg import price_exact, reduced_cost
from mp_pricing_env import INFEASIBLE_RC
import rl_pricing_trainer as T


@torch.no_grad()
def policy_rc_batch(policy, probs: List[dict], n_anchor: int, S: int, device) -> np.ndarray:
    """Best reduced cost per problem over its anchored rollouts, with the batched GPU pipeline.
    The episode return is exactly -(reduced cost)/scale, so no column rebuild is needed."""
    import rl_pricing_trainer_batch as TB
    g = torch.Generator(device=device)
    g.manual_seed(0)
    env, rewards, _, _ = TB.run_batch(policy, probs, n_anchor, S, greedy=True, device=device,
                                      rng_t=g, track_grad=False)
    rc = -(rewards.sum(1)) * env.scale
    rc = torch.where(env.dead, torch.full_like(rc, INFEASIBLE_RC), rc)
    return rc.view(len(probs), n_anchor).min(1).values.cpu().numpy()


@torch.no_grad()
def evaluate_theta(policy, source, theta, rng: random.Random, n_problems: int = 6,
                   exact_tl: float = 5.0, mode: str = "perturb") -> Tuple[float, List[dict]]:
    """Mean (policy rc - exact rc) / scale over a few pricing problems drawn from theta.

    mode "perturb": `source` is a list of (instance, pi, beta) CG snapshots, perturbed by theta
    mode "synth":   `source` is a list of instances and the duals are generated from theta
    """
    probs: List[dict] = []
    while len(probs) < n_problems:
        item = rng.choice(source)
        if mode == "perturb":
            inst, pi, beta = item
            p = rng.randrange(len(inst.projects))
            probs += make_problems_from_snapshot(inst, pi, beta, theta, rng, projects=[p])
        else:
            p = rng.randrange(len(item.projects))
            probs += make_problems(item, theta, rng, projects=[p])
    probs = probs[:n_problems]

    device = policy.device if hasattr(policy, "device") else next(policy.parameters()).device
    batched = policy.__class__.__name__.endswith("Batch")
    if batched:
        best_arr = policy_rc_batch(policy, probs, len(T.ANCHORS), 1, device)
        best = {i: float(v) for i, v in enumerate(best_arr)}
    else:
        _, meta, recs = T.run_rollouts(policy, probs, list(T.ANCHORS), S=1, greedy=True, rng=rng,
                                       track_grad=False)
        best = {}
        for (i, _), rc in zip(meta, recs):
            v = rc["info"].get("reduced_cost", INFEASIBLE_RC)
            best[i] = min(best.get(i, INFEASIBLE_RC), v)

    gaps = []
    for i, pr in enumerate(probs):
        col, _, _ = price_exact(pr["inst"], pr["p"], pr["pi"], pr["beta"], pr["mu"], exact_tl)
        ex = reduced_cost(col, pr["pi"], pr["beta"], pr["mu"])
        pr["exact_rc"] = ex
        env = T.make_env(pr)
        pol = best.get(i, INFEASIBLE_RC)
        gaps.append(min((pol - ex) / env.scale, 50.0))       # cap dead-end episodes
    return float(np.mean(gaps)), probs


def _fit_gp(X, y):
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(X, y)
    return gp


def _expected_improvement(gp, cand, best_y, xi=0.01):
    mu, sd = gp.predict(cand, return_std=True)
    sd = np.maximum(sd, 1e-9)
    z = (mu - best_y - xi) / sd
    return (mu - best_y - xi) * norm.cdf(z) + sd * norm.pdf(z)


def _unit(theta, bounds):
    return (np.asarray(theta) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _from_unit(u, bounds):
    return bounds[:, 0] + np.asarray(u) * (bounds[:, 1] - bounds[:, 0])


def bo_search(policy, source, rng: random.Random, n_init: int = 6, n_iter: int = 10,
              n_problems: int = 6, n_cand: int = 2000, verbose: bool = True, mode: str = "perturb"):
    """Returns (thetas, gaps, problems_by_theta) with the evaluated points, hardest last."""
    bounds = PERTURB_BOUNDS if mode == "perturb" else BOUNDS
    names = PERTURB_NAMES if mode == "perturb" else NAMES
    X, y, probs_all = [], [], []
    t0 = time.time()
    for k in range(n_init + n_iter):
        if k < n_init:
            theta = random_perturb_theta(rng) if mode == "perturb" else random_theta(rng)
        else:
            gp = _fit_gp(np.array(X), np.array(y))
            cand = np.random.rand(n_cand, len(bounds))
            ei = _expected_improvement(gp, cand, max(y))
            theta = _from_unit(cand[int(np.argmax(ei))], bounds)
        gap, probs = evaluate_theta(policy, source, theta, rng, n_problems, mode=mode)
        X.append(_unit(theta, bounds))
        y.append(gap)
        probs_all.append(probs)
        if verbose:
            tag = "init" if k < n_init else "bo"
            print(f"  [{tag} {k + 1:2d}/{n_init + n_iter}] gap={gap:8.3f}  "
                  + " ".join(f"{n}={v:.2f}" for n, v in zip(names, theta))
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
    order = np.argsort(y)
    return ([_from_unit(X[i], bounds) for i in order], [y[i] for i in order], [probs_all[i] for i in order])


def hard_problems(policy, source, rng: random.Random, n_keep: int = 3, **kw) -> List[dict]:
    """Run BO and return the pricing problems from the hardest thetas (with exact rc attached)."""
    thetas, gaps, probs = bo_search(policy, source, rng, **kw)
    out: List[dict] = []
    for pr in probs[-n_keep:]:
        out += pr
    return out
