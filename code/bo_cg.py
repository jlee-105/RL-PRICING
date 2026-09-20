"""
Bayesian optimization of the training distribution, scored by the objective we actually report:
the cost of the solution column generation produces.

    theta  ->  training data (real CG duals mixed with duals perturbed by theta)
           ->  short fine-tune of the pricing policy
           ->  run CG on fixed validation instances
           ->  master LP after a fixed number of iterations, relative to the base policy
                                                            <- BO minimizes this

This replaces the earlier proxy objective (make the pricing gap as large as possible). Difficulty
on its own says nothing about whether training on those duals helps; a regime can be hard because
it is unrealistic, or hard for every pricer. Scoring by the downstream CG cost removes that gap
between what BO searches for and what the paper measures.

Noise control: every theta is scored on the same validation instances with the same seeds, and
costs are reported relative to the base policy on those instances, so instances of different size
do not dominate the mean.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

import cg_runner as R
from dual_generator import PERTURB_BOUNDS, PERTURB_NAMES, make_problems_from_snapshot, random_perturb_theta
from multiproject_instance import generate_multiproject_instance
from pricing_gnn import PricingGNNBatch
from rl_pricing_trainer import load_problems, load_snapshots
import rl_pricing_trainer_batch as TB

HERE = Path(__file__).parent


# ----------------------------------------------------------------- scoring
def cg_costs(policy, instances, S: int, time_limit: float, ip_time: float, seed: int = 0,
             metric: str = "lp", iters: int = 15) -> np.ndarray:
    """CG score per instance. "lp": master LP after a fixed number of CG iterations -- the direct
    measure of how well the pricer drives CG. "cost": the final plan after the integer master and
    the repair, which is what the paper reports but four times noisier across training seeds
    (0.8 % vs 0.2 % spread), so it is too noisy to steer BO with."""
    out = []
    for k, inst in enumerate(instances):
        r = R.run(inst, "rl", time_limit_s=time_limit, policy=policy, S=S, seed=seed + k,
                  max_iters=iters if metric == "lp" else 500,
                  ip_time_s=1.0 if metric == "lp" else ip_time)
        out.append(r["lp"] if metric == "lp" else
                   (r["cost"] if r["cost"] is not None else float("inf")))
    return np.array(out)


def fine_tune(policy, train_problems, hard_pool, updates: int, batch: int, mix: float, lr: float,
              S: int, device, seed: int):
    rng = random.Random(seed)
    rng_t = torch.Generator(device=device)
    rng_t.manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n_anchor = len(TB.ANCHOR_NAMES)
    for _ in range(updates):
        n_hard = min(int(batch * mix), len(hard_pool)) if hard_pool else 0
        probs = rng.sample(hard_pool, n_hard) + rng.sample(train_problems, batch - n_hard)
        _, rewards, logps, lives = TB.run_batch(policy, probs, n_anchor, S, False, device, rng_t)
        loss = TB.loss_from(rewards, logps, lives, n_anchor)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return policy


def hard_pool_from_theta(theta, snapshots, rng: random.Random, n_problems: int) -> List[dict]:
    pool = []
    while len(pool) < n_problems:
        inst, pi, beta = rng.choice(snapshots)
        p = rng.randrange(len(inst.projects))
        pool += make_problems_from_snapshot(inst, pi, beta, theta, rng, projects=[p])
    return pool[:n_problems]


# ----------------------------------------------------------------- BO
def _fit_gp(X, y):
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(X.shape[1]), nu=2.5) + WhiteKernel(1e-2)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(X, y)
    return gp


def _ei_min(gp, cand, best_y, xi=0.01):
    """Expected improvement for minimization."""
    mu, sd = gp.predict(cand, return_std=True)
    sd = np.maximum(sd, 1e-9)
    z = (best_y - xi - mu) / sd
    return (best_y - xi - mu) * norm.cdf(z) + sd * norm.pdf(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(HERE / "pricer_rl_base.pt"))
    ap.add_argument("--n_init", type=int, default=5)
    ap.add_argument("--n_iter", type=int, default=7)
    ap.add_argument("--updates", type=int, default=200, help="fine-tune updates per theta")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pool", type=int, default=64, help="perturbed problems drawn per theta")
    ap.add_argument("--val_P", type=int, default=10)
    ap.add_argument("--val_n", type=int, default=2)
    ap.add_argument("--cg_limit", type=float, default=120.0)
    ap.add_argument("--ip_limit", type=float, default=20.0)
    ap.add_argument("--metric", choices=["lp", "cost"], default="lp")
    ap.add_argument("--cg_iters", type=int, default=15, help="CG iterations for the lp metric")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "logs" / "bo_cg.jsonl"))
    # search a subset of the generator parameters, with the rest fixed at realistic values:
    # the first 12-evaluation run showed only a few of the seven mattered, and 7 dimensions with
    # that budget never got past exploration
    ap.add_argument("--search", default="pi_scale,keep,beta_scale,beta_tilt")
    ap.add_argument("--fixed", default="spread=0.1,shift=0.0,noise=0.3")
    ap.add_argument("--bounds", default="pi_scale:-0.5:1.5,keep:0.4:1.0,beta_scale:-0.5:0.5,beta_tilt:-1:1")
    args = ap.parse_args()

    search = [n.strip() for n in args.search.split(",") if n.strip()]
    dims = [PERTURB_NAMES.index(n) for n in search]
    theta_fixed = np.array([(PERTURB_BOUNDS[i, 0] + PERTURB_BOUNDS[i, 1]) / 2 for i in range(len(PERTURB_NAMES))])
    for item in filter(None, args.fixed.split(",")):
        k, v = item.split("=")
        theta_fixed[PERTURB_NAMES.index(k.strip())] = float(v)
    bnd = {n: (PERTURB_BOUNDS[PERTURB_NAMES.index(n), 0], PERTURB_BOUNDS[PERTURB_NAMES.index(n), 1])
           for n in search}
    for item in filter(None, args.bounds.split(",")):
        k, a, b = item.split(":")
        bnd[k.strip()] = (float(a), float(b))
    lo = np.array([bnd[n][0] for n in search])
    hi = np.array([bnd[n][1] for n in search])

    def full_theta(sub):
        t = theta_fixed.copy()
        t[dims] = sub
        return t

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    ck = torch.load(args.base, map_location="cpu")
    S, hidden = ck["args"]["S"], ck["args"]["hidden"]
    train = load_problems("train")
    snaps = load_snapshots("train")
    val = [generate_multiproject_instance(num_projects=args.val_P, tasks_per_project=10,
                                          tool_unary="qual", seed=6000 + i) for i in range(args.val_n)]

    def fresh_policy():
        pol = PricingGNNBatch(hidden=hidden).to(args.device)
        pol.load_state_dict(ck["state"])
        return pol

    t0 = time.time()
    score = lambda pol: cg_costs(pol, val, S, args.cg_limit, args.ip_limit,
                                 metric=args.metric, iters=args.cg_iters)
    base_costs = score(fresh_policy())
    print(f"base policy CG {args.metric} {np.round(base_costs, 2)}  ({time.time() - t0:.0f}s)", flush=True)

    def objective(theta) -> float:
        """Mean CG cost relative to the base policy after fine-tuning on theta's distribution."""
        pol = fresh_policy()
        pool = hard_pool_from_theta(theta, snaps, random.Random(args.seed), args.pool)
        fine_tune(pol, train, pool, args.updates, args.batch, args.mix, args.lr, S, args.device, args.seed)
        return float(np.mean(score(pol) / base_costs))

    X, y = [], []
    to_unit = lambda t: (np.asarray(t) - lo) / (hi - lo)
    from_unit = lambda u: lo + np.asarray(u) * (hi - lo)

    # reference point: fine-tune on real duals only (mix = 0), same budget
    pol0 = fresh_policy()
    fine_tune(pol0, train, [], args.updates, args.batch, 0.0, args.lr, S, args.device, args.seed)
    ref = float(np.mean(score(pol0) / base_costs))
    print(f"fine-tune on real duals only: relative cost {ref:.4f}", flush=True)

    for k in range(args.n_init + args.n_iter):
        if k < args.n_init:
            sub = np.array([rng.uniform(a, b) for a, b in zip(lo, hi)])
        else:
            gp = _fit_gp(np.array(X), np.array(y))
            cand = np.random.rand(4000, len(lo))
            sub = from_unit(cand[int(np.argmax(_ei_min(gp, cand, min(y))))])
        theta = full_theta(sub)
        t1 = time.time()
        v = objective(theta)
        X.append(to_unit(sub))
        y.append(v)
        row = {"k": k, "mode": "init" if k < args.n_init else "bo", "relative_cost": v,
               "theta": {n: round(float(t), 3) for n, t in zip(PERTURB_NAMES, theta)},
               "searched": {n: round(float(v2), 3) for n, v2 in zip(search, sub)},
               "seconds": round(time.time() - t1, 1)}
        print(json.dumps(row), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    best = int(np.argmin(y))
    print(f"\nbest relative cost {y[best]:.4f} (real-duals-only reference {ref:.4f}, base 1.0)")
    print("best searched dims:", {n: round(float(t), 3) for n, t in zip(search, from_unit(X[best]))}, flush=True)
    rnd = [v for v in y[:args.n_init]]
    print(f"random-search best over the {args.n_init} init points: {min(rnd):.4f}", flush=True)


if __name__ == "__main__":
    main()
