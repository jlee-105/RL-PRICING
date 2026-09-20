"""
Parametric generator of master duals for training the pricing policy.

POMO-CG samples training duals at random and reports that results depend strongly on the
sampling parameters. Real duals here are not uniform at all: measured on CG trajectories
(data/pricing_val.pkl), 95-96 % of the resource duals pi are exactly zero and the non-zero ones
are concentrated and large (down to -10^4 while slack is still priced), while the material duals
beta sit in roughly [-22, -1.6], close to the material unit costs.

The generator therefore produces sparse, concentrated duals from seven parameters, low enough
dimensional for Bayesian optimization to search:

    0 log10 amplitude of pi          [-1, 4]
    1 fraction of periods priced     [0.01, 0.4]
    2 centre of the congested window [0, 1]      (fraction of the horizon)
    3 width of that window           [0.05, 0.6]
    4 tool price ratio               [0, 1]      (dedicated tool rows vs shared rows)
    5 level of beta                  [0.5, 25]
    6 slope of beta over time        [-1, 1]

mu is left at zero: it enters the reduced cost as a constant per project and cancels in the
gap (policy rc - exact rc) that we train and search against.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np

from multiproject_instance import MultiProjectInstance

BOUNDS = np.array([[-1.0, 4.0], [0.01, 0.4], [0.0, 1.0], [0.05, 0.6], [0.0, 1.0], [0.5, 25.0], [-1.0, 1.0]])
NAMES = ["log_amp", "frac", "centre", "width", "tool_ratio", "beta_level", "beta_slope"]


def random_theta(rng: random.Random) -> np.ndarray:
    return np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])


def sample_duals(inst: MultiProjectInstance, theta, rng: random.Random):
    """(pi, beta, mu) in the format the master returns and MPPricingEnv expects."""
    log_amp, frac, centre, width, tool_ratio, b_level, b_slope = theta
    H, amp = inst.horizon, 10.0 ** log_amp
    centre_t, width_t = centre * H, max(1.0, width * H)

    pi: Dict[Tuple[str, int], float] = {}
    for r in inst.capacity:
        ratio = tool_ratio if r.startswith("TOOL_") else 1.0
        if ratio <= 0:
            continue
        for t in range(H):
            # bump around the congested window, then thin out to the requested sparsity
            w = math.exp(-0.5 * ((t - centre_t) / width_t) ** 2)
            if rng.random() < frac * w:
                pi[r, t] = -amp * ratio * w * rng.uniform(0.5, 1.5)

    beta: List[Dict[Tuple[str, int], float]] = []
    for p in range(len(inst.projects)):
        bp: Dict[Tuple[str, int], float] = {}
        for m in inst.parts:
            base = b_level * inst.unit_cost[m] / 5.0
            for t in range(H):
                bp[m, t] = -base * (1.0 + b_slope * (t / max(H - 1, 1))) * rng.uniform(0.8, 1.2)
        beta.append(bp)
    mu = [0.0] * len(inst.projects)
    return pi, beta, mu


# ---------------------------------------------------------------------------------------------
# Perturbation of real duals. Duals drawn from scratch turned out to be much easier for the
# policy than real ones (gap 0.01-0.17 against 0.31 on CG snapshots): the master's duals carry
# structure -- which periods are actually binding -- that an independent sample does not
# reproduce. So the search space is a perturbation of a real snapshot instead.
#
#     0 log10 scale of pi        [-1.0, 1.5]   overall price level
#     1 keep fraction of pi      [0.2, 1.0]    sparsify the priced periods
#     2 spread                   [0.0, 0.5]    copy a price onto neighbouring periods
#     3 time shift               [-0.3, 0.3]   move the congested periods (fraction of horizon)
#     4 noise                    [0.0, 1.0]    multiplicative lognormal noise
#     5 log10 scale of beta      [-0.5, 1.0]
#     6 tilt of beta over time   [-1.0, 1.0]
PERTURB_BOUNDS = np.array([[-1.0, 1.5], [0.2, 1.0], [0.0, 0.5], [-0.3, 0.3],
                           [0.0, 1.0], [-0.5, 1.0], [-1.0, 1.0]])
PERTURB_NAMES = ["pi_scale", "keep", "spread", "shift", "noise", "beta_scale", "beta_tilt"]


def random_perturb_theta(rng: random.Random) -> np.ndarray:
    return np.array([rng.uniform(lo, hi) for lo, hi in PERTURB_BOUNDS])


def perturb_duals(inst: MultiProjectInstance, pi, beta, theta, rng: random.Random):
    """Perturb a real (pi, beta) snapshot; mu is dropped since it cancels in the gap."""
    pi_scale, keep, spread, shift, noise, beta_scale, tilt = theta
    H = inst.horizon
    shift_t = int(round(shift * H))
    amp, bamp = 10.0 ** pi_scale, 10.0 ** beta_scale

    def jitter():
        return math.exp(rng.gauss(0.0, noise)) if noise > 0 else 1.0

    new_pi: Dict[Tuple[str, int], float] = {}
    for (r, t), v in pi.items():
        if v == 0.0 or rng.random() > keep:
            continue
        t2 = min(max(t + shift_t, 0), H - 1)
        new_pi[r, t2] = new_pi.get((r, t2), 0.0) + v * amp * jitter()
        if spread > 0:                      # bleed the price onto the neighbouring periods
            for d in (-1, 1):
                if rng.random() < spread and 0 <= t2 + d < H:
                    new_pi[r, t2 + d] = new_pi.get((r, t2 + d), 0.0) + v * amp * spread * jitter()

    new_beta: List[Dict[Tuple[str, int], float]] = []
    for bp in beta:
        out: Dict[Tuple[str, int], float] = {}
        for (m, t), v in bp.items():
            t2 = min(max(t + shift_t, 0), H - 1)
            out[m, t2] = v * bamp * (1.0 + tilt * (t / max(H - 1, 1))) * jitter()
        new_beta.append(out)
    return new_pi, new_beta, [0.0] * len(inst.projects)


def make_problems_from_snapshot(inst: MultiProjectInstance, pi, beta, theta, rng: random.Random,
                                projects=None) -> List[dict]:
    """Pricing problems under a perturbed real snapshot."""
    npi, nbeta, mu = perturb_duals(inst, pi, beta, theta, rng)
    ps = range(len(inst.projects)) if projects is None else projects
    return [{"inst": inst, "p": p, "pi": npi, "beta": nbeta, "mu": mu, "theta": np.asarray(theta)} for p in ps]


def make_problems(inst: MultiProjectInstance, theta, rng: random.Random, projects=None) -> List[dict]:
    """Pricing problems (one per project) under duals drawn from theta."""
    pi, beta, mu = sample_duals(inst, theta, rng)
    ps = range(len(inst.projects)) if projects is None else projects
    return [{"inst": inst, "p": p, "pi": pi, "beta": beta, "mu": mu, "theta": np.asarray(theta)} for p in ps]
