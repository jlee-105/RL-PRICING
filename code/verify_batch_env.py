"""
Equivalence check: BatchPricingEnv must reproduce MPPricingEnv exactly.

Drives both with the same random actions and compares, at every step, the candidate start of
each (task, mode, delay), the step reward, and the terminal / dead-end flags. Also times both.
"""
from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from mp_pricing_env import MPPricingEnv, INFEASIBLE_RC
from mp_pricing_env_batch import BatchPricingEnv

DATA = Path(__file__).parent / "data"


def load_problems(n: int, rng: random.Random, split="val", skip_first=3):
    with open(DATA / f"pricing_{split}.pkl", "rb") as f:
        data = pickle.load(f)
    probs = []
    for d in data:
        for s in d["snaps"][skip_first:]:
            for p in range(len(d["inst"].projects)):
                probs.append({"inst": d["inst"], "p": p, "pi": s["pi"], "beta": s["beta"], "mu": s["mu"]})
    rng.shuffle(probs)
    return probs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rng = random.Random(0)
    probs = load_problems(args.problems, rng)

    scal = [MPPricingEnv(p["inst"], p["p"], p["pi"], p["beta"], p["mu"]) for p in probs]
    for e in scal:
        e.reset()
    batch = BatchPricingEnv(probs, device=args.device)

    worst_cand, worst_rew = 0, 0.0
    flag_mismatch = 0
    steps = 0
    done = [False] * len(scal)
    while not all(done):
        starts = batch._starts.cpu().numpy()
        actions = torch.zeros(len(scal), dtype=torch.long)
        for b, e in enumerate(scal):
            if done[b]:
                continue
            # candidate tables must agree exactly
            sc = np.full((e.n, e.Q, e.D), -1)
            sc[:e.n] = e.cand
            worst_cand = max(worst_cand, int(np.abs(sc - starts[b, :e.n]).max()))
            valid = list(zip(*np.nonzero(e.cand >= 0)))
            i, q, d = valid[rng.randrange(len(valid))]
            actions[b] = int((i * e.Q + q) * e.D + d)
        rew, bdone, bdead = batch.step(actions.to(batch.device))
        rew = rew.cpu().numpy()
        for b, e in enumerate(scal):
            if done[b]:
                continue
            _, r, d_, info = e.step(int(actions[b]))
            is_dead = info.get("reduced_cost", 0.0) >= INFEASIBLE_RC
            r_cmp = r + (INFEASIBLE_RC / e.scale if is_dead else 0.0)   # batch env leaves the penalty out
            worst_rew = max(worst_rew, abs(r_cmp - float(rew[b])))
            flag_mismatch += int(d_ != bool(bdone[b].item())) + int(is_dead != bool(bdead[b].item()))
            done[b] = d_
        steps += 1

    print(f"steps {steps}, problems {len(scal)}")
    print(f"max |candidate start difference| = {worst_cand}")
    print(f"max |step reward difference|     = {worst_rew:.3e}")
    print(f"terminal/dead flag mismatches    = {flag_mismatch}")

    # timing: same rollouts, random policy
    for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        env = BatchPricingEnv(probs, device=dev)
        t0 = time.time()
        for _ in range(3):
            env.reset()
            while not bool(env.done.all()):
                mask = env.action_mask()
                probs_u = mask.float() + 1e-9
                a = torch.multinomial(probs_u, 1).squeeze(1)
                env.step(a)
        print(f"batch env, 3 rollout sweeps of {len(probs)} envs on {dev}: {time.time() - t0:.2f}s")

    t0 = time.time()
    for _ in range(3):
        for e in scal:
            e.reset()
            while True:
                valid = list(zip(*np.nonzero(e.cand >= 0)))
                if not valid:
                    break
                i, q, d = valid[rng.randrange(len(valid))]
                _, _, dn, _ = e.step(int((i * e.Q + q) * e.D + d))
                if dn:
                    break
    print(f"scalar envs, same work: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
