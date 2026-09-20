"""
Check that BatchPricingEnv.obs() and PricingGNNBatch reproduce the scalar graph_obs and
PricingGNN: same node features, same per-action features, same logits (after loading the same
weights). Then time a full anchored-rollout sweep both ways.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from mp_pricing_env import MPPricingEnv
from mp_pricing_env_batch import BatchPricingEnv
from pricing_gnn import PricingGNN, PricingGNNBatch
from verify_batch_env import load_problems


def compare(problems, device="cpu", steps=4, seed=0):
    rng = random.Random(seed)
    scal = [MPPricingEnv(p["inst"], p["p"], p["pi"], p["beta"], p["mu"]) for p in problems]
    for e in scal:
        e.reset()
    batch = BatchPricingEnv(problems, device=device)

    net = PricingGNN(hidden=64)
    bnet = PricingGNNBatch(hidden=64).to(device)
    bnet.load_state_dict(net.state_dict())
    net.eval(), bnet.eval()

    worst = {"x": 0.0, "feats": 0.0, "g": 0.0, "logits": 0.0}
    for _ in range(steps):
        bobs = batch.obs()
        with torch.no_grad():
            blog = bnet(bobs, n_tasks=batch.n, n_res=batch.R).cpu()
        sobs = [e.graph_obs() for e in scal]
        with torch.no_grad():
            slog = net(sobs)
        for b, e in enumerate(scal):
            k = e.n + len(e.res) + len(e.inst.parts)
            xb = torch.cat([bobs["x"][b, :e.n], bobs["x"][b, batch.n:batch.n + len(e.res)],
                            bobs["x"][b, batch.n + batch.R:batch.n + batch.R + len(e.inst.parts)]]).cpu()
            worst["x"] = max(worst["x"], float((xb - torch.as_tensor(sobs[b]["x"])).abs().max()))
            worst["g"] = max(worst["g"], float((bobs["g"][b].cpu() - torch.as_tensor(sobs[b]["global"])).abs().max()))
            # per-action features and logits, in the scalar env's candidate order
            fb = bobs["feats"][b].cpu().reshape(batch.n * batch.Q * batch.D, -1)
            lb_ = blog[b]
            sf = torch.as_tensor(sobs[b]["action_feats"])
            for j, a in enumerate(sobs[b]["actions"]):
                worst["feats"] = max(worst["feats"], float((fb[a] - sf[j]).abs().max()))
                worst["logits"] = max(worst["logits"], float(abs(lb_[a] - slog[b][j])))
        # advance both with the same random valid actions
        acts = torch.zeros(len(scal), dtype=torch.long)
        for b, e in enumerate(scal):
            if e.placed.all() or e.dead:
                continue
            valid = list(zip(*np.nonzero(e.cand >= 0)))
            i, q, d = valid[rng.randrange(len(valid))]
            acts[b] = int((i * e.Q + q) * e.D + d)
        batch.step(acts.to(device))
        for b, e in enumerate(scal):
            if not (e.placed.all() or e.dead):
                e.step(int(acts[b]))
    return worst


def timing(problems, device, repeats=3):
    batch = BatchPricingEnv(problems, device=device)
    net = PricingGNNBatch(hidden=64).to(device).eval()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            batch.reset()
            while not bool(batch.done.all()):
                logits = net(batch.obs(), n_tasks=batch.n, n_res=batch.R)
                # finished rollouts have every action masked; give them a dummy row, step() ignores them
                logits = torch.where(batch.done[:, None], torch.zeros_like(logits), logits)
                a = torch.distributions.Categorical(logits=logits).sample()
                batch.step(a)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=16)
    args = ap.parse_args()
    rng = random.Random(0)
    probs = load_problems(args.problems, rng)
    w = compare(probs)
    print("max |difference| vs the scalar implementation:")
    for k, v in w.items():
        print(f"  {k:7s} {v:.3e}")

    for n in (64, 256, 1024):
        ps = load_problems(n, random.Random(1))
        for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            print(f"  policy+env sweep, {n:5d} rollouts on {dev}: {1000 * timing(ps, dev):8.1f} ms")


if __name__ == "__main__":
    main()
