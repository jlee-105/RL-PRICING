"""
Tensorized trainer: the same anchored multi-rollout REINFORCE as rl_pricing_trainer, but every
rollout of every problem advances together inside BatchPricingEnv, so a whole update is a handful
of GPU kernels instead of B Python environments.

Layout: a batch is P problems x N anchors, flattened to B = P*N rollouts in that order, which
makes the shared baseline a reshape. Anchors pick the first S actions by rule (all tensorized);
the policy acts afterwards. Advantage of step t = return-to-go minus the leave-one-out mean over
the other rollouts of the same problem, exactly as in the scalar trainer.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from mp_pricing_env_batch import BatchPricingEnv
from pricing_gnn import PricingGNNBatch
from rl_pricing_trainer import load_problems, DEAD_PENALTY

ANCHOR_NAMES = ["myopic", "nodelay", "cheap_mode", "fast_mode", "critical", "late", "random", "rl"]
BIG = 1e9


def make_generator(device, seed: int = 0) -> torch.Generator:
    """torch.Generator exists for cpu and cuda but not for every backend (mps rejects it), so fall
    back to a cpu generator there; anchor_actions moves the draw onto the env's device."""
    try:
        g = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        g = torch.Generator()
    g.manual_seed(seed)
    return g


def anchor_actions(env: BatchPricingEnv, anchor_ids: torch.Tensor, rng_t: torch.Generator) -> torch.Tensor:
    """Action per rollout for its anchor rule; rollouts whose anchor is "rl" get -1."""
    B, n, Q, D = env.B, env.n, env.Q, env.D
    s = env._starts
    valid = s >= 0
    c_raw = env.coef.gather(3, s.clamp(min=0).clamp(max=env.W - 1))            # [B,n,Q,D]
    # scale for the lexicographic keys, taken over valid entries only (using the sentinel here
    # would make a valid key larger than an invalid one and let argmin pick an invalid action)
    span = torch.where(valid, c_raw, torch.zeros_like(c_raw)).abs().amax(dim=(1, 2, 3), keepdim=True) + 1.0
    c = torch.where(valid, c_raw, torch.full_like(c_raw, BIG))

    d_idx = torch.arange(D, device=env.device).view(1, 1, 1, D).expand_as(c)
    dmax = torch.where(valid, d_idx, torch.full_like(d_idx, -1)).amax(dim=(1, 2, 3), keepdim=True)
    tail = env.tail[:, :, None, None].expand_as(c)
    tmax = torch.where(valid, tail, torch.full_like(tail, -BIG)).amax(dim=(1, 2, 3), keepdim=True)

    keys = {
        "myopic": c,
        "nodelay": c + (d_idx != 0).float() * BIG,
        "cheap_mode": env.mode_cost[..., None].expand_as(c) * span + c,
        "fast_mode": env.dur[..., None].expand_as(c).float() * span + c,
        "critical": c + (tail < tmax).float() * BIG,
        "late": c + (d_idx != dmax).float() * BIG,
        # drawn on the generator's device (cpu when the backend has no generator) then moved
        "random": torch.rand(c.shape, device=rng_t.device, generator=rng_t).to(env.device),
    }
    out = torch.full((B,), -1, dtype=torch.long, device=env.device)
    for k, name in enumerate(ANCHOR_NAMES):
        if name == "rl":
            continue
        sel = anchor_ids == k
        if not sel.any():
            continue
        key = torch.where(valid, keys[name], torch.full_like(keys[name], float("inf")))
        out = torch.where(sel, key.reshape(B, -1).argmin(-1), out)
    return out


def run_batch(policy, problems: List[dict], n_anchor: int, S: int, greedy: bool, device,
              rng_t: torch.Generator, track_grad: bool = True):
    """P problems x n_anchor rollouts. Returns (env, rewards [B,T], logps [B,T] or None, live [B,T])."""
    env = BatchPricingEnv(problems, device=device, repeat=n_anchor)
    anchor_ids = torch.arange(env.B, device=device) % n_anchor
    rewards, logps, lives = [], [], []
    if bool(env.dead.any()):                      # stuck before the first action
        rewards.append(torch.where(env.dead, torch.full((env.B,), -DEAD_PENALTY, device=device),
                                   torch.zeros(env.B, device=device)))
        logps.append(torch.zeros(env.B, device=device))
        lives.append(torch.zeros(env.B, dtype=torch.bool, device=device))
    step = 0
    while not bool(env.done.all()):
        live = ~env.done.clone()             # clone: env.step updates done in place, autograd needs a copy
        a = torch.zeros(env.B, dtype=torch.long, device=device)
        lp = torch.zeros(env.B, device=device)
        use_anchor = (step < S) & (anchor_ids < n_anchor - 1)          # last anchor slot is the policy
        if use_anchor.any():
            a = torch.where(use_anchor, anchor_actions(env, anchor_ids, rng_t), a)
        learn = live & ~use_anchor
        if learn.any():
            logits = policy(env.obs(), n_tasks=env.n, n_res=env.R)
            logits = torch.where(~live[:, None], torch.zeros_like(logits), logits)
            dist = torch.distributions.Categorical(logits=logits)
            pick = logits.argmax(-1) if greedy else dist.sample()
            a = torch.where(learn, pick, a)
            if track_grad:
                lp = dist.log_prob(a)
        r, _, _ = env.step(torch.where(live, a, torch.zeros_like(a)))
        r = torch.where(env.dead & live, torch.full_like(r, -DEAD_PENALTY), r / env.scale)
        rewards.append(torch.where(live, r, torch.zeros_like(r)))
        logps.append(torch.where(learn, lp, torch.zeros_like(lp)))
        lives.append(learn)
        step += 1
    return env, torch.stack(rewards, 1), torch.stack(logps, 1), torch.stack(lives, 1)


def loss_from(rewards, logps, lives, n_anchor: int):
    """REINFORCE with return-to-go and a leave-one-out baseline across each problem's rollouts."""
    B, T = rewards.shape
    P = B // n_anchor
    G = rewards.flip(1).cumsum(1).flip(1).view(P, n_anchor, T)
    tot = G[:, :, 0]
    norm = tot.std(dim=1, keepdim=True).clamp(min=0.1)
    loo = (G.sum(1, keepdim=True) - G) / max(n_anchor - 1, 1)
    adv = ((G - loo) / norm[..., None]).view(B, T)
    mask = lives.float()
    terms = -(logps * adv * mask)
    return terms.sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(policy, problems: List[dict], n_anchor: int, S: int, device, rng_t, chunk: int = 64):
    """Gap to the exact pricer on real CG duals. "rl" = anchored rollouts with the policy acting
    after step S; "heur" = the same rollouts with the anchors acting all the way (no learning)."""
    policy.eval()
    out = {}
    for name, s_steps in (("rl", S), ("heur", 10_000)):
        gaps, neg, tot_neg = [], 0, 0
        for c0 in range(0, len(problems), chunk):
            probs = problems[c0:c0 + chunk]
            env, rewards, _, _ = run_batch(policy, probs, n_anchor, s_steps, True, device, rng_t,
                                           track_grad=False)
            rc = -(rewards.sum(1)) * env.scale
            rc = torch.where(env.dead, torch.full_like(rc, 1e6), rc).view(len(probs), n_anchor)
            if name == "heur":
                rc = rc[:, :n_anchor - 1]                    # drop the un-anchored (policy) rollout
            best = rc.min(1).values.cpu().numpy()
            scale = env.scale.view(len(probs), n_anchor)[:, 0].cpu().numpy()
            ex = np.array([p["exact_rc"] for p in probs])
            gaps += list(np.clip((best - ex) / scale, 0, None))
            neg += int(((best < -1e-6) & (ex < -1e-6)).sum())
            tot_neg += int((ex < -1e-6).sum())
        out[name] = {"gap": float(np.mean(gaps)), "neg_found": neg / max(tot_neg, 1)}
    policy.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16, help="problems per update")
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="rlb")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_problems", type=int, default=256)
    # BO curriculum: search the dual-perturbation space for the regimes where the policy is
    # weakest and mix those pricing problems into the batch
    ap.add_argument("--curriculum", choices=["none", "bo"], default="none")
    ap.add_argument("--bo_every", type=int, default=100)
    ap.add_argument("--bo_mix", type=float, default=0.5)
    ap.add_argument("--bo_init", type=int, default=5)
    ap.add_argument("--bo_iter", type=int, default=5)
    ap.add_argument("--bo_problems", type=int, default=8)
    ap.add_argument("--bo_keep", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rng_t = make_generator(args.device, args.seed)

    train = load_problems("train")
    val = load_problems("val")
    rng.shuffle(val)
    val = val[:args.eval_problems]
    print(f"train problems {len(train)}, val {len(val)}, device {args.device}", flush=True)
    policy = PricingGNNBatch(hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    n_anchor = len(ANCHOR_NAMES)
    ckpt = Path(__file__).parent / f"pricer_{args.tag}.pt"

    best_gap = float("inf")
    snapshots, hard_pool = [], []
    if args.curriculum == "bo":
        import bo_curriculum as BO
        from rl_pricing_trainer import load_snapshots
        snapshots = load_snapshots("train")
        print(f"BO curriculum on, {len(snapshots)} dual snapshots to perturb", flush=True)

    t0 = time.time()
    for u in range(1, args.updates + 1):
        if args.curriculum == "bo" and (u == 1 or u % args.bo_every == 0):
            tb = time.time()
            hard_pool = BO.hard_problems(policy, snapshots, rng, n_keep=args.bo_keep,
                                         n_init=args.bo_init, n_iter=args.bo_iter,
                                         n_problems=args.bo_problems, verbose=False)
            print(f"  BO at update {u}: {len(hard_pool)} hard problems in {time.time() - tb:.0f}s", flush=True)
        n_hard = min(int(args.batch * args.bo_mix), len(hard_pool)) if hard_pool else 0
        batch = rng.sample(hard_pool, n_hard) + rng.sample(train, args.batch - n_hard)
        env, rewards, logps, lives = run_batch(policy, batch, n_anchor, args.S, False, args.device, rng_t)
        loss = loss_from(rewards, logps, lives, n_anchor)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if u % args.log_every == 0:
            print(f"u {u:5d}  loss {loss.item():8.4f}  mean return {rewards.sum(1).mean().item():8.3f}  "
                  f"{(time.time() - t0) / u:6.3f} s/update", flush=True)
        if u % args.eval_every == 0:
            ev = evaluate(policy, val, n_anchor, args.S, args.device, rng_t)
            print(f"  EVAL u {u}: " + "  ".join(f"{k}: gap {v['gap']:.3f} neg {v['neg_found']:.2f}"
                                                for k, v in ev.items()), flush=True)
            if ev["rl"]["gap"] < best_gap:
                best_gap = ev["rl"]["gap"]
                torch.save({"state": policy.state_dict(), "args": vars(args), "update": u,
                            "eval": ev}, ckpt)
                print(f"  saved {ckpt.name} (best rl gap {best_gap:.3f})", flush=True)
    last = ckpt.with_name(f"{ckpt.stem}_last.pt")
    torch.save({"state": policy.state_dict(), "args": vars(args), "update": args.updates}, last)
    print(f"best rl gap {best_gap:.3f} in {ckpt.name}; final weights in {last.name}", flush=True)


if __name__ == "__main__":
    main()
