"""
Anchored multi-rollout training for the pricing policy, following the
lithography model's training (Lithography/ha_pomo_trainer.py; paper Alg. 1): for every pricing
problem, N rollouts run in parallel; the first S decisions of rollout n follow anchor
heuristic h_n (one rollout is un-anchored), and the policy acts afterwards.

Differences from the lithography trainer, both decided with the user on 2026-09-15:
  * step rewards: the pricing return decomposes exactly into per-step dual costs, so the
    advantage of step t uses the return-to-go G_t, not the episodic return;
  * the baseline of step t is the leave-one-out mean of G_t over the other rollouts of the same
    problem (a shared baseline in the POMO sense, made unbiased by excluding the rollout).
Rewards are divided by the problem's dual scale so problems with very different duals mix.
Anchor rollouts also give N distinct columns per pricing call at inference.
"""
from __future__ import annotations

import argparse
import math
import pickle
import random
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from mp_cg import shift_project
from mp_pricing_env import MPPricingEnv, INFEASIBLE_RC
from pricing_gnn import PricingGNN

DATA_DIR = Path(__file__).parent / "data"
DEAD_PENALTY = 10.0          # dead end, in units of the problem's dual scale


# ---------------------------------------------------------------- anchor heuristics
def _cands(env):
    return env.candidates()                          # (action, i, q, d, s)


def a_myopic(env, rng):
    return min(_cands(env), key=lambda c: env.coef[c[1]][c[2]][c[4]])[0]


def a_nodelay(env, rng):
    cs = [c for c in _cands(env) if c[3] == 0] or _cands(env)
    return min(cs, key=lambda c: env.coef[c[1]][c[2]][c[4]])[0]


def a_cheap_mode(env, rng):
    cs = [c for c in _cands(env) if c[3] == 0] or _cands(env)
    return min(cs, key=lambda c: (env.acts[c[1]].modes[c[2]].cost, env.coef[c[1]][c[2]][c[4]]))[0]


def a_fast_mode(env, rng):
    cs = [c for c in _cands(env) if c[3] == 0] or _cands(env)
    return min(cs, key=lambda c: (env.acts[c[1]].modes[c[2]].duration, env.coef[c[1]][c[2]][c[4]]))[0]


def a_critical(env, rng):
    cs = _cands(env)
    top = max(env.tail[c[1]] for c in cs)
    return min([c for c in cs if env.tail[c[1]] == top], key=lambda c: env.coef[c[1]][c[2]][c[4]])[0]


def a_late(env, rng):
    cs = _cands(env)
    dmax = max(c[3] for c in cs)
    return min([c for c in cs if c[3] == dmax], key=lambda c: env.coef[c[1]][c[2]][c[4]])[0]


def a_random(env, rng):
    return rng.choice(_cands(env))[0]


ANCHORS: Dict[str, Callable] = {"myopic": a_myopic, "nodelay": a_nodelay, "cheap_mode": a_cheap_mode,
                                "fast_mode": a_fast_mode, "critical": a_critical, "late": a_late,
                                "random": a_random, "rl": None}


def heuristic_rollout(env, anchor, rng):
    """Run one anchor heuristic for the whole episode (a learning-free pricer)."""
    env.reset()
    done, info = False, {}
    while not done:
        if not env.candidates():
            return INFEASIBLE_RC, None
        _, _, done, info = env.step(ANCHORS[anchor](env, rng))
    return info["reduced_cost"], info.get("column")


# ---------------------------------------------------------------- data
def load_snapshots(split: str, skip_first: int = 3):
    """(instance, pi, beta) per CG iteration, for the BO curriculum to perturb. The first
    iterations are skipped: while the master still needs slack their duals are ~1e4 and
    unlike anything the policy sees once CG has settled."""
    with open(DATA_DIR / f"pricing_{split}.pkl", "rb") as f:
        data = pickle.load(f)
    return [(d["inst"], s["pi"], s["beta"]) for d in data for s in d["snaps"][skip_first:]]


def load_problems(split: str):
    """One problem per (iteration, project, candidate tool-order period). The instance is shifted
    so the project's window starts when its tool arrives, exactly as cg_runner prices it; without
    a tool purchase there is a single candidate and the instance is untouched. Data written before
    tool ordering stored one dict per project instead of a list, and still loads."""
    with open(DATA_DIR / f"pricing_{split}.pkl", "rb") as f:
        data = pickle.load(f)
    probs = []
    for d in data:
        for s in d["snaps"]:
            for p in range(len(d["inst"].projects)):
                recs = s["exact"][p]
                if isinstance(recs, dict):
                    recs = [dict(recs, order=0)]
                for r in recs:
                    probs.append({"inst": shift_project(d["inst"], p, r["order"]), "p": p,
                                  "pi": s["pi"], "beta": s["beta"], "mu": s["mu"],
                                  "exact_rc": r["rc"], "iter": s["iter"], "order": r["order"]})
    return probs


def make_env(prob, device="cpu"):
    return MPPricingEnv(prob["inst"], prob["p"], prob["pi"], prob["beta"], prob["mu"], device=device)


# ---------------------------------------------------------------- rollouts
def run_rollouts(policy: PricingGNN, probs, anchors: List[str], S: int, greedy: bool, rng,
                 track_grad: bool = True):
    """Lockstep rollouts: len(probs) x len(anchors) environments. Returns per-env records."""
    envs, meta = [], []
    for pi_, pr in enumerate(probs):
        for an in anchors:
            envs.append(make_env(pr, policy.device))
            meta.append((pi_, an))
    recs = [{"logp": [], "r": [], "info": None, "done": False} for _ in envs]
    step = 0
    while True:
        active = [k for k, rc in enumerate(recs) if not rc["done"]]
        if not active:
            break
        # anchored steps need no forward pass
        learn = [k for k in active if not (step < S and ANCHORS[meta[k][1]] is not None)]
        logits = {}
        if learn:
            obs = [envs[k].graph_obs() for k in learn]
            with torch.set_grad_enabled(track_grad):
                outs = policy(obs)
            logits = {k: (o, lg) for k, o, lg in zip(learn, obs, outs)}
        for k in active:
            env, rc = envs[k], recs[k]
            if not env.candidates():
                rc["done"], rc["info"] = True, {"reduced_cost": INFEASIBLE_RC}
                rc["r"].append(-DEAD_PENALTY)
                rc["logp"].append(None)
                continue
            if k in logits:
                o, lg = logits[k]
                dist = torch.distributions.Categorical(logits=lg)
                j = int(lg.argmax()) if greedy else int(dist.sample())
                rc["logp"].append(dist.log_prob(torch.tensor(j, device=lg.device)) if track_grad else None)
                a = o["actions"][j]
            else:
                a = ANCHORS[meta[k][1]](env, rng)
                rc["logp"].append(None)
            _, r, done, info = env.step(a)
            dead = done and info.get("reduced_cost", 0.0) >= INFEASIBLE_RC
            rc["r"].append(-DEAD_PENALTY if dead else r / env.scale)
            if done:
                rc["done"], rc["info"] = True, info
        step += 1
    return envs, meta, recs


def rl_loss(meta, recs, n_probs):
    """REINFORCE with return-to-go and a leave-one-out per-step baseline over each problem's rollouts."""
    groups: Dict[int, List[int]] = {}
    for k, (pi_, _) in enumerate(meta):
        groups.setdefault(pi_, []).append(k)
    terms = []
    for ks in groups.values():
        G = {k: np.cumsum(recs[k]["r"][::-1])[::-1] for k in ks}
        tot = np.array([G[k][0] for k in ks])
        norm = max(tot.std(), 0.1)
        for k in ks:
            for t, lp in enumerate(recs[k]["logp"]):
                if lp is None:
                    continue
                others = [G[o][t] for o in ks if o != k and len(G[o]) > t]
                b = float(np.mean(others)) if others else 0.0
                terms.append(-lp * float((G[k][t] - b) / norm))
    if not terms:
        return None
    return torch.stack(terms).sum() / max(1, len(terms))


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(policy, probs, anchors, S, rng, max_probs=400):
    """Best rc over the anchored rollouts (greedy policy after anchors) vs exact and vs heuristics."""
    policy.eval()
    probs = probs[:max_probs]
    res = {"rl": [], "greedy": [], "myopic": [], "best_heur": [], "exact": []}
    B = 32
    for b0 in range(0, len(probs), B):
        chunk = probs[b0:b0 + B]
        _, meta, recs = run_rollouts(policy, chunk, anchors, S, greedy=True, rng=rng, track_grad=False)
        per = {}
        for (pi_, an), rc in zip(meta, recs):
            per.setdefault(pi_, {})[an] = rc["info"]["reduced_cost"]
        for pi_, pr in enumerate(chunk):
            env = make_env(pr)
            scale = env.scale
            heur = {an: heuristic_rollout(env, an, rng)[0] for an in ANCHORS if ANCHORS[an] is not None}
            res["rl"].append(min(per[pi_].values()) / scale)
            res["greedy"].append(per[pi_].get("rl", INFEASIBLE_RC) / scale)
            res["myopic"].append(heur["myopic"] / scale)
            res["best_heur"].append(min(heur.values()) / scale)
            res["exact"].append(pr["exact_rc"] / scale)
    policy.train()
    ex = np.array(res["exact"])
    out = {}
    for k in ("rl", "greedy", "myopic", "best_heur"):
        v = np.array(res[k])
        ok = v < INFEASIBLE_RC / 1e4           # finite results only for the gap
        gap = v - ex
        out[k] = {"gap": float(np.mean(np.clip(gap[ok], 0, None))) if ok.any() else float("nan"),
                  "neg_found": float(np.mean((v < -1e-6)[ex < -1e-6])) if (ex < -1e-6).any() else float("nan"),
                  "match": float(np.mean(gap < 1e-6 * np.maximum(1, abs(ex))))}
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="rl")
    # BO curriculum: every --bo_every updates, search the dual-perturbation space for the regimes
    # where the policy is weakest and mix those problems into the batch (--bo_mix of it).
    ap.add_argument("--device", default="cpu", help="cpu or cuda; the GNN is small, the win is batching")
    ap.add_argument("--curriculum", choices=["none", "bo"], default="none")
    ap.add_argument("--bo_every", type=int, default=200)
    ap.add_argument("--bo_mix", type=float, default=0.5)
    ap.add_argument("--bo_init", type=int, default=5)
    ap.add_argument("--bo_iter", type=int, default=5)
    ap.add_argument("--bo_problems", type=int, default=6)
    ap.add_argument("--bo_keep", type=int, default=3)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(4)

    train = load_problems("train")
    val = load_problems("val")
    rng.shuffle(val)
    print(f"train problems {len(train)}, val problems {len(val)}", flush=True)

    policy = PricingGNN(hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    anchors = list(ANCHORS)
    ckpt = Path(__file__).parent / f"pricer_{args.tag}.pt"
    best = math.inf
    t0 = time.time()

    snapshots, hard_pool = [], []
    if args.curriculum == "bo":
        import bo_curriculum as BO
        snapshots = load_snapshots("train")
        print(f"BO curriculum on, {len(snapshots)} dual snapshots to perturb", flush=True)

    for u in range(1, args.updates + 1):
        if args.curriculum == "bo" and (u == 1 or u % args.bo_every == 0):
            print(f"  BO search at update {u} (policy so far)", flush=True)
            hard_pool = BO.hard_problems(policy, snapshots, rng, n_keep=args.bo_keep,
                                         n_init=args.bo_init, n_iter=args.bo_iter,
                                         n_problems=args.bo_problems)
            print(f"  curriculum pool: {len(hard_pool)} problems", flush=True)
        n_hard = min(int(args.batch * args.bo_mix), len(hard_pool)) if hard_pool else 0
        batch = rng.sample(hard_pool, n_hard) + rng.sample(train, args.batch - n_hard)
        _, meta, recs = run_rollouts(policy, batch, anchors, args.S, greedy=False, rng=rng)
        loss = rl_loss(meta, recs, len(batch))
        if loss is not None:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        if u % 20 == 0:
            rets = [sum(rc["r"]) for rc in recs]
            print(f"u {u:5d}  loss {0 if loss is None else loss.item():8.4f}  mean return {np.mean(rets):8.3f}  "
                  f"elapsed {time.time() - t0:6.0f}s", flush=True)
        if u % args.eval_every == 0:
            ev = evaluate(policy, val, anchors, args.S, rng)
            print(f"  EVAL u {u}: " + "  ".join(f"{k}: gap {v['gap']:.3f} neg {v['neg_found']:.2f} "
                                                 f"match {v['match']:.2f}" for k, v in ev.items()), flush=True)
            if ev["rl"]["gap"] < best:
                best = ev["rl"]["gap"]
                torch.save({"state": policy.state_dict(), "args": vars(args), "update": u, "eval": ev}, ckpt)
                print(f"  saved {ckpt.name} (rl gap {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
