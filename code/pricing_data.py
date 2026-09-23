"""
Pricing problems for training and evaluating learned pricers, taken from real CG trajectories.

For each generated instance, run column generation with the exact pricer and record the master
duals at every iteration. Each (instance, iteration, project) is one pricing problem; the exact
pricer's column and proven bound are stored with it, so evaluation needs no further solves.

    python pricing_data.py --split train --n 40 --seed0 0
    python pricing_data.py --split val   --n 10 --seed0 1000
"""
from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path

from multiproject_instance import generate_multiproject_instance
from mp_cg import (initial_columns, solve_master, price_exact, reduced_cost, make_column,
                   order_candidates, shift_project, RC_TOL)

DATA_DIR = Path(__file__).parent / "data"


def cg_snapshots(inst, max_iters: int = 30, pricing_tl: float = 5.0, log=None,
                 pricing_workers: int = 8):
    """One pricing problem per (iteration, project, candidate tool-order period).

    With a capex budget the order period is part of the column, so CG prices each project over a
    few candidate periods (mp_cg.order_candidates) on the window that the tool's arrival opens.
    Recording the same candidates here keeps the training problems in the distribution the learned
    pricer meets inside cg_runner. Two reduced costs are stored per candidate:
      rc      on the shifted window, without capex and without eta -- what the env and the policy
              see, so it is the reference for the training gap;
      rc_full what the master uses to accept a column (capex in the cost, minus eta_o * capex).
    They coincide when there is no tool purchase, and the older single-dict format still loads."""
    pools = initial_columns(inst)
    seen = [{c.key() for c in pl} for pl in pools]
    snaps = []
    for it in range(max_iters):
        m = solve_master(inst, pools)
        eta = m.get("eta") or {}
        cands = order_candidates(inst, m)
        exact = []
        added = 0
        for p in range(len(inst.projects)):
            recs = []
            for o in cands[p]:
                col, lb, proven = price_exact(shift_project(inst, p, o), p,
                                              m["pi"], m["beta"], m["mu"], pricing_tl,
                                              num_workers=pricing_workers)
                full = make_column(inst, p, col.starts, col.modes, order=o)
                rc_full = reduced_cost(full, m["pi"], m["beta"], m["mu"], eta)
                rc = reduced_cost(full, m["pi"], m["beta"], m["mu"]) - full.capex
                recs.append({"order": o, "rc": rc, "rc_full": rc_full, "bound": lb, "proven": proven})
                if rc_full < -RC_TOL and full.key() not in seen[p]:
                    pools[p].append(full)
                    seen[p].add(full.key())
                    added += 1
            exact.append(recs)
        snaps.append({"iter": it, "pi": m["pi"], "beta": m["beta"], "mu": m["mu"], "eta": eta,
                      "rmp": m["obj"], "slack": m["slack"], "exact": exact})
        if log:
            flat = [r for recs in exact for r in recs]
            log(f"    it {it:2d} rmp {m['obj']:12.2f} slack {m['slack']:7.2f} +{added} "
                f"proven {sum(r['proven'] for r in flat)}/{len(flat)}")
        if added == 0:
            break
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--plan_offset", type=float, default=1.0)
    ap.add_argument("--tool_purchase", action="store_true")
    ap.add_argument("--pricing_workers", type=int, default=8,
                    help="CP-SAT threads per exact pricing solve. Use 1 when running "
                         "several shards at once.")
    ap.add_argument("--out", default=None,
                    help="output filename inside data/; defaults to pricing_<split>.pkl. "
                         "Set it to shard generation across processes.")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    out_name = args.out or f"pricing_{args.split}.pkl"
    rng = random.Random(args.seed0)
    out = []
    t0 = time.time()
    for k in range(args.n):
        seed = args.seed0 + k
        cfg = dict(num_projects=rng.randint(3, 6), tasks_per_project=rng.choice([8, 10]),
                   num_templates=rng.randint(2, 3), tool_unary=rng.choice(["qual", "qual", "none"]),
                   plan_offset=args.plan_offset, tool_purchase=args.tool_purchase, seed=seed)
        inst = generate_multiproject_instance(**cfg)
        snaps = cg_snapshots(inst, log=lambda s: print(s, flush=True),
                             pricing_workers=args.pricing_workers)
        out.append({"cfg": cfg, "inst": inst, "snaps": snaps})
        n_prob = sum(len(recs) for s in snaps for recs in s["exact"])
        print(f"[{k + 1}/{args.n}] seed={seed} P={cfg['num_projects']} J={cfg['tasks_per_project']} "
              f"tool={cfg['tool_unary']} iters={len(snaps)} problems={n_prob} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
        with open(DATA_DIR / out_name, "wb") as f:   # save as we go
            pickle.dump(out, f)
    print("done", flush=True)


if __name__ == "__main__":
    main()
