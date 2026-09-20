"""
Large-scale comparison under a common time limit (the paper's main experiment, first version).

Methods, all scored by the same evaluator (schedule + optimal Wagner-Whitin procurement):
  MIP        compact model (PDT precedence, big-M procurement) solved by SCIP to the limit
  CG-exact   column generation, exact CP-SAT pricing
  CG-heur    column generation, learning-free anchor heuristics as pricer
  CG-RL      column generation, learned pricer (skipped if no checkpoint)
Test instances use seeds 5000+ (disjoint from the pricing train/val data).

    python exp_large_scale.py --P 5 10 20 40 --seeds 3 --ckpt pricer_rl.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from multiproject_instance import generate_multiproject_instance
from mp_compact import solve_compact
from pricing_gnn import PricingGNNBatch
import cg_runner as R

HERE = Path(__file__).parent


def load_policy(path, device="cpu"):
    """The batched policy shares the scalar one's weight layout, so either checkpoint loads here;
    cg_runner then prices all (project, anchor, order) cases of an iteration in one rollout."""
    if not path or not Path(path).exists():
        return None, None
    ck = torch.load(path, map_location="cpu")
    pol = PricingGNNBatch(hidden=ck["args"]["hidden"]).to(device)
    pol.load_state_dict(ck["state"])
    pol.eval()
    return pol, ck["args"]["S"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--ckpt", default=str(HERE / "pricer_rl_base.pt"))
    ap.add_argument("--plan_offset", type=float, default=1.0,
                    help="<1 makes material lead times bind the schedule (0.25-0.5 recommended)")
    ap.add_argument("--tool_purchase", action="store_true",
                    help="the tool itself is ordered with a lead time, under a capex budget")
    ap.add_argument("--methods", nargs="+", default=["MIP", "CG-exact", "CG-heur", "CG-RL"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(HERE / "logs" / "large_scale.jsonl"))
    args = ap.parse_args()
    policy, S = load_policy(args.ckpt, args.device)
    torch.set_num_threads(4)

    for P in args.P:
        limit = 300.0 if P <= 10 else 600.0
        for k in range(args.seeds):
            seed = 5000 + 100 * P + k
            inst = generate_multiproject_instance(num_projects=P, tasks_per_project=args.tasks,
                                                  tool_unary="qual", plan_offset=args.plan_offset,
                                                  tool_purchase=args.tool_purchase, seed=seed)
            for meth in args.methods:
                t0 = time.time()
                if meth == "MIP":
                    r = solve_compact(inst, "agg", "bigm", mode="mip", time_limit_s=limit)
                    row = {"cost": r.get("incumbent"), "bound": r["bound"], "status": r["status"]}
                elif meth == "CG-RL" and policy is None:
                    continue
                else:
                    pricer = {"CG-exact": "exact", "CG-heur": "heur", "CG-RL": "rl"}[meth]
                    r = R.run(inst, pricer, time_limit_s=limit, policy=policy, S=S or 1, ip_time_s=60)
                    row = {k2: r[k2] for k2 in ("cost", "lp", "iters", "cols", "t_cg", "t_pricing", "feasible")}
                row.update(P=P, seed=seed, method=meth, time=round(time.time() - t0, 1), limit=limit)
                print(json.dumps(row), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
