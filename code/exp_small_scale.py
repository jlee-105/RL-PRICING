"""
Small-scale comparison where the compact MIP can prove optimality, with generous limits so
column generation converges instead of being cut off.

For every instance:
  MIP       compact model (disaggregated precedence + facility-location procurement), SCIP,
            long limit -> optimum (status and bound tell whether it was proven)
  CG-exact  CG with exact CP-SAT pricing, long pricing and CG limits -> converged LP
  CG-heur   CG with the anchor heuristics
  CG-RL     CG with the learned pricer
Reported: cost vs the MIP optimum, the converged LP bound, iterations, and time.

    python exp_small_scale.py --P 3 --tasks 8 --seeds 3
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tool", default="qual")
    ap.add_argument("--plan_offset", type=float, default=1.0,
                    help="<1 makes material lead times bind the schedule (0.25-0.5 recommended)")
    ap.add_argument("--tool_purchase", action="store_true",
                    help="the tool itself is ordered with a lead time, under a capex budget")
    ap.add_argument("--mip_limit", type=float, default=1800.0)
    ap.add_argument("--cg_limit", type=float, default=1800.0)
    ap.add_argument("--pricing_limit", type=float, default=60.0)
    ap.add_argument("--ip_limit", type=float, default=300.0)
    ap.add_argument("--ckpt", default=str(HERE / "pricer_rl_base.pt"))
    ap.add_argument("--methods", nargs="+", default=["CG-exact", "CG-heur"],
                    help="CG variants to run; CG-RL is left out until the BO curriculum is in")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(HERE / "logs" / "small_scale.jsonl"))
    args = ap.parse_args()
    torch.set_num_threads(4)

    policy, S = None, 1
    if "CG-RL" in args.methods:
        # the batched policy shares the scalar one's weight layout, so either checkpoint loads;
        # cg_runner then prices every (project, anchor, order) case in one rollout
        ck = torch.load(args.ckpt, map_location="cpu")
        policy = PricingGNNBatch(hidden=ck["args"]["hidden"]).to(args.device)
        policy.load_state_dict(ck["state"])
        policy.eval()
        S = ck["args"]["S"]

    for k in range(args.seeds):
        seed = 8000 + k
        inst = generate_multiproject_instance(num_projects=args.P, tasks_per_project=args.tasks,
                                              num_templates=2, tool_unary=args.tool,
                                              plan_offset=args.plan_offset,
                                              tool_purchase=args.tool_purchase, seed=seed)
        t0 = time.time()
        mip = solve_compact(inst, "dis", "fl", mode="mip", time_limit_s=args.mip_limit)
        opt = mip.get("incumbent")
        proven = opt is not None and abs(mip["bound"] - opt) < 1e-6 * max(1.0, abs(opt))
        rows = [{"method": "MIP", "cost": opt, "bound": mip["bound"], "proven": proven,
                 "time": round(time.time() - t0, 1), "P": args.P, "tasks": args.tasks, "seed": seed}]
        print(json.dumps(rows[0]), flush=True)

        # the MIP always runs above, so "MIP" in --methods is accepted and skipped here
        pricers = {"CG-exact": "exact", "CG-heur": "heur", "CG-RL": "rl"}
        for meth, pricer in [(m, pricers[m]) for m in args.methods if m != "MIP"]:
            t0 = time.time()
            r = R.run(inst, pricer, time_limit_s=args.cg_limit, exact_tl=args.pricing_limit,
                      ip_time_s=args.ip_limit, policy=policy, S=S)
            row = {"method": meth, "cost": r["cost"], "lp": r["lp"], "iters": r["iters"],
                   "cols": r["cols"], "converged": r["iters"] < 500 and r["t_cg"] < args.cg_limit,
                   "t_cg": round(r["t_cg"], 1), "time": round(time.time() - t0, 1),
                   "P": args.P, "tasks": args.tasks, "seed": seed}
            if opt:
                row["gap_to_opt_%"] = None if r["cost"] is None else round(100 * (r["cost"] - opt) / opt, 2)
                row["lp_below_opt_%"] = round(100 * (opt - r["lp"]) / opt, 2)
            rows.append(row)
            print(json.dumps(row), flush=True)

        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
