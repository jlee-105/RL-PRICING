"""
Bound strength: compact formulations vs Dantzig-Wolfe (by project), following the
comparison of Kolter, Grunow & Kolisch (2025, Tables 6 and 8).

For each instance:
  OPT              compact PDDT + FL solved to optimality (SCIP)
  cLP  agg/bigm    LP relaxation of the formulation in mp_monolithic
  cLP  dis/fl      LP relaxation of the strongest compact formulation
  cRoot dis/fl     SCIP root bound (with its cuts) of the strongest compact formulation
  DW   no-cap      CG root bound, pricing without the project's capacity (Kolter's weak case)
  DW   cap         CG root bound, pricing with the project's capacity (resource-constrained)
Gaps are (OPT - bound) / OPT; a smaller gap is a stronger bound.
"""
from __future__ import annotations

import argparse
import io
import contextlib
import time

from multiproject_instance import generate_multiproject_instance
from mp_compact import solve_compact
from mp_cg import run_cg, exact_pricer


def dw_bound(inst, project_capacity: bool):
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_cg(inst, pricer=exact_pricer(60.0, project_capacity), verbose=False, integer_time_limit_s=5)
    converged = r["iters"] < 200
    return {"bound": r["lp_obj"] if converged else r["lower_bound"], "time": time.time() - t0,
            "iters": r["iters"], "cols": r["columns"], "converged": converged}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    names = ["cLP agg/bigm", "cLP dis/fl", "cRoot dis/fl", "DW no-cap", "DW cap"]
    print(f"P={args.projects} tasks={args.tasks}  gap = (OPT - bound)/OPT", flush=True)
    sums = {n: [] for n in names}
    for seed in args.seeds:
        inst = generate_multiproject_instance(num_projects=args.projects, tasks_per_project=args.tasks,
                                              num_templates=2, seed=seed)
        opt = solve_compact(inst, "dis", "fl", mode="mip", time_limit_s=1800)
        OPT = opt["incumbent"]
        proven = abs(opt["bound"] - OPT) < 1e-6 * max(1.0, abs(OPT))
        res = {
            "cLP agg/bigm": solve_compact(inst, "agg", "bigm", mode="lp"),
            "cLP dis/fl": solve_compact(inst, "dis", "fl", mode="lp"),
            "cRoot dis/fl": solve_compact(inst, "dis", "fl", mode="root", time_limit_s=600),
            "DW no-cap": dw_bound(inst, False),
            "DW cap": dw_bound(inst, True),
        }
        print(f"\nseed {seed}: OPT={OPT:.3f} ({'proven' if proven else 'NOT proven, bound %.3f' % opt['bound']}, "
              f"{opt['time']:.1f}s)", flush=True)
        for n in names:
            b = res[n]["bound"]
            gap = 100 * (OPT - b) / OPT
            sums[n].append(gap)
            extra = f"  iters={res[n]['iters']} cols={res[n]['cols']}" if n.startswith("DW") else ""
            print(f"  {n:14s} bound={b:10.3f}  gap={gap:6.2f}%  time={res[n]['time']:7.1f}s{extra}", flush=True)

    print("\nmean gap over seeds:", flush=True)
    for n in names:
        print(f"  {n:14s} {sum(sums[n]) / len(sums[n]):6.2f}%", flush=True)


if __name__ == "__main__":
    main()
