"""
Sanity checks for the multi-project CG on small instances, against the monolithic MIP.

  (a) duals: at the final RMP, every column with lam > 0 has rc ~ 0 and every column rc >= 0
  (b) CG lower bound <= monolithic optimum <= CG integer solution
  (c) independent evaluator reproduces solver objectives
  (d) Wagner-Whitin procurement for the MIP schedule equals the MIP's procurement cost
"""
from __future__ import annotations

import sys

from multiproject_instance import generate_multiproject_instance, summarize
from mp_monolithic import solve_monolithic, evaluate_solution
from mp_cg import run_cg, solve_master, reduced_cost, initial_columns

TOL = 1e-4


def check_duals(inst, pools):
    m = solve_master(inst, pools)
    worst_basic, worst_neg = 0.0, 0.0
    for p, pool in enumerate(pools):
        for k, c in enumerate(pool):
            rc = reduced_cost(c, m["pi"], m["beta"], m["mu"])
            if m["lam"][p][k] > 1e-7:
                worst_basic = max(worst_basic, abs(rc))
            worst_neg = min(worst_neg, rc)
    return worst_basic, worst_neg


def main(num_instances: int = 5, num_projects: int = 3, tasks: int = 6):
    ok = True
    for seed in range(num_instances):
        inst = generate_multiproject_instance(num_projects=num_projects, tasks_per_project=tasks,
                                              num_templates=2, seed=seed)
        print(f"\n=== seed {seed} ===\n{summarize(inst).splitlines()[0]}")
        mono = solve_monolithic(inst, time_limit_s=300)
        print(f"monolithic: {mono['status']}  obj={mono.get('objective', float('nan')):.3f}  "
              f"bound={mono.get('bound', float('nan')):.3f}  time={mono['wall_time']:.1f}s")
        if "starts" not in mono:
            print("  instance infeasible -- generator should never produce this")
            ok = False
            continue
        cg = run_cg(inst, verbose=False)
        print(f"CG: LP={cg['lp_obj']:.3f} (slack {cg['lp_slack']:.3g})  LB={cg['lower_bound']:.3f}  "
              f"IP={cg['ip_obj']:.3f} (slack {cg['ip_slack']:.3g})  iters={cg['iters']}  "
              f"cols={cg['columns']}  time={cg['time_total']:.1f}s")

        # (c) evaluator vs solvers
        c_mono = mono["check"]["total"] if mono.get("check") else None
        c_cg = cg["check"]["total"] if cg.get("check") else None
        print(f"  (c) evaluator: mono {c_mono}  vs obj {mono.get('objective')};  CG {c_cg} vs IP {cg['ip_obj']:.3f}")
        ok &= c_mono is not None and abs(c_mono - mono["objective"]) < TOL
        ok &= c_cg is not None and abs(c_cg - cg["ip_obj"]) < TOL

        # (b) bound ordering (monolithic proven optimal only)
        if mono["status"] == "OPTIMAL":
            opt = mono["objective"]
            b = cg["lower_bound"] <= opt + TOL and opt <= cg["ip_obj"] + TOL
            print(f"  (b) LB {cg['lower_bound']:.3f} <= OPT {opt:.3f} <= CG-IP {cg['ip_obj']:.3f}: {b}   "
                  f"CG gap to OPT = {100 * (cg['ip_obj'] - opt) / opt:.2f}%   "
                  f"DW bound gap = {100 * (opt - cg['lp_obj']) / opt:.2f}%")
            ok &= b

        # (d) Wagner-Whitin on the MIP's schedule
        ww = evaluate_solution(inst, mono["starts"], mono["modes"], orders=None)
        if mono["status"] == "OPTIMAL":
            d = abs(ww["total"] - mono["objective"]) < TOL
            print(f"  (d) WW procurement {ww['procurement']:.3f} vs MIP {mono['check']['procurement']:.3f}: {d}")
            ok &= d

    # (a) dual sign convention on one instance, with a few CG iterations worth of columns
    inst = generate_multiproject_instance(num_projects=num_projects, tasks_per_project=tasks, num_templates=2, seed=0)
    from mp_cg import price_exact
    pools = initial_columns(inst)
    for _ in range(5):
        m = solve_master(inst, pools)
        for p in range(len(pools)):
            pools[p].append(price_exact(inst, p, m["pi"], m["beta"], m["mu"])[0])
    wb, wn = check_duals(inst, pools)
    print(f"\n(a) max |rc| over basic columns = {wb:.2e};  min rc over pool = {wn:.2e}")
    ok &= wb < 1e-5 and wn > -1e-5
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECK FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
