"""Branch-and-price must reproduce the monolithic MIP optimum on small instances."""
from __future__ import annotations

import sys

from multiproject_instance import generate_multiproject_instance
from mp_monolithic import solve_monolithic
from mp_bp import BranchAndPrice


def main(seeds=range(5), num_projects: int = 3, tasks: int = 6):
    ok = True
    rows = []
    for seed in seeds:
        inst = generate_multiproject_instance(num_projects=num_projects, tasks_per_project=tasks,
                                              num_templates=2, seed=seed)
        mono = solve_monolithic(inst, time_limit_s=600)
        bp = BranchAndPrice(inst, verbose=False).solve(time_limit_s=1800)
        match = mono["status"] == "OPTIMAL" and bp["solved"] and abs(bp["objective"] - mono["objective"]) < 1e-4
        ok &= match
        rows.append((seed, mono["objective"], mono["wall_time"], bp))
        print(f"seed {seed}: MIP={mono['objective']:9.3f} ({mono['status']}, {mono['wall_time']:6.1f}s)   "
              f"B&P={bp['objective']:9.3f}  LB={bp['lower_bound']:9.3f}  root LB={bp['root_lb']:9.3f}  "
              f"solved={bp['solved']}  nodes={bp['nodes']}  unresolved={bp['unresolved']}  "
              f"cols={bp['columns']}  time={bp['time']:6.1f}s  match={match}", flush=True)
    print("\nB&P MATCHES MIP ON ALL" if ok else "\nMISMATCH")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
