"""
Verify CG+RL matches optimal solution from CG+CP-SAT.

1. CG + CP-SAT  (optimal baseline)
2. Train RL policy (100 epochs, matched structure)
3. CG + trained RL (with and without CP-SAT fallback)
"""
import torch
from ColumnGenerationHeuristic import (
    build_toy_instance, run_cg_cpsat, schedule_with_cpsat, schedule_cost,
    solve_master_lp,
)
from rcpsp_env import RCPSPEnv
from gnn_policy import SchedulingGNN
from rl_trainer import POMOTrainer
from cg_rl_main import cg_rl_solve, _make_policy, train


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    inst = build_toy_instance()

    # ---------------------------------------------------------------
    # 1. CG + CP-SAT (optimal reference)
    # ---------------------------------------------------------------
    header("1. CG + CP-SAT (optimal reference)")
    ref = run_cg_cpsat(inst, max_iters=15)
    opt_obj = ref["final_obj"]
    opt_cost = ref["final_cost"]
    opt_starts = ref["final_starts"]
    print(f"\n>>> OPTIMAL: obj={opt_obj:.3f}  cost={opt_cost:.3f}")
    print(f">>> Schedule: {opt_starts}")

    # ---------------------------------------------------------------
    # 2. Train policy (100 epochs, matched structure)
    # ---------------------------------------------------------------
    header("2. Training RL policy (300 epochs, toy-matched structure)")
    policy = train(
        num_epochs=300,
        instances_per_epoch=10,
        cg_iters_per_instance=4,
        num_tasks=5,
        num_resources=1,
        num_parts=2,
        horizon=50,
        hidden_dim=64,
        num_layers=3,
        lr=1e-4,
        num_pomo=8,
        save_path="policy_verify.pt",
        verbose=True,
    )

    # ---------------------------------------------------------------
    # 3. Debug: check what RL generates under toy duals
    # ---------------------------------------------------------------
    header("3. RL column generation under toy duals")
    trainer = POMOTrainer(policy, num_pomo=8)

    # Reproduce the exact duals from CG iter 1
    s0, m0, _, _ = schedule_with_cpsat(inst, time_limit_s=2)
    c0, _, d0 = schedule_cost(inst, s0, m0)
    _, alpha_duals, mu, obj, _ = solve_master_lp(inst, [d0], [c0])
    print(f"Duals: mu={mu:.3f}")
    for k, v in sorted(alpha_duals.items()):
        print(f"  alpha{k} = {v:.3f}")

    columns = trainer.generate_columns(inst, alpha_duals, mu, num_samples=32, greedy_ratio=0.25)
    print(f"\nGenerated {len(columns)} columns:")
    for i, col in enumerate(columns[:5]):
        print(f"  #{i}: rc={col['reduced_cost']:.3f}  cost={col['cost']:.3f}  "
              f"starts={col['starts']}")
    best = columns[0]
    print(f"\nBest column: rc={best['reduced_cost']:.3f}  cost={best['cost']:.3f}")
    print(f"  starts={best['starts']}")
    improving = sum(1 for c in columns if c["reduced_cost"] < -1e-6)
    print(f"Improving columns (rc<0): {improving}/{len(columns)}")

    # ---------------------------------------------------------------
    # 4. CG + trained RL (with fallback)
    # ---------------------------------------------------------------
    header("4. CG + trained RL (with CP-SAT fallback)")
    res4 = cg_rl_solve(inst, policy, trainer, max_iters=15, cpsat_fallback=True, verbose=True)
    gap4 = (res4["final_obj"] - opt_obj) / max(abs(opt_obj), 1e-9) * 100
    print(f"\n>>> obj={res4['final_obj']:.3f}  gap={gap4:.2f}%")
    print(f">>> RL cols={res4['rl_cols']}  CP-SAT cols={res4['cpsat_cols']}")

    # ---------------------------------------------------------------
    # 5. CG + trained RL (NO fallback)
    # ---------------------------------------------------------------
    header("5. CG + trained RL (NO fallback -- pure RL)")
    res5 = cg_rl_solve(inst, policy, trainer, max_iters=15, cpsat_fallback=False, verbose=True)
    gap5 = (res5["final_obj"] - opt_obj) / max(abs(opt_obj), 1e-9) * 100
    print(f"\n>>> obj={res5['final_obj']:.3f}  gap={gap5:.2f}%")
    print(f">>> RL cols={res5['rl_cols']}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    header("OPTIMALITY VERIFICATION SUMMARY")
    print(f"{'Method':<35} {'Obj':>10} {'Cost':>10} {'Gap':>8} {'RL cols':>8} {'CP cols':>8}")
    print("-" * 80)
    print(f"{'CG+CP-SAT (optimal)':<35} {opt_obj:>10.3f} {opt_cost:>10.3f} {'0.00%':>8} {'-':>8} {ref['num_columns']:>8}")
    print(f"{'CG+RL trained + fallback':<35} {res4['final_obj']:>10.3f} {res4['final_cost']:>10.3f} {gap4:>7.2f}% {res4['rl_cols']:>8} {res4['cpsat_cols']:>8}")
    print(f"{'CG+RL trained (pure RL)':<35} {res5['final_obj']:>10.3f} {res5['final_cost']:>10.3f} {gap5:>7.2f}% {res5['rl_cols']:>8} {'-':>8}")

    match_fb = abs(res4["final_obj"] - opt_obj) < 1e-3
    match_pure = abs(res5["final_obj"] - opt_obj) < 1e-3
    print(f"\nWith fallback matches optimal?  {'YES' if match_fb else 'NO'}")
    print(f"Pure RL matches optimal?        {'YES' if match_pure else 'NO'}")

    import os
    try:
        os.remove("policy_verify.pt")
    except OSError:
        pass


if __name__ == "__main__":
    main()
