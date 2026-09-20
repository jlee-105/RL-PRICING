"""
CG + RL main driver.

Three modes (all runnable from the CLI):
  1. train   – online CG training: interleave master LP solves with REINFORCE
  2. solve   – CG+RL loop on a single instance (with optional CP-SAT fallback)
  3. bench   – benchmark CG+RL vs CG+CP-SAT on generated instances
"""
from __future__ import annotations

import argparse
import time
import json
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ColumnGenerationHeuristic import (
    Instance,
    build_toy_instance,
    schedule_with_cpsat,
    solve_master_lp,
    schedule_cost,
    IMPROVEMENT_TOL,
    SCHED_TIME_LIMIT_S,
    MAX_PRICING_ITERS,
    run_cg_cpsat,
)
from rcpsp_env import RCPSPEnv, DEVICE
from gnn_policy import SchedulingGNN
from rl_trainer import POMOTrainer
from instance_generator import generate_random_instance, generate_training_set


# =====================================================================
# Helpers
# =====================================================================

def _compute_rc(cost, D, alpha_duals, mu, inst):
    alpha_dot = 0.0
    for m in inst.parts:
        for t, d in enumerate(D[m]):
            if d:
                alpha_dot += alpha_duals.get((m, t), 0.0) * d
    return cost - alpha_dot - mu


MAX_DELAY = 10
MAX_MODES = 3

def _make_policy(inst: Instance, hidden: int = 64, layers: int = 3) -> SchedulingGNN:
    env = RCPSPEnv(inst, max_delay=MAX_DELAY, max_modes=MAX_MODES)
    return SchedulingGNN(
        node_dim=env.node_feature_dim,
        global_dim=env.global_feature_dim,
        num_modes=env.num_modes,
        num_delays=env.num_delays,
        hidden_dim=hidden,
        num_layers=layers,
    ).to(DEVICE)


# =====================================================================
# CG + RL solve loop
# =====================================================================

def cg_rl_solve(
    inst: Instance,
    policy: SchedulingGNN,
    trainer: POMOTrainer,
    max_iters: int = 20,
    cpsat_fallback: bool = True,
    train_during_cg: bool = False,
    verbose: bool = True,
) -> dict:
    """Run column generation with RL pricing on a single instance."""
    policy.eval()

    # -- seed pool with a greedy RL rollout (no duals) --
    env = RCPSPEnv(inst, max_delay=MAX_DELAY, max_modes=MAX_MODES)
    env.reset()
    done = False
    obs = env.get_observation()
    while not done:
        nf, adj, gf, mask = obs
        with torch.no_grad():
            logits, _ = policy(nf, adj, gf, mask)
        obs, _, done, info = env.step(logits.argmax().item())

    starts0 = info["starts"]
    modes0 = info.get("modes", {j: 0 for j in starts0})
    cost0, cmax0, D0 = schedule_cost(inst, starts0, modes0)
    if verbose:
        print(f"RL seed: {starts0}  Cmax={cmax0}  Cost={cost0:.3f}")

    pool_starts = [starts0]
    pool_cost = [cost0]
    pool_D = [D0]

    stats = {"rl_cols": 0, "cpsat_cols": 0, "iters": 0, "rl_times": [], "cpsat_times": []}

    for it in range(1, max_iters + 1):
        stats["iters"] = it
        lam, alpha_duals, mu, obj, slack = solve_master_lp(inst, pool_D, pool_cost)

        if verbose:
            print(f"\n--- CG+RL iter {it}  obj={obj:.3f}  slack={slack:.6f}  mu={mu:.6f}")

        # --- optional: train policy on this (inst, duals) pair ---
        if train_during_cg and trainer is not None:
            tr = trainer.train_on_instance(inst, alpha_duals, mu)
            if verbose:
                print(f"  train loss={tr['loss']:.4f}  mean_reward={tr['mean_reward']:.4f}")

        # --- RL pricing ---
        t0 = time.time()
        columns = trainer.generate_columns(inst, alpha_duals, mu, num_samples=16)
        rl_time = time.time() - t0
        stats["rl_times"].append(rl_time)

        best_col = columns[0]  # lowest reduced cost
        rc_rl = best_col["reduced_cost"]
        if verbose:
            print(f"  RL pricing: rc={rc_rl:.6f}  cost={best_col['cost']:.3f}  time={rl_time:.3f}s")

        if rc_rl < -IMPROVEMENT_TOL:
            pool_starts.append(best_col["starts"])
            pool_cost.append(best_col["cost"])
            pool_D.append(best_col["D"])
            stats["rl_cols"] += 1
            if verbose:
                print("  -> added RL column")
            continue

        # --- CP-SAT fallback ---
        if cpsat_fallback:
            t0 = time.time()
            starts_cp, modes_cp, _, _ = schedule_with_cpsat(inst, penalty_alpha=alpha_duals, time_limit_s=SCHED_TIME_LIMIT_S)
            cp_time = time.time() - t0
            stats["cpsat_times"].append(cp_time)
            cost_cp, _, Dcp = schedule_cost(inst, starts_cp, modes_cp)
            rc_cp = _compute_rc(cost_cp, Dcp, alpha_duals, mu, inst)
            if verbose:
                print(f"  CP-SAT fallback: rc={rc_cp:.6f}  cost={cost_cp:.3f}  time={cp_time:.3f}s")
            if rc_cp < -IMPROVEMENT_TOL:
                pool_starts.append(starts_cp)
                pool_cost.append(cost_cp)
                pool_D.append(Dcp)
                stats["cpsat_cols"] += 1
                if verbose:
                    print("  -> added CP-SAT column")
                continue

        if verbose:
            print("  No improving column. Stopping.")
        break

    # --- final master ---
    lam, _, _, obj, slack = solve_master_lp(inst, pool_D, pool_cost)
    best_k = max(range(len(lam)), key=lambda k: lam[k])

    result = {
        "final_obj": obj,
        "final_cost": pool_cost[best_k],
        "final_starts": pool_starts[best_k],
        "num_columns": len(pool_starts),
        **stats,
    }

    if verbose:
        print(f"\n=== CG+RL FINAL ===  obj={obj:.3f}  cost={pool_cost[best_k]:.3f}")
        print(f"  schedule: {pool_starts[best_k]}")
        print(f"  RL cols={stats['rl_cols']}  CP-SAT cols={stats['cpsat_cols']}")

    return result


# =====================================================================
# Training
# =====================================================================

def train(
    num_epochs: int = 100,
    instances_per_epoch: int = 10,
    cg_iters_per_instance: int = 3,
    num_tasks: int = 8,
    num_resources: int = 1,
    num_parts: int = 2,
    horizon: int = 30,
    hidden_dim: int = 64,
    num_layers: int = 3,
    lr: float = 3e-4,
    num_pomo: int = 8,
    save_path: str = "policy_cg_rl.pt",
    verbose: bool = True,
):
    """Online CG training: interleave LP master solves with REINFORCE updates."""

    # Build policy sized for the target instance structure
    ref_inst = generate_random_instance(num_tasks, num_resources, num_parts, horizon, seed=0)
    policy = _make_policy(ref_inst, hidden_dim, num_layers)
    print(f"Training on {DEVICE} ({torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'CPU'})")
    trainer = POMOTrainer(policy, lr=lr, num_pomo=num_pomo, entropy_coef=0.05, max_delay=MAX_DELAY, max_modes=MAX_MODES)

    best_avg_rc = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_best_rc = 0.0
        n_updates = 0

        instances = generate_training_set(
            instances_per_epoch, num_tasks, num_resources, num_parts, horizon,
            max_modes=MAX_MODES, base_seed=epoch * 1000,
        )

        for inst in instances:
            # Seed CG pool with a CP-SAT schedule (reliable seed)
            try:
                s0, m0, _, _ = schedule_with_cpsat(inst, time_limit_s=2)
            except RuntimeError:
                continue
            c0, _, d0 = schedule_cost(inst, s0, m0)
            pool_D = [d0]
            pool_cost = [c0]

            for cg_it in range(cg_iters_per_instance):
                try:
                    lam, alpha_duals, mu, obj, slack = solve_master_lp(inst, pool_D, pool_cost)
                except RuntimeError:
                    break

                tr = trainer.train_on_instance(inst, alpha_duals, mu)
                epoch_loss += tr["loss"]
                epoch_reward += tr["mean_reward"]
                epoch_best_rc += tr["best_rc"]
                n_updates += 1

                # Add best RL column to pool for next CG iter
                cols = trainer.generate_columns(inst, alpha_duals, mu, num_samples=8)
                if cols and cols[0]["reduced_cost"] < -1e-6:
                    pool_D.append(cols[0]["D"])
                    pool_cost.append(cols[0]["cost"])

        if n_updates == 0:
            continue

        avg_loss = epoch_loss / n_updates
        avg_rew = epoch_reward / n_updates
        avg_rc = epoch_best_rc / n_updates

        if verbose and epoch % 5 == 0:
            print(
                f"Epoch {epoch:4d}  loss={avg_loss:.4f}  "
                f"mean_reward={avg_rew:.4f}  avg_best_rc={avg_rc:.4f}"
            )

        if avg_rc < best_avg_rc:
            best_avg_rc = avg_rc
            torch.save(policy.state_dict(), save_path)
            if verbose and epoch % 5 == 0:
                print(f"  ^ saved best model (avg rc={avg_rc:.4f})")

    print(f"\nTraining complete.  Best avg reduced cost = {best_avg_rc:.4f}")
    print(f"Model saved to {save_path}")
    return policy


# =====================================================================
# Benchmark: CG+RL vs CG+CP-SAT
# =====================================================================

def benchmark(
    policy: SchedulingGNN,
    num_instances: int = 20,
    num_tasks: int = 8,
    num_resources: int = 1,
    num_parts: int = 2,
    horizon: int = 30,
    verbose: bool = True,
):
    trainer = POMOTrainer(policy, num_pomo=8, max_delay=MAX_DELAY, max_modes=MAX_MODES)
    instances = generate_training_set(
        num_instances, num_tasks, num_resources, num_parts, horizon,
        max_modes=MAX_MODES, base_seed=99999,
    )

    rl_results = []
    cp_results = []

    for i, inst in enumerate(instances):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Instance {i+1}/{num_instances}")
            print(f"{'='*60}")

        # CG + RL
        if verbose:
            print("\n[CG + RL]")
        t0 = time.time()
        try:
            res_rl = cg_rl_solve(inst, policy, trainer, max_iters=15, cpsat_fallback=True, verbose=verbose)
            res_rl["total_time"] = time.time() - t0
            rl_results.append(res_rl)
        except Exception as e:
            print(f"  CG+RL failed: {e}")
            rl_results.append(None)

        # CG + CP-SAT
        if verbose:
            print("\n[CG + CP-SAT]")
        t0 = time.time()
        try:
            res_cp = run_cg_cpsat(inst, max_iters=15)
            res_cp["total_time"] = time.time() - t0
            cp_results.append(res_cp)
        except Exception as e:
            print(f"  CG+CP-SAT failed: {e}")
            cp_results.append(None)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Instance':>10} {'CG+RL obj':>12} {'CG+CP obj':>12} {'RL time':>10} {'CP time':>10} {'RL cols':>8} {'CP cols':>8}")
    print("-" * 74)

    for i in range(num_instances):
        rl = rl_results[i]
        cp = cp_results[i]
        rl_obj = f"{rl['final_obj']:.3f}" if rl else "FAIL"
        cp_obj = f"{cp['final_obj']:.3f}" if cp else "FAIL"
        rl_t = f"{rl['total_time']:.2f}s" if rl else "-"
        cp_t = f"{cp['total_time']:.2f}s" if cp else "-"
        rl_c = str(rl["num_columns"]) if rl else "-"
        cp_c = str(cp["num_columns"]) if cp else "-"
        print(f"{i+1:>10} {rl_obj:>12} {cp_obj:>12} {rl_t:>10} {cp_t:>10} {rl_c:>8} {cp_c:>8}")

    valid = [(rl, cp) for rl, cp in zip(rl_results, cp_results) if rl and cp]
    if valid:
        avg_gap = sum(
            (rl["final_obj"] - cp["final_obj"]) / max(abs(cp["final_obj"]), 1e-9)
            for rl, cp in valid
        ) / len(valid)
        avg_rl_time = sum(rl["total_time"] for rl, _ in valid) / len(valid)
        avg_cp_time = sum(cp["total_time"] for _, cp in valid) / len(valid)
        print(f"\nAvg gap (RL-CP)/CP: {avg_gap:.4f}  ({avg_gap*100:.2f}%)")
        print(f"Avg time  RL: {avg_rl_time:.2f}s   CP-SAT: {avg_cp_time:.2f}s")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="CG + RL for RCPSP + Procurement")
    sub = parser.add_subparsers(dest="mode")

    # -- train --
    p_train = sub.add_parser("train", help="Online CG training")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--instances-per-epoch", type=int, default=10)
    p_train.add_argument("--cg-iters", type=int, default=3)
    p_train.add_argument("--num-tasks", type=int, default=8)
    p_train.add_argument("--num-resources", type=int, default=1)
    p_train.add_argument("--num-parts", type=int, default=2)
    p_train.add_argument("--horizon", type=int, default=30)
    p_train.add_argument("--hidden", type=int, default=64)
    p_train.add_argument("--layers", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--pomo", type=int, default=8)
    p_train.add_argument("--save", type=str, default="policy_cg_rl.pt")

    # -- solve --
    p_solve = sub.add_parser("solve", help="CG+RL on toy instance")
    p_solve.add_argument("--model", type=str, default="policy_cg_rl.pt")
    p_solve.add_argument("--max-iters", type=int, default=20)
    p_solve.add_argument("--no-fallback", action="store_true")
    p_solve.add_argument("--train-online", action="store_true")

    # -- bench --
    p_bench = sub.add_parser("bench", help="Benchmark CG+RL vs CG+CP-SAT")
    p_bench.add_argument("--model", type=str, default="policy_cg_rl.pt")
    p_bench.add_argument("--num-instances", type=int, default=20)
    p_bench.add_argument("--num-tasks", type=int, default=8)
    p_bench.add_argument("--num-resources", type=int, default=1)
    p_bench.add_argument("--num-parts", type=int, default=2)
    p_bench.add_argument("--horizon", type=int, default=30)

    args = parser.parse_args()

    if args.mode == "train":
        train(
            num_epochs=args.epochs,
            instances_per_epoch=args.instances_per_epoch,
            cg_iters_per_instance=args.cg_iters,
            num_tasks=args.num_tasks,
            num_resources=args.num_resources,
            num_parts=args.num_parts,
            horizon=args.horizon,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            lr=args.lr,
            num_pomo=args.pomo,
            save_path=args.save,
        )

    elif args.mode == "solve":
        inst = build_toy_instance()
        policy = _make_policy(inst)
        if Path(args.model).exists():
            policy.load_state_dict(torch.load(args.model, weights_only=True, map_location=DEVICE))
            print(f"Loaded model from {args.model}")
        else:
            print("No trained model found — using random policy")
        trainer = POMOTrainer(policy, num_pomo=8, max_delay=MAX_DELAY, max_modes=MAX_MODES)
        cg_rl_solve(
            inst, policy, trainer,
            max_iters=args.max_iters,
            cpsat_fallback=not args.no_fallback,
            train_during_cg=args.train_online,
        )

    elif args.mode == "bench":
        ref = generate_random_instance(args.num_tasks, args.num_resources, args.num_parts, args.horizon, seed=0)
        policy = _make_policy(ref)
        if Path(args.model).exists():
            policy.load_state_dict(torch.load(args.model, weights_only=True, map_location=DEVICE))
            print(f"Loaded model from {args.model}")
        else:
            print("No trained model found — using random policy")
        benchmark(
            policy,
            num_instances=args.num_instances,
            num_tasks=args.num_tasks,
            num_resources=args.num_resources,
            num_parts=args.num_parts,
            horizon=args.horizon,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
