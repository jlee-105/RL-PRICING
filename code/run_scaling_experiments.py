"""
Scaling experiments for MRCPSP-P with CG+RL.

Tests three configurations:
  1. Small multi-mode:  |J|=5,  |Q|=3, |R|=1, |M|=2, H=30
  2. Medium:            |J|=10, |Q|=3, |R|=2, |M|=3, H=50
  3. Large:             |J|=20, |Q|=3, |R|=2, |M|=3, H=80

For each configuration:
  - Train a policy (50 epochs, fast)
  - Benchmark CG+RL vs CG+CP-SAT on 10 held-out instances
"""
import time
import torch
import sys

from cg_rl_main import (
    _make_policy, train, benchmark,
    MAX_DELAY, MAX_MODES, DEVICE,
)
from instance_generator import generate_random_instance
from rcpsp_env import RCPSPEnv
from gnn_policy import SchedulingGNN
from rl_trainer import POMOTrainer

CONFIGS = [
    {
        "name": "Small (|J|=5, |Q|=3)",
        "num_tasks": 5,
        "num_resources": 1,
        "num_parts": 2,
        "horizon": 30,
        "train_epochs": 50,
        "train_instances": 10,
        "bench_instances": 10,
    },
    {
        "name": "Medium (|J|=10, |Q|=3)",
        "num_tasks": 10,
        "num_resources": 2,
        "num_parts": 3,
        "horizon": 50,
        "train_epochs": 80,
        "train_instances": 10,
        "bench_instances": 10,
    },
    {
        "name": "Large (|J|=20, |Q|=3)",
        "num_tasks": 20,
        "num_resources": 2,
        "num_parts": 3,
        "horizon": 80,
        "train_epochs": 100,
        "train_instances": 10,
        "bench_instances": 10,
    },
]


def run_config(cfg):
    print(f"\n{'='*70}")
    print(f"  CONFIG: {cfg['name']}")
    print(f"{'='*70}")

    save_path = f"policy_{cfg['num_tasks']}t_{MAX_MODES}q.pt"

    # --- Train ---
    print(f"\n--- Training ({cfg['train_epochs']} epochs) ---")
    t0 = time.time()
    policy = train(
        num_epochs=cfg["train_epochs"],
        instances_per_epoch=cfg["train_instances"],
        cg_iters_per_instance=3,
        num_tasks=cfg["num_tasks"],
        num_resources=cfg["num_resources"],
        num_parts=cfg["num_parts"],
        horizon=cfg["horizon"],
        hidden_dim=64,
        num_layers=3,
        lr=3e-4,
        num_pomo=8,
        save_path=save_path,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    # --- Benchmark ---
    print(f"\n--- Benchmark ({cfg['bench_instances']} instances) ---")
    benchmark(
        policy,
        num_instances=cfg["bench_instances"],
        num_tasks=cfg["num_tasks"],
        num_resources=cfg["num_resources"],
        num_parts=cfg["num_parts"],
        horizon=cfg["horizon"],
        verbose=False,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("  MRCPSP-P SCALING EXPERIMENTS")
    print(f"  Device: {DEVICE} ({torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'CPU'})")
    print(f"  Max modes: {MAX_MODES}, Max delay: {MAX_DELAY}")
    print("=" * 70)

    total_t0 = time.time()

    for cfg in CONFIGS:
        try:
            run_config(cfg)
        except Exception as e:
            print(f"\n!!! CONFIG {cfg['name']} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"  TOTAL TIME: {time.time() - total_t0:.1f}s")
    print(f"{'='*70}")
