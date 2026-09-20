"""Quick end-to-end smoke test for the CG+RL pipeline."""
from ColumnGenerationHeuristic import (
    build_toy_instance, schedule_cost, solve_master_lp, schedule_with_cpsat,
)
from rcpsp_env import RCPSPEnv, DEVICE
from gnn_policy import SchedulingGNN
from rl_trainer import POMOTrainer
from instance_generator import generate_random_instance
import torch

MAX_MODES = 3

def test_env_and_policy():
    inst = build_toy_instance()
    print("=== 1. Environment + Policy smoke test ===")
    print("Tasks:", list(inst.activities.keys()))

    env = RCPSPEnv(inst, max_modes=MAX_MODES)
    obs = env.reset()
    nf, adj, gf, mask = obs
    print(f"Node features: {nf.shape}, Adj: {adj.shape}, Global: {gf.shape}")
    eligible = [env.task_ids[i] for i in range(env.num_tasks) if env.eligible[i]]
    print(f"Initial eligible: {eligible}")

    policy = SchedulingGNN(
        node_dim=env.node_feature_dim,
        global_dim=env.global_feature_dim,
        num_modes=env.num_modes,
        num_delays=env.num_delays,
        hidden_dim=64,
        num_layers=3,
    ).to(DEVICE)
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}  modes={env.num_modes}  delays={env.delay_options}  device={DEVICE}")

    # Greedy rollout
    env.reset()
    done = False
    obs = env.get_observation()
    steps = 0
    while not done:
        nf, adj, gf, mask = obs
        with torch.no_grad():
            logits, _ = policy(nf, adj, gf, mask)
        action = logits.argmax().item()
        obs, reward, done, info = env.step(action)
        steps += 1

    print(f"Greedy rollout: {steps} steps")
    print(f"  Schedule: {info['starts']}")
    print(f"  Modes: {info['modes']}")
    print(f"  Cost: {info['cost']:.3f}, Makespan: {info['makespan']}")
    print(f"  Reduced cost: {info['reduced_cost']:.6f}")
    print("PASS\n")
    return policy


def test_trainer(policy):
    print("=== 2. POMO Trainer smoke test ===")
    inst = build_toy_instance()

    # Get some duals from a real master solve
    s0, m0, _, _ = schedule_with_cpsat(inst, time_limit_s=2)
    c0, _, d0 = schedule_cost(inst, s0, m0)
    _, alpha_duals, mu, obj, _ = solve_master_lp(inst, [d0], [c0])
    print(f"Master LP duals: mu={mu:.6f}, alpha entries={len(alpha_duals)}")

    trainer = POMOTrainer(policy, lr=3e-4, num_pomo=4, max_modes=MAX_MODES)
    result = trainer.train_on_instance(inst, alpha_duals, mu)
    print(f"  Train loss: {result['loss']:.4f}")
    print(f"  Mean reward: {result['mean_reward']:.4f}")
    print(f"  Best reduced cost: {result['best_rc']:.6f}")

    # Generate columns
    columns = trainer.generate_columns(inst, alpha_duals, mu, num_samples=8)
    print(f"  Generated {len(columns)} columns")
    print(f"  Best column rc: {columns[0]['reduced_cost']:.6f}")
    print(f"  Worst column rc: {columns[-1]['reduced_cost']:.6f}")
    print("PASS\n")
    return trainer


def test_cg_rl_loop(policy, trainer):
    print("=== 3. CG+RL solve loop (toy instance) ===")
    from cg_rl_main import cg_rl_solve

    inst = build_toy_instance()
    result = cg_rl_solve(
        inst, policy, trainer,
        max_iters=5, cpsat_fallback=True, verbose=True,
    )
    print(f"\nResult: obj={result['final_obj']:.3f}, cost={result['final_cost']:.3f}")
    print(f"  RL columns: {result['rl_cols']}, CP-SAT columns: {result['cpsat_cols']}")
    print("PASS\n")


def test_instance_generator():
    print("=== 4. Random instance generator ===")
    inst = generate_random_instance(num_tasks=10, num_resources=2, num_parts=3, horizon=50, max_modes=3, seed=42)
    print(f"Tasks: {len(inst.activities)}, Parts: {len(inst.parts)}, Resources: {len(inst.renewable_capacity)}")
    print(f"Precedence edges: {len(inst.precedence)}")
    for tid, act in list(inst.activities.items())[:3]:
        print(f"  {tid}: {act.num_modes} modes, durations={[m.duration for m in act.modes]}")

    env = RCPSPEnv(inst, max_modes=MAX_MODES)
    print(f"Feature dim: {env.node_feature_dim}, Global dim: {env.global_feature_dim}")

    policy = SchedulingGNN(
        node_dim=env.node_feature_dim,
        global_dim=env.global_feature_dim,
        num_modes=env.num_modes,
        num_delays=env.num_delays,
    ).to(DEVICE)

    # Quick rollout
    env.reset()
    done = False
    obs = env.get_observation()
    while not done:
        nf, adj, gf, mask = obs
        with torch.no_grad():
            logits, _ = policy(nf, adj, gf, mask)
        obs, _, done, info = env.step(logits.argmax().item())

    print(f"Random instance schedule: cost={info['cost']:.3f}, makespan={info['makespan']}")
    print(f"  Modes: {info['modes']}")
    print("PASS\n")


def test_short_training():
    print("=== 5. Short training run (3 epochs) ===")
    from cg_rl_main import train
    policy = train(
        num_epochs=3,
        instances_per_epoch=3,
        cg_iters_per_instance=2,
        num_tasks=5,
        num_resources=1,
        num_parts=2,
        horizon=20,
        hidden_dim=32,
        num_layers=2,
        num_pomo=4,
        save_path="test_policy.pt",
        verbose=True,
    )
    print("PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("CG+RL Pipeline End-to-End Test")
    print("=" * 60 + "\n")

    policy = test_env_and_policy()
    trainer = test_trainer(policy)
    test_cg_rl_loop(policy, trainer)
    test_instance_generator()
    test_short_training()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
