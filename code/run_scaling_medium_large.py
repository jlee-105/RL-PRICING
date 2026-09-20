"""Run Medium and Large scaling experiments only."""
import time
import torch
from cg_rl_main import train, benchmark, MAX_DELAY, MAX_MODES, DEVICE

print("=" * 70)
print(f"  Device: {DEVICE} ({torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'CPU'})")
print("=" * 70)

for cfg in [
    {"name": "Medium (|J|=10)", "nt": 10, "nr": 2, "np": 3, "h": 50, "ep": 80},
    {"name": "Large (|J|=20)",  "nt": 20, "nr": 2, "np": 3, "h": 80, "ep": 100},
]:
    print(f"\n{'='*70}\n  {cfg['name']}\n{'='*70}")
    try:
        t0 = time.time()
        policy = train(
            num_epochs=cfg["ep"], instances_per_epoch=10, cg_iters_per_instance=3,
            num_tasks=cfg["nt"], num_resources=cfg["nr"], num_parts=cfg["np"],
            horizon=cfg["h"], hidden_dim=64, num_layers=3, lr=1e-4, num_pomo=8,
            save_path=f"policy_{cfg['nt']}t_3q.pt", verbose=True,
        )
        print(f"Train time: {time.time()-t0:.1f}s")

        benchmark(
            policy, num_instances=10,
            num_tasks=cfg["nt"], num_resources=cfg["nr"], num_parts=cfg["np"],
            horizon=cfg["h"], verbose=False,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback; traceback.print_exc()
