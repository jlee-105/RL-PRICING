"""Baseline vs BO curriculum, same budget, on the batched GPU trainer."""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
common = ["--updates", "1500", "--batch", "32", "--S", "1", "--eval_every", "250",
          "--eval_problems", "256", "--log_every", "100"]
runs = [("rl_base", ["--curriculum", "none"]),
        ("rl_bo", ["--curriculum", "bo", "--bo_every", "250", "--bo_mix", "0.5",
                   "--bo_init", "5", "--bo_iter", "5", "--bo_problems", "8", "--bo_keep", "4"])]
for tag, extra in runs:
    print(f"=== {tag} ===", flush=True)
    with open(HERE / "logs" / f"train_{tag}.txt", "w") as out:
        subprocess.run([sys.executable, "-u", "rl_pricing_trainer_batch.py", "--tag", tag] + common + extra,
                       cwd=HERE, stdout=out, stderr=subprocess.STDOUT)
    print(f"=== {tag} done ===", flush=True)
