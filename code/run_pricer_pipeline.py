"""Wait for pricing_data.py (train and val) to finish, then train the RL pricer. Detached helper."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
LOGS = HERE / "logs"


def finished(name):
    f = LOGS / f"data_{name}.txt"
    return f.exists() and "done" in f.read_text(errors="ignore")


while not (finished("train") and finished("val")):
    time.sleep(60)

with open(LOGS / "train_rl.txt", "w") as out, open(LOGS / "train_rl.err", "w") as err:
    subprocess.run([sys.executable, "-u", "rl_pricing_trainer.py", "--updates", "1500", "--batch", "16",
                    "--S", "1", "--eval_every", "100", "--tag", "rl"],
                   cwd=HERE, stdout=out, stderr=err)
