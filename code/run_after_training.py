"""Wait for the training pipeline (run_pricer_pipeline.py) to exit, then run the large-scale experiment."""
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).parent
PIPELINE_PID = int(sys.argv[1])

while psutil.pid_exists(PIPELINE_PID):
    time.sleep(120)

with open(HERE / "logs" / "large_scale.txt", "w") as out, open(HERE / "logs" / "large_scale.err", "w") as err:
    subprocess.run([sys.executable, "-u", "exp_large_scale.py", "--P", "5", "10", "20", "40", "--seeds", "3"],
                   cwd=HERE, stdout=out, stderr=err)
