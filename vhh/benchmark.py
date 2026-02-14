import subprocess
import sys
import os
from vhh.af3 import af3
import json

NOTEBOOK = "vhh/notebooks/optimizer_exploration.py"
RESULTS_DIR = "results/abmpnn_test6"

af3_computed_runs = set()
os.makedirs(RESULTS_DIR, exist_ok=True)

while True:
    # Run notebook once
    env = os.environ.copy()
    env.update({"RESULTS_DIR": RESULTS_DIR})
    subprocess.run([sys.executable, NOTEBOOK], env=env, check=False)

    # Get AF3 predictions for sequences
    runs = os.listdir(RESULTS_DIR)
    for filename in runs:
        # Don't calculate if already computed
        if filename in af3_computed_runs:
            continue

        # Get run data
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, 'r') as f:
            data = json.load(f)

        # Make AF3 prediction (fold output will be cached)
        output = af3.fold(data['binder_sequence'], data['target_sequence'])
        af3_computed_runs.add(filename)

        print(f"\n==== AF3 prediction for {filename} ====")
        print(f"AF3 iptm:       {output.iptm}")
        print(f"AF3 ipsae:      {output.ipsae}")
        print(f"AF3 ipsae_min:  {output.ipsae_min}")
        print("=======================================\n")
