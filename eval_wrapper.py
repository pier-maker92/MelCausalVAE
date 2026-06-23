import sys
import subprocess

# Filter out hydra/training arguments starting with settings=, training., hydra.
filtered_args = []
for arg in sys.argv[1:]:
    if any(arg.startswith(prefix) for prefix in ["settings=", "training.", "hydra."]):
        continue
    filtered_args.append(arg)

# Run eval.py with python, forwarding the filtered arguments
cmd = [sys.executable, "eval.py"] + filtered_args
print(f"Wrapper executing: {' '.join(cmd)}")
sys.exit(subprocess.call(cmd))
