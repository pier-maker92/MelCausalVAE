import subprocess
import sys
import os


def main():
    env = os.environ.copy()
    env["HF_HOME"] = "/Volumes/Crucial X6/HF_HOME"
    env["SLURM_TMPDIR"] = os.path.expanduser("~/Research/")

    command = ["python", "train.py", "settings=debug"] + sys.argv[1:]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, env=env)


if __name__ == "__main__":
    main()
