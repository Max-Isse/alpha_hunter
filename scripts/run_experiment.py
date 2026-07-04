import subprocess
import sys


def run_experiment(config_path: str):
    """Launch a training run with a specific config."""
    subprocess.run([sys.executable, "-m", "src.main", "--config", config_path])


if __name__ == "__main__":
    run_experiment("configs/base.yaml")