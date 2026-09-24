"""Run every experiment in configs/experiments.yaml and log each one to MLflow.

Usage:
    python -m src.run_experiments --config configs/config.yaml --experiments configs/experiments.yaml
"""

import argparse
import sys

from src.train import train_and_log
from src.utils import load_config, merge_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run all configured experiments.")
    parser.add_argument("--config", default="configs/config.yaml", help="Base config file")
    parser.add_argument(
        "--experiments", default="configs/experiments.yaml", help="Experiments file"
    )
    args = parser.parse_args(argv)

    base_config = load_config(args.config)
    experiments = load_config(args.experiments)["experiments"]

    results = []
    for experiment in experiments:
        run_name = experiment["run_name"]
        print(f"\nRunning experiment: {run_name}")
        config = merge_config(base_config, experiment.get("overrides"))
        config["mlflow"]["run_name"] = run_name
        metrics, _ = train_and_log(config)
        results.append((run_name, metrics))

    print("\nSummary (test set):")
    print(f"  {'run_name':<30} {'roc_auc':>8} {'f1':>8} {'recall':>8} {'precision':>10}")
    for run_name, m in results:
        print(
            f"  {run_name:<30} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} "
            f"{m['recall']:>8.4f} {m['precision']:>10.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())