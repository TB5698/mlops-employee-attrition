"""Query MLflow for all runs in the experiment and report the best one.

Usage:
    python -m src.compare_experiments --config configs/config.yaml
"""

import argparse
import sys

import mlflow

from src.utils import load_config


def get_runs(config):
    """Return every finished run in the experiment, best primary metric first."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    metric = config["evaluation"]["primary_metric"]
    return mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare MLflow experiment runs.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    metric = config["evaluation"]["primary_metric"]
    runs = get_runs(config)

    if runs.empty:
        print("No finished runs found. Train a model first with: python -m src.train")
        return 1

    columns = {
        "tags.mlflow.runName": "run_name",
        "params.model_type": "model_type",
        f"metrics.{metric}": metric,
        "metrics.f1": "f1",
        "metrics.recall": "recall",
        "metrics.precision": "precision",
        "params.data_version": "data_version",
    }
    table = runs[list(columns)].rename(columns=columns)
    table["data_version"] = table["data_version"].str[:8]

    print(f"Found {len(table)} runs, sorted by {metric} (primary metric):\n")
    print(table.to_string(index=False, float_format="{:.4f}".format))

    best = runs.iloc[0]
    print("\nBest run:")
    print(f"  run name:   {best['tags.mlflow.runName']}")
    print(f"  run ID:     {best['run_id']}")
    print(f"  model type: {best['params.model_type']}")
    print(f"  {metric}:    {best[f'metrics.{metric}']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())