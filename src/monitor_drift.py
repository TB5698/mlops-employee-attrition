"""Drift monitoring with Evidently.

Compares the training data (reference) against a simulated batch of production
data, prints which features drifted, and saves an HTML report to reports/.

Usage:
    python -m src.monitor_drift --config configs/config.yaml
    python -m src.monitor_drift --threshold 0.1     (override the configured threshold)

Exits with code 1 if the share of drifted features is above the configured threshold.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.preprocess import get_feature_lists
from src.train import load_train_test
from src.utils import load_config

# Evidently tests that return a p-value. The summary below reads every score as a
# p-value (drift when p < threshold), so only these tests are accepted in the config.
P_VALUE_TESTS = {
    "ks", "chisquare", "z", "anderson", "cramer_von_mises",
    "mannw", "t_test", "fisher_exact", "g_test",
}


def simulate_drift(df, changes, seed=0):
    """Return a copy of df with the configured distribution changes applied.

    Supported change types:
      shift:        add a fixed value to a numeric column (values stay at or above 0)
      scale:        multiply a numeric column by a factor
      set_fraction: set a random fraction of rows in a column to one value
    """
    drifted = df.copy()
    rng = np.random.default_rng(seed)
    for change in changes:
        column, kind = change["column"], change["type"]
        if column not in drifted.columns:
            raise KeyError(f"Drift column not found: {column}")

        if kind == "shift":
            drifted[column] = (drifted[column] + change["value"]).clip(lower=0)
        elif kind == "scale":
            drifted[column] = drifted[column] * change["value"]
        elif kind == "set_fraction":
            n_rows = int(round(len(drifted) * change["fraction"]))
            rows = rng.choice(drifted.index, size=n_rows, replace=False)
            drifted.loc[rows, column] = change["value"]
        else:
            raise ValueError(f"Unknown drift type: {kind}")
    return drifted


def load_reference_and_production(config):
    """Reference = the training split. Production = the held-out split with drift applied.

    The held-out rows were never used for training, so they stand in for new
    employees' records arriving after deployment.
    """
    X_train, X_test, _, _ = load_train_test(config)
    drift_cfg = config["monitoring"]["simulated_drift"]
    production = simulate_drift(X_test, drift_cfg["changes"], seed=drift_cfg["seed"])
    return X_train, production


def run_drift_report(reference, production, config):
    """Run Evidently's data drift preset on every feature and return the snapshot."""
    monitor_cfg = config["monitoring"]
    for key in ("num_method", "cat_method"):
        if monitor_cfg[key] not in P_VALUE_TESTS:
            raise ValueError(
                f"monitoring.{key} must be a p-value test, one of {sorted(P_VALUE_TESTS)}"
            )

    numeric, categorical = get_feature_lists(config)
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)

    report = Report(
        [
            DataDriftPreset(
                num_method=monitor_cfg["num_method"],
                cat_method=monitor_cfg["cat_method"],
                threshold=monitor_cfg["stattest_threshold"],
                drift_share=monitor_cfg["drift_share_threshold"],
            )
        ]
    )
    return report.run(
        Dataset.from_pandas(production, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )


def summarize_drift(snapshot):
    """Pull the per-feature results and the overall drift share out of the snapshot."""
    features = []
    drift_share = None
    for metric in snapshot.dict()["metrics"]:
        settings = metric["config"]
        if settings["type"].endswith("DriftedColumnsCount"):
            drift_share = metric["value"]["share"]
        elif settings["type"].endswith("ValueDrift"):
            p_value = metric["value"]
            features.append(
                {
                    "feature": settings["column"],
                    "test": settings["method"],
                    "p_value": p_value,
                    "drifted": p_value < settings["threshold"],
                }
            )
    return features, drift_share


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check for data drift with Evidently.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override the drift share threshold from the config (for example 0.1)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    monitor_cfg = config["monitoring"]
    if args.threshold is not None:
        monitor_cfg["drift_share_threshold"] = args.threshold

    reference, production = load_reference_and_production(config)
    snapshot = run_drift_report(reference, production, config)
    features, drift_share = summarize_drift(snapshot)

    report_dir = Path(monitor_cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / monitor_cfg["report_name"]
    snapshot.save_html(str(report_path))

    drifted = [f for f in features if f["drifted"]]
    print(f"Reference rows: {len(reference)}   Production rows: {len(production)}")
    print(f"Features checked: {len(features)}   Drifted: {len(drifted)}\n")
    print(f"  {'feature':<26} {'test':<10} {'p_value':>10}  drifted")
    for f in sorted(features, key=lambda f: (not f["drifted"], f["feature"])):
        flag = "YES" if f["drifted"] else "no"
        print(f"  {f['feature']:<26} {f['test']:<10} {f['p_value']:>10.4g}  {flag}")

    threshold = monitor_cfg["drift_share_threshold"]
    print(f"\nOverall drift share: {drift_share:.1%} (threshold: {threshold:.0%})")
    print(f"HTML report saved to: {report_path}")

    if drift_share > threshold:
        print("\nALERT: drift share is above the threshold. Investigate and consider retraining.")
        return 1

    print("\nOK: drift share is within the threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())