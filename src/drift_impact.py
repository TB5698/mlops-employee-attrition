"""Estimate how the simulated drift affects model performance.

Trains the configured model, then scores it on the original held-out rows and on
the same rows after the simulated drift is applied. The true labels are the same
in both cases, so any change in the metrics comes from the shifted inputs alone.

Usage:
    python -m src.drift_impact --config configs/config.yaml
"""

import argparse
import sys

from src.evaluate import compute_metrics
from src.monitor_drift import simulate_drift
from src.train import build_pipeline, load_train_test
from src.utils import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the effect of drift on the model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    X_train, X_test, y_train, y_test = load_train_test(config)
    model = build_pipeline(config).fit(X_train, y_train)

    drift_cfg = config["monitoring"]["simulated_drift"]
    X_drifted = simulate_drift(X_test, drift_cfg["changes"], seed=drift_cfg["seed"])

    print(f"  {'data':<20} {'roc_auc':>8} {'f1':>7} {'recall':>7} {'precision':>10} "
          f"{'accuracy':>9} {'flagged_to_leave':>17}")
    for name, X in [("original test set", X_test), ("drifted batch", X_drifted)]:
        y_pred = model.predict(X)
        m = compute_metrics(y_test, y_pred, model.predict_proba(X)[:, 1])
        print(
            f"  {name:<20} {m['roc_auc']:>8.3f} {m['f1']:>7.3f} {m['recall']:>7.3f} "
            f"{m['precision']:>10.3f} {m['accuracy']:>9.3f} {y_pred.mean():>17.1%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())