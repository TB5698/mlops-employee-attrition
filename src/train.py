"""Train the attrition model, log the run to MLflow, and check performance thresholds.

Usage:
    python -m src.train --config configs/config.yaml

Exits with code 1 if the model misses any threshold in the config's evaluation section.
"""

import argparse
import sys

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import check_thresholds, compute_metrics
from src.preprocess import (
    build_preprocessor,
    get_feature_lists,
    prepare_dataset,
    split_features_target,
)
from src.utils import file_md5, load_config

# MLflow saves sklearn models in the "skops" format, which refuses to load object
# types it has not been told to trust. These are the types our own models use:
# OneHotEncoder stores a numpy dtype, and the tree models store sklearn Tree objects.
TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]

# Model types the config can ask for.
MODELS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_model(model_config):
    """Create the classifier described in the config's model section."""
    model_type = model_config["type"]
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type '{model_type}'. Options: {sorted(MODELS)}")
    return MODELS[model_type](**(model_config.get("params") or {}))


def build_pipeline(config):
    """Chain the preprocessor and the model so they are trained and saved together."""
    numeric, categorical = get_feature_lists(config)
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric, categorical)),
            ("model", build_model(config["model"])),
        ]
    )


def load_train_test(config):
    """Prepare the dataset and split it into stratified train and test sets."""
    df = prepare_dataset(config)
    X, y = split_features_target(df, config["data"]["target_column"])
    numeric, categorical = get_feature_lists(config)
    # Numeric columns are stored as floats so a missing value at prediction time
    # does not break the input schema MLflow saves with the model.
    X = X[numeric + categorical].astype({col: "float64" for col in numeric})
    return train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=y,
    )


def collect_params(config):
    """Gather the settings to log as MLflow parameters."""
    numeric, categorical = get_feature_lists(config)
    params = {
        "model_type": config["model"]["type"],
        "test_size": config["split"]["test_size"],
        "random_state": config["split"]["random_state"],
        "missing_rate": config["missing_values"]["rate"],
        "n_numeric_features": len(numeric),
        "n_categorical_features": len(categorical),
        "excluded_features": ",".join(config["features"].get("exclude") or []) or "none",
    }
    for name, value in (config["model"].get("params") or {}).items():
        params[f"model__{name}"] = value
    return params


def train_and_log(config):
    """Train one model, evaluate it on the test set, and log everything to MLflow.

    Returns the metrics dictionary and the MLflow run ID.
    """
    X_train, X_test, y_train, y_test = load_train_test(config)

    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    data_version = file_md5(config["data"]["raw_path"])

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=config["mlflow"]["run_name"]) as run:
        mlflow.log_params(collect_params(config))
        mlflow.log_param("data_version", data_version)
        mlflow.set_tag("data_source", config["data"]["raw_path"])
        mlflow.set_tag("dvc_file", config["data"]["dvc_file"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_test, y_pred),
            input_example=X_train.head(5),
            skops_trusted_types=TRUSTED_TYPES,
        )

    return metrics, run.info.run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the employee attrition model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to the YAML config")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    metrics, run_id = train_and_log(config)

    print(f"\nMLflow run ID: {run_id}")
    print("Test set metrics:")
    for name, value in metrics.items():
        print(f"  {name:<10} {value:.4f}")

    failures = check_thresholds(metrics, config["evaluation"]["thresholds"])
    if failures:
        print("\nFAILED: the model did not meet the minimum thresholds:")
        for message in failures:
            print(f"  - {message}")
        return 1

    print("\nPASSED: all performance thresholds were met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())