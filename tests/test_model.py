"""Model validation tests: train on a small sample and check the predictions."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.evaluate import check_thresholds, compute_metrics
from src.preprocess import get_feature_lists
from src.train import build_model, build_pipeline, load_train_test

MIN_TEST_ROC_AUC = 0.70
SAMPLE_SIZE = 600


@pytest.fixture(scope="module")
def trained_model(config):
    """Train the configured pipeline on a stratified sample of the training data."""
    X_train, X_test, y_train, y_test = load_train_test(config)
    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=SAMPLE_SIZE,
        stratify=y_train,
        random_state=config["split"]["random_state"],
    )
    pipeline = build_pipeline(config)
    pipeline.fit(X_sample, y_sample)
    return pipeline, X_test, y_test


def test_predictions_have_correct_type_and_shape(trained_model):
    pipeline, X_test, _ = trained_model
    predictions = pipeline.predict(X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(X_test),)
    assert set(np.unique(predictions)) <= {0, 1}


def test_predicted_probabilities_are_valid(trained_model):
    pipeline, X_test, _ = trained_model
    probabilities = pipeline.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_model_meets_minimum_performance(trained_model):
    pipeline, X_test, y_test = trained_model
    roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    assert roc_auc >= MIN_TEST_ROC_AUC, f"ROC-AUC {roc_auc:.3f} is below {MIN_TEST_ROC_AUC}"


def test_model_handles_missing_values_at_prediction_time(trained_model, config):
    pipeline, X_test, _ = trained_model
    numeric, categorical = get_feature_lists(config)
    rows = X_test.head(3).copy()
    rows.loc[:, [numeric[0], categorical[0]]] = np.nan
    assert pipeline.predict(rows).shape == (3,)


def test_build_model_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model({"type": "not_a_model", "params": {}})


def test_check_thresholds_reports_failures():
    metrics = compute_metrics(
        y_true=[0, 1, 1, 0], y_pred=[0, 0, 1, 1], y_proba=[0.2, 0.4, 0.9, 0.6]
    )
    failures = check_thresholds(metrics, {"roc_auc": 0.99, "accuracy": 0.1})
    assert len(failures) == 1
    assert "roc_auc" in failures[0]