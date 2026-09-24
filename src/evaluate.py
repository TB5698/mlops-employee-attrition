"""Model evaluation: metric calculation and threshold checks."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba):
    """Return the evaluation metrics for a binary classifier.

    y_pred holds the predicted classes (0 or 1).
    y_proba holds the predicted probability of class 1 (the employee leaves).
    """
    if len(y_true) != len(y_pred) or len(y_true) != len(y_proba):
        raise ValueError("y_true, y_pred, and y_proba must be the same length")

    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def check_thresholds(metrics, thresholds):
    """Compare metrics against minimum thresholds.

    Returns a list of readable failure messages. An empty list means every check passed.
    """
    failures = []
    for name, minimum in thresholds.items():
        if name not in metrics:
            raise KeyError(f"Threshold set for unknown metric: {name}")
        if metrics[name] < minimum:
            failures.append(f"{name} = {metrics[name]:.4f} is below the minimum of {minimum}")
    return failures