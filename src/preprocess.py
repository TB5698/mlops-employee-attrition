"""Data loading and preprocessing for the employee attrition dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(path):
    """Load the raw CSV.

    The file starts with a byte order mark, so "utf-8-sig" is used to keep the
    first column named "Age" instead of "\\ufeffAge".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `dvc pull` to download it."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_data(df, drop_columns, target_column, positive_label="Yes", negative_label="No"):
    """Drop unused columns and convert the target to 1 (left) and 0 (stayed).

    Returns a new dataframe. The input dataframe is not modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    labels = set(df[target_column].dropna().unique())
    if not labels.issubset({positive_label, negative_label}):
        raise ValueError(f"Unexpected target values: {sorted(labels)}")

    cleaned = df.drop(columns=[c for c in drop_columns if c in df.columns])
    cleaned[target_column] = (cleaned[target_column] == positive_label).astype(int)
    return cleaned


def inject_missing_values(df, columns, rate, seed=42):
    """Randomly blank out a share of values in the given columns.

    The original dataset is fully clean, so this simulates the missing data a
    real pipeline would receive. Returns a new dataframe; the input is not modified.
    """
    if not 0 <= rate < 1:
        raise ValueError(f"rate must be between 0 and 1, got {rate}")
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in dataframe: {missing_cols}")

    result = df.copy()
    rng = np.random.default_rng(seed)
    n_missing = int(round(len(result) * rate))
    for col in columns:
        rows = rng.choice(result.index, size=n_missing, replace=False)
        result.loc[rows, col] = np.nan
    return result


def get_feature_lists(config):
    """Return the numeric and categorical feature names, minus any excluded ones."""
    exclude = set(config["features"].get("exclude") or [])
    numeric = [c for c in config["features"]["numeric"] if c not in exclude]
    categorical = [c for c in config["features"]["categorical"] if c not in exclude]
    return numeric, categorical


def split_features_target(df, target_column):
    """Split a cleaned dataframe into features (X) and target (y)."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")
    return df.drop(columns=[target_column]), df[target_column]


def build_preprocessor(numeric_features, categorical_features):
    """Build the sklearn transformer that imputes, scales, and encodes features.

    Numeric columns: fill missing values with the median, then standardize.
    Categorical columns: fill missing values with the most common value, then one-hot encode.
    The imputers learn their fill values from the training data only, which avoids leakage.
    """
    if not numeric_features and not categorical_features:
        raise ValueError("At least one numeric or categorical feature is required")
    overlap = set(numeric_features) & set(categorical_features)
    if overlap:
        raise ValueError(f"Features listed as both numeric and categorical: {sorted(overlap)}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )


def prepare_dataset(config):
    """Run the full data preparation defined in the config.

    Load the raw CSV, clean it, then add the simulated missing values.
    """
    data_cfg = config["data"]
    missing_cfg = config["missing_values"]

    raw = load_raw_data(data_cfg["raw_path"])
    cleaned = clean_data(
        raw,
        drop_columns=data_cfg["drop_columns"],
        target_column=data_cfg["target_column"],
        positive_label=data_cfg["positive_label"],
        negative_label=data_cfg["negative_label"],
    )
    return inject_missing_values(
        cleaned,
        columns=missing_cfg["columns"],
        rate=missing_cfg["rate"],
        seed=missing_cfg["seed"],
    )