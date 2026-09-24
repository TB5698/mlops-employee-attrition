"""Unit tests for the preprocessing functions in src/preprocess.py."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    build_preprocessor,
    clean_data,
    inject_missing_values,
    load_raw_data,
)

NUMERIC = ["Age", "MonthlyIncome"]
CATEGORICAL = ["Department", "OverTime"]


# ---------- clean_data ----------

def test_clean_data_encodes_target_as_zero_and_one(sample_df):
    cleaned = clean_data(sample_df, drop_columns=[], target_column="Attrition")
    assert cleaned["Attrition"].tolist() == [1, 0, 0, 1, 0, 0]


def test_clean_data_drops_configured_columns(sample_df):
    cleaned = clean_data(sample_df, drop_columns=["EmployeeCount"], target_column="Attrition")
    assert "EmployeeCount" not in cleaned.columns


def test_clean_data_does_not_modify_original(sample_df):
    original = sample_df.copy()
    clean_data(sample_df, drop_columns=["EmployeeCount"], target_column="Attrition")
    pd.testing.assert_frame_equal(sample_df, original)


def test_clean_data_raises_when_target_column_missing(sample_df):
    with pytest.raises(ValueError, match="Target column"):
        clean_data(sample_df.drop(columns=["Attrition"]), drop_columns=[], target_column="Attrition")


def test_clean_data_raises_on_unexpected_target_values(sample_df):
    bad = sample_df.copy()
    bad.loc[0, "Attrition"] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected target values"):
        clean_data(bad, drop_columns=[], target_column="Attrition")


def test_clean_data_raises_on_non_dataframe_input():
    with pytest.raises(TypeError):
        clean_data([1, 2, 3], drop_columns=[], target_column="Attrition")


# ---------- inject_missing_values ----------

def test_inject_missing_values_adds_expected_number_of_nans(sample_df):
    complete = sample_df.dropna().reset_index(drop=True)
    result = inject_missing_values(complete, columns=["MonthlyIncome"], rate=0.5, seed=0)
    assert result["MonthlyIncome"].isna().sum() == 2


def test_inject_missing_values_does_not_modify_original(sample_df):
    original = sample_df.copy()
    inject_missing_values(sample_df, columns=["Age", "OverTime"], rate=0.5, seed=0)
    pd.testing.assert_frame_equal(sample_df, original)


@pytest.mark.parametrize("bad_rate", [-0.1, 1.0, 1.5])
def test_inject_missing_values_rejects_invalid_rate(sample_df, bad_rate):
    with pytest.raises(ValueError, match="rate must be between"):
        inject_missing_values(sample_df, columns=["Age"], rate=bad_rate)


def test_inject_missing_values_rejects_unknown_column(sample_df):
    with pytest.raises(KeyError, match="NotAColumn"):
        inject_missing_values(sample_df, columns=["NotAColumn"], rate=0.1)


# ---------- build_preprocessor ----------

def test_preprocessor_fills_all_missing_values(sample_df):
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL)
    output = preprocessor.fit_transform(sample_df[NUMERIC + CATEGORICAL])
    assert not np.isnan(output).any()


def test_preprocessor_one_hot_encodes_categorical_columns(sample_df):
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL)
    output = preprocessor.fit_transform(sample_df[NUMERIC + CATEGORICAL])
    # 2 numeric columns + 3 departments + 2 overtime values = 7 output columns
    assert output.shape == (len(sample_df), 7)
    encoded = output[:, len(NUMERIC):]
    assert set(np.unique(encoded)) <= {0.0, 1.0}
    # Each row has exactly one "1" per categorical column
    assert (encoded.sum(axis=1) == len(CATEGORICAL)).all()


def test_preprocessor_ignores_unseen_categories(sample_df):
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL)
    preprocessor.fit(sample_df[NUMERIC + CATEGORICAL])
    new_row = sample_df[NUMERIC + CATEGORICAL].head(1).copy()
    new_row["Department"] = "Marketing"
    output = preprocessor.transform(new_row)
    assert output.shape == (1, 7)


def test_preprocessor_does_not_modify_input(sample_df):
    original = sample_df.copy()
    build_preprocessor(NUMERIC, CATEGORICAL).fit_transform(sample_df[NUMERIC + CATEGORICAL])
    pd.testing.assert_frame_equal(sample_df, original)


def test_build_preprocessor_rejects_empty_feature_lists():
    with pytest.raises(ValueError, match="At least one"):
        build_preprocessor([], [])


def test_build_preprocessor_rejects_overlapping_features():
    with pytest.raises(ValueError, match="both numeric and categorical"):
        build_preprocessor(["Age"], ["Age"])


# ---------- load_raw_data ----------

def test_load_raw_data_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="dvc pull"):
        load_raw_data(tmp_path / "does_not_exist.csv")