"""Data validation tests. These load the real dataset pulled by DVC."""

import pytest

EXPECTED_ROW_MINIMUM = 1000

# Allowed ranges for numeric columns, based on how each field is defined.
EXPECTED_RANGES = {
    "Age": (18, 70),
    "MonthlyIncome": (1, 50000),
    "DistanceFromHome": (0, 100),
    "Education": (1, 5),
    "EnvironmentSatisfaction": (1, 4),
    "JobInvolvement": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "PerformanceRating": (1, 4),
    "RelationshipSatisfaction": (1, 4),
    "WorkLifeBalance": (1, 4),
    "StockOptionLevel": (0, 3),
    "NumCompaniesWorked": (0, 20),
    "TotalWorkingYears": (0, 50),
    "YearsAtCompany": (0, 50),
}


def test_dataset_has_enough_rows(raw_data):
    assert len(raw_data) >= EXPECTED_ROW_MINIMUM


def test_expected_columns_are_present(raw_data, config):
    expected = (
        config["features"]["numeric"]
        + config["features"]["categorical"]
        + config["data"]["drop_columns"]
        + [config["data"]["target_column"]]
    )
    missing = sorted(set(expected) - set(raw_data.columns))
    assert not missing, f"Missing columns: {missing}"


def test_target_contains_only_expected_values(raw_data, config):
    target = raw_data[config["data"]["target_column"]]
    allowed = {config["data"]["positive_label"], config["data"]["negative_label"]}
    assert set(target.unique()) == allowed


def test_target_has_both_classes_with_reasonable_balance(raw_data, config):
    target = raw_data[config["data"]["target_column"]]
    positive_share = (target == config["data"]["positive_label"]).mean()
    assert 0.05 < positive_share < 0.5


@pytest.mark.parametrize("column, bounds", EXPECTED_RANGES.items())
def test_numeric_features_are_within_expected_ranges(raw_data, column, bounds):
    low, high = bounds
    values = raw_data[column]
    assert values.min() >= low, f"{column} has values below {low}"
    assert values.max() <= high, f"{column} has values above {high}"


def test_categorical_features_have_no_blank_values(raw_data, config):
    for column in config["features"]["categorical"]:
        assert raw_data[column].notna().all(), f"{column} has missing values"
        assert (raw_data[column].str.strip() != "").all(), f"{column} has blank strings"


def test_employee_ids_are_unique(raw_data):
    assert raw_data["EmployeeNumber"].is_unique