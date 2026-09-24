"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import load_raw_data
from src.utils import load_config

CONFIG_PATH = "configs/config.yaml"


@pytest.fixture(scope="session")
def config():
    """The project config, loaded once for the whole test session."""
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def raw_data(config):
    """The real dataset tracked by DVC. Run `dvc pull` first if this fails."""
    return load_raw_data(config["data"]["raw_path"])


@pytest.fixture
def sample_df():
    """A small hand-made dataframe with the same kinds of columns as the real data."""
    return pd.DataFrame(
        {
            "Age": [25, 40, 35, np.nan, 50, 29],
            "MonthlyIncome": [3000.0, 8000.0, np.nan, 4500.0, 12000.0, 2800.0],
            "Department": ["Sales", "Research & Development", "Sales", None, "Human Resources", "Sales"],
            "OverTime": ["Yes", "No", "No", "Yes", "No", "Yes"],
            "EmployeeCount": [1, 1, 1, 1, 1, 1],
            "Attrition": ["Yes", "No", "No", "Yes", "No", "No"],
        }
    )