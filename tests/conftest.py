"""Shared pytest fixtures for unit tests.

These fixtures build lightweight Spark DataFrame stubs backed by real
pandas DataFrames so tests never need a live Databricks connection.
"""

import logging
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helper: a minimal fake Spark DataFrame that wraps a pandas DataFrame.
# It supports the subset of methods used by the code under test:
#   .count(), .columns, .fillna(), .withColumnRenamed(), .join(),
#   .drop(), .select(), .distinct(), .orderBy(), .collect(),
#   .filter(), .toPandas()
# ---------------------------------------------------------------------------


class FakeSparkDataFrame:
    """A pandas-backed stub that mimics the Spark DataFrame methods used in tests."""

    def __init__(self, pdf: pd.DataFrame):
        self._pdf = pdf.copy()

    # --- read-only ----------------------------------------------------------
    @property
    def columns(self):
        return list(self._pdf.columns)

    def count(self):
        return len(self._pdf)

    def toPandas(self):
        return self._pdf.copy()

    def select(self, *cols):
        if len(cols) == 1 and isinstance(cols[0], list):
            cols = cols[0]
        return FakeSparkDataFrame(self._pdf[list(cols)])

    def distinct(self):
        return FakeSparkDataFrame(self._pdf.drop_duplicates())

    def collect(self):
        return [row for _, row in self._pdf.iterrows()]

    # --- transformations ----------------------------------------------------
    def fillna(self, value_dict):
        pdf = self._pdf.copy()
        pdf = pdf.fillna(value_dict)
        return FakeSparkDataFrame(pdf)

    def withColumnRenamed(self, old, new):
        pdf = self._pdf.rename(columns={old: new})
        return FakeSparkDataFrame(pdf)

    def drop(self, col):
        pdf = self._pdf.drop(columns=[col], errors="ignore")
        return FakeSparkDataFrame(pdf)

    def join(self, other, on, how="inner"):
        # `on` is a Column expression (feature_uid == label_feature_uid)
        # For the test we know the join key semantics; just merge on the
        # underlying column names.
        left_key = "feature_uid"
        right_key = "label_feature_uid"
        pdf = self._pdf.merge(other._pdf, left_on=left_key, right_on=right_key, how=how)
        pdf = pdf.drop(columns=[right_key])
        return FakeSparkDataFrame(pdf)

    def filter(self, condition):
        # condition can be:
        #   1. A pandas Series (boolean mask) — used directly.
        #   2. A MockColumn (see mock_f_col below) — carries the column
        #      name and list of isin values.
        #   3. A callable — apply it to the pandas DataFrame.
        if isinstance(condition, pd.Series):
            return FakeSparkDataFrame(self._pdf[condition.values].copy())
        if callable(condition):
            mask = condition(self._pdf)
            return FakeSparkDataFrame(self._pdf[mask].copy())
        # Handle our MockColumn helper
        if hasattr(condition, "_col_name") and hasattr(condition, "_isin_values"):
            mask = self._pdf[condition._col_name].isin(condition._isin_values)
            return FakeSparkDataFrame(self._pdf[mask].copy())
        raise TypeError(
            f"Unsupported filter condition type: {type(condition)}."
        )

    def orderBy(self, *cols):
        if len(cols) == 1 and isinstance(cols[0], str):
            cols = [cols[0]]
        return FakeSparkDataFrame(self._pdf.sort_values(by=list(cols)).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Helper: a MockColumn that replaces F.col(...).isin(list) so tests
# do not need a real Spark session.  FakeSparkDataFrame.filter checks
# for _col_name and _isin_values attributes.
# ---------------------------------------------------------------------------


class MockColumn:
    """A lightweight stand-in for pyspark.sql.Column that captures isin() calls."""

    def __init__(self, col_name):
        self._col_name = col_name
        self._isin_values = None

    def isin(self, values):
        self._isin_values = values
        return self


@pytest.fixture
def mock_f_col(monkeypatch):
    """Patch pyspark.sql.functions.col to return MockColumn objects."""
    import pyspark.sql.functions as F_mod

    def _fake_col(name):
        return MockColumn(name)

    monkeypatch.setattr(F_mod, "col", _fake_col)
    return _fake_col


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_logger():
    """A no-op logger that silently absorbs .info() calls."""
    logger = MagicMock(spec=logging.Logger)
    return logger


@pytest.fixture
def training_config():
    """Minimal config dict matching the model_training section of config.yml."""
    return {
        "source": {
            "feature_schema": "gold",
            "feature_table": "demand_feature_store",
            "label_schema": "gold",
            "label_table": "demand_labels",
        },
        "target": {
            "ml_schema": "ml_model",
            "train_table": "training_data",
            "val_table": "validation_data",
            "test_table": "testing_data",
            "predictions_table": "predictions",
        },
        "model": {
            "model_name": "random_forest_demand",
            "experiment_name": "/Shared/demand_forecasting_experiment",
        },
        "features": {
            "ts_feature_cols": ["demand_lag_1", "demand_lag_7", "rolling_avg_7"],
            "cat_cols": ["product_name", "destination_city"],
            "drop_cols": [
                "feature_uid",
                "transaction_date",
                "total_demand",
                "product_name",
                "destination_city",
            ],
        },
        "split": {
            "train": 0.64,
            "val": 0.16,
            "test": 0.20,
        },
        "rf": {
            "n_estimators": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
    }


@pytest.fixture
def champion_challenger_config():
    """Minimal config dict matching the champion_challenger section."""
    return {
        "source": {
            "ml_schema": "ml_model",
            "validation_table": "validation_data",
        },
        "model": {
            "model_name": "random_forest_demand",
            "champion_alias": "champion",
            "challenger_alias": "challenger",
        },
    }


@pytest.fixture
def predictions_config():
    """Minimal config dict matching the model_predictions section."""
    return {
        "source": {
            "ml_schema": "ml_model",
            "test_table": "testing_data",
        },
        "target": {
            "output_table": "model_predictions_eval",
        },
        "model": {
            "model_name": "random_forest_demand",
        },
    }


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_features_pdf():
    """A small pandas DataFrame mimicking the feature store table."""
    return pd.DataFrame(
        {
            "feature_uid": [f"f{i}" for i in range(12)],
            "transaction_date": [
                date(2024, 1, 1), date(2024, 1, 1),
                date(2024, 1, 2), date(2024, 1, 2),
                date(2024, 1, 3), date(2024, 1, 3),
                date(2024, 1, 4), date(2024, 1, 4),
                date(2024, 1, 5), date(2024, 1, 5),
                date(2024, 1, 6), date(2024, 1, 6),
            ],
            "product_name": ["oil", "gas"] * 6,
            "destination_city": ["Houston", "Dallas"] * 6,
            "demand_lag_1": [10.0, 20.0, None, 40.0, 50.0, None, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
            "demand_lag_7": [5.0, None, 15.0, 20.0, None, 30.0, 35.0, 40.0, 45.0, 50.0, None, 60.0],
            "rolling_avg_7": [7.5, 10.0, 12.5, 30.0, 37.5, None, 52.5, 60.0, 67.5, 75.0, 82.5, 90.0],
        }
    )


@pytest.fixture
def sample_labels_pdf():
    """A small pandas DataFrame mimicking the labels table."""
    return pd.DataFrame(
        {
            "feature_uid": [f"f{i}" for i in range(12)],
            "total_demand": [100, 200, 150, 250, 180, 300, 220, 350, 280, 400, 320, 450],
        }
    )