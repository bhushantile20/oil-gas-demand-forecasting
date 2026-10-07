"""Unit tests for DemandForecastTraining.

These tests exercise the pure-logic methods (handle_missing_values,
chronological_split, prepare_features) without connecting to Spark,
MLflow, or Unity Catalog.  Spark DataFrames are replaced by lightweight
pandas-backed stubs via the FakeSparkDataFrame class in conftest.py.
"""

import sys
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure the source directory is importable
sys.path.insert(0, "src/Model_tranining_and_Prediction")

from tests.conftest import FakeSparkDataFrame


# ---------------------------------------------------------------------------
# Helper to build a partially-instantiated DemandForecastTraining object
# without going through __init__ (which would try to read from a live
# Spark session).  We call __new__ and set only the attributes each test
# needs.
# ---------------------------------------------------------------------------


def _make_training_obj(config, logger, **overrides):
    """Create a DemandForecastTraining instance bypassing __init__."""
    from Training.model_training import DemandForecastTraining

    obj = DemandForecastTraining.__new__(DemandForecastTraining)
    obj.logger = logger
    obj.spark = MagicMock()
    obj.ts_feature_cols = config["features"]["ts_feature_cols"]
    obj.cat_cols = config["features"]["cat_cols"]
    obj.drop_cols = config["features"]["drop_cols"]
    obj.split_train = config["split"]["train"]
    obj.split_val = config["split"]["val"]
    # Apply any test-specific overrides
    for key, value in overrides.items():
        setattr(obj, key, value)
    return obj


# ===========================================================================
# 1. handle_missing_values
# ===========================================================================


class TestHandleMissingValues:
    """Tests for DemandForecastTraining.handle_missing_values."""

    def test_nulls_filled_in_ts_feature_cols(self, training_config, mock_logger, sample_features_pdf):
        """Nulls in ts_feature_cols are replaced with 0.0; other columns untouched."""
        obj = _make_training_obj(training_config, mock_logger)
        obj.joined = FakeSparkDataFrame(sample_features_pdf)

        obj.handle_missing_values()

        result = obj.joined.toPandas()
        for col in training_config["features"]["ts_feature_cols"]:
            assert result[col].isna().sum() == 0, f"Column '{col}' still has nulls"
            # Verify that the filled values are 0.0 (not some other fill)
            filled_mask = sample_features_pdf[col].isna()
            if filled_mask.any():
                assert result.loc[filled_mask, col].tolist() == [0.0] * filled_mask.sum()

    def test_non_ts_columns_not_filled(self, training_config, mock_logger, sample_features_pdf):
        """Columns not listed in ts_feature_cols should retain any nulls they had."""
        # Add a non-ts column with nulls
        pdf = sample_features_pdf.copy()
        pdf["non_ts_nullable"] = [None] * len(pdf)
        obj = _make_training_obj(training_config, mock_logger)
        obj.joined = FakeSparkDataFrame(pdf)

        obj.handle_missing_values()

        result = obj.joined.toPandas()
        assert result["non_ts_nullable"].isna().all(), "Non-ts column should still have nulls"

    def test_nonexistent_ts_column_no_error(self, training_config, mock_logger, sample_features_pdf):
        """A ts_feature_col that doesn't exist in the DataFrame should be skipped gracefully."""
        config = {
            **training_config,
            "features": {
                **training_config["features"],
                "ts_feature_cols": ["demand_lag_1", "nonexistent_column"],
            },
        }
        obj = _make_training_obj(config, mock_logger)
        obj.joined = FakeSparkDataFrame(sample_features_pdf)

        # Should not raise
        obj.handle_missing_values()

        result = obj.joined.toPandas()
        assert result["demand_lag_1"].isna().sum() == 0


# ===========================================================================
# 2. chronological_split
# ===========================================================================


class TestChronologicalSplit:
    """Tests for DemandForecastTraining.chronological_split."""

    def test_correct_split_counts(self, training_config, mock_logger, mock_f_col):
        """With 10 distinct dates and 70/15/15 split, train=7, val=1, test=2."""
        dates = [date(2024, 1, d) for d in range(1, 11)]
        pdf = pd.DataFrame(
            {
                "feature_uid": [f"f{i}" for i in range(10)],
                "transaction_date": dates,
                "total_demand": list(range(10)),
            }
        )
        config = {
            **training_config,
            "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        }
        obj = _make_training_obj(config, mock_logger, split_train=0.7, split_val=0.15)
        obj.joined = FakeSparkDataFrame(pdf)

        obj.chronological_split()

        assert obj.train_df.count() == 7
        assert obj.val_df.count() == 1
        assert obj.test_df.count() == 2

    def test_no_date_overlap_between_splits(self, training_config, mock_logger, mock_f_col):
        """Train, val, and test date sets must be mutually disjoint."""
        dates = [date(2024, 1, d) for d in range(1, 11)]
        pdf = pd.DataFrame(
            {
                "feature_uid": [f"f{i}" for i in range(10)],
                "transaction_date": dates,
                "total_demand": list(range(10)),
            }
        )
        obj = _make_training_obj(
            {**training_config, "split": {"train": 0.7, "val": 0.15, "test": 0.15}},
            mock_logger,
            split_train=0.7,
            split_val=0.15,
        )
        obj.joined = FakeSparkDataFrame(pdf)

        obj.chronological_split()

        train_dates = set(obj.train_df.toPandas()["transaction_date"])
        val_dates = set(obj.val_df.toPandas()["transaction_date"])
        test_dates = set(obj.test_df.toPandas()["transaction_date"])

        assert train_dates.isdisjoint(val_dates)
        assert train_dates.isdisjoint(test_dates)
        assert val_dates.isdisjoint(test_dates)
        assert train_dates | val_dates | test_dates == set(dates)

    def test_chronological_order_preserved(self, training_config, mock_logger, mock_f_col):
        """Train split contains the earliest dates, test split the latest."""
        dates = [date(2024, 1, d) for d in range(1, 11)]
        pdf = pd.DataFrame(
            {
                "feature_uid": [f"f{i}" for i in range(10)],
                "transaction_date": dates,
                "total_demand": list(range(10)),
            }
        )
        obj = _make_training_obj(
            {**training_config, "split": {"train": 0.7, "val": 0.15, "test": 0.15}},
            mock_logger,
            split_train=0.7,
            split_val=0.15,
        )
        obj.joined = FakeSparkDataFrame(pdf)

        obj.chronological_split()

        train_dates = set(obj.train_df.toPandas()["transaction_date"])
        test_dates = set(obj.test_df.toPandas()["transaction_date"])

        assert max(train_dates) < min(test_dates), "Train dates should all precede test dates"


# ===========================================================================
# 3. prepare_features
# ===========================================================================


class TestPrepareFeatures:
    """Tests for DemandForecastTraining.prepare_features."""

    @pytest.fixture
    def training_obj_with_splits(self, training_config, mock_logger):
        """Build a training object with small train/val/test pandas-backed DataFrames."""
        pdf = pd.DataFrame(
            {
                "feature_uid": ["f1", "f2", "f3", "f4", "f5", "f6"],
                "transaction_date": [
                    date(2024, 1, 1), date(2024, 1, 1),
                    date(2024, 1, 2), date(2024, 1, 2),
                    date(2024, 1, 3), date(2024, 1, 3),
                ],
                "product_name": ["oil", "gas", "oil", "gas", "oil", "gas"],
                "destination_city": ["Houston", "Dallas", "Houston", "Dallas", "Houston", "Dallas"],
                "demand_lag_1": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "total_demand": [100, 200, 150, 250, 180, 300],
            }
        )
        # 4 train rows (first 2 dates), 2 val rows (3rd date) -- simple split
        obj = _make_training_obj(training_config, mock_logger)
        obj.train_df = FakeSparkDataFrame(pdf.iloc[:4])
        obj.val_df = FakeSparkDataFrame(pdf.iloc[4:5])
        obj.test_df = FakeSparkDataFrame(pdf.iloc[5:])
        return obj

    def test_encoded_columns_match_across_splits(self, training_obj_with_splits):
        """X_train, X_val, X_test must have identical column sets after OneHotEncoding."""
        obj = training_obj_with_splits

        obj.prepare_features()

        assert list(obj.X_train.columns) == list(obj.X_val.columns) == list(obj.X_test.columns)

    def test_dropped_columns_absent_from_features(self, training_obj_with_splits):
        """All columns in drop_cols should be removed from X_* DataFrames."""
        obj = training_obj_with_splits
        drop_cols = obj.drop_cols

        obj.prepare_features()

        for ds_name, ds in [("X_train", obj.X_train), ("X_val", obj.X_val), ("X_test", obj.X_test)]:
            for col in drop_cols:
                assert col not in ds.columns, f"'{col}' should be absent from {ds_name}"

    def test_y_values_match_total_demand(self, training_obj_with_splits):
        """y_train/y_val/y_test should equal the total_demand column from each split."""
        obj = training_obj_with_splits
        expected_y_train = [100, 200, 150, 250]

        obj.prepare_features()

        assert obj.y_train.tolist() == expected_y_train
        assert obj.y_val.tolist() == [180]
        assert obj.y_test.tolist() == [300]

    def test_onehot_column_names_follow_sklearn_convention(self, training_obj_with_splits):
        """Encoded column names should follow '<col>_<category>' naming."""
        obj = training_obj_with_splits

        obj.prepare_features()

        # With cat_cols = ["product_name", "destination_city"], categories are
        # oil/gas and Houston/Dallas, so we expect 4 encoded columns.
        encoded = [c for c in obj.X_train.columns if "_" in c and c not in obj.X_train.columns.difference(obj.encoded_cols)]
        # Verify the encoded_cols attribute is populated correctly
        assert "product_name_oil" in obj.encoded_cols
        assert "product_name_gas" in obj.encoded_cols
        assert "destination_city_Houston" in obj.encoded_cols
        assert "destination_city_Dallas" in obj.encoded_cols