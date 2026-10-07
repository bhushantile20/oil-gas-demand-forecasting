"""Unit tests for ModelPredictions.

These tests exercise the evaluate method without connecting to MLflow,
Spark, or Unity Catalog.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src/Model_tranining_and_Prediction")


# ---------------------------------------------------------------------------
# Helper to bypass __init__
# ---------------------------------------------------------------------------


def _make_predictions_obj(config, logger, **overrides):
    """Create a ModelPredictions instance bypassing __init__."""
    from Prediction.model_predictions import ModelPredictions

    obj = ModelPredictions.__new__(ModelPredictions)
    obj.logger = logger
    obj.spark = MagicMock()
    obj.catalog_name = "oil_gas_demo"
    obj.config = config
    obj.test_table = "oil_gas_demo.ml_model.testing_data"
    obj.model_name = "oil_gas_demo.ml_model.random_forest_demand"
    obj.output_table = "oil_gas_demo.ml_model.model_predictions_eval"
    obj.client = MagicMock()
    obj.data = None
    obj.data_pd = None
    obj.encoder = None
    obj.X_test = None
    obj.y_test = None
    obj.predictions = None
    obj.latest_version = None
    for key, value in overrides.items():
        setattr(obj, key, value)
    return obj


# ===========================================================================
# 6. evaluate (bonus — pure metric computation)
# ===========================================================================


class TestEvaluate:
    """Tests for ModelPredictions.evaluate."""

    def test_perfect_predictions_return_ideal_metrics(self, predictions_config, mock_logger):
        """When predictions == y_test, MAE=0, MSE=0, R2=1."""
        y = pd.Series([100.0, 200.0, 300.0, 400.0])
        obj = _make_predictions_obj(
            predictions_config, mock_logger,
            y_test=y,
            predictions=np.array([100.0, 200.0, 300.0, 400.0]),
        )

        mae, mse, r2 = obj.evaluate()

        assert pytest.approx(mae) == 0.0
        assert pytest.approx(mse) == 0.0
        assert pytest.approx(r2) == 1.0

    def test_return_type_and_ordering(self, predictions_config, mock_logger):
        """Returns a 3-tuple of floats in (mae, mse, r2) order."""
        y = pd.Series([1.0, 2.0, 3.0])
        preds = np.array([1.5, 2.5, 2.5])
        obj = _make_predictions_obj(
            predictions_config, mock_logger,
            y_test=y,
            predictions=preds,
        )

        result = obj.evaluate()

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

        # Verify the ordering is (mae, mse, r2) — not (mae, rmse, r2)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        expected_mae = mean_absolute_error(y, preds)
        expected_mse = mean_squared_error(y, preds)
        expected_r2 = r2_score(y, preds)

        assert pytest.approx(result[0]) == expected_mae
        assert pytest.approx(result[1]) == expected_mse  # MSE, not RMSE
        assert pytest.approx(result[2]) == expected_r2

    def test_known_metric_values(self, predictions_config, mock_logger):
        """Metrics are computed correctly for hand-calculable predictions."""
        y = pd.Series([10.0, 20.0, 30.0])
        preds = np.array([12.0, 18.0, 33.0])
        obj = _make_predictions_obj(
            predictions_config, mock_logger,
            y_test=y,
            predictions=preds,
        )

        mae, mse, r2 = obj.evaluate()

        # MAE = mean(|2, 2, 3|) = 7/3
        assert pytest.approx(mae) == 7.0 / 3.0
        # MSE = mean(4, 4, 9) = 17/3
        assert pytest.approx(mse) == 17.0 / 3.0
        # R2 = 1 - (17/3) / variance-based TSS
        from sklearn.metrics import r2_score
        assert pytest.approx(r2) == r2_score(y, preds)

    def test_negative_r2_for_very_bad_predictions(self, predictions_config, mock_logger):
        """R2 should be negative when predictions are worse than the mean."""
        y = pd.Series([0.0, 10.0, 20.0])
        # Predict the opposite pattern — very bad
        preds = np.array([100.0, 0.0, 100.0])
        obj = _make_predictions_obj(
            predictions_config, mock_logger,
            y_test=y,
            predictions=preds,
        )

        mae, mse, r2 = obj.evaluate()

        assert r2 < 0, "R2 should be negative for very poor predictions"