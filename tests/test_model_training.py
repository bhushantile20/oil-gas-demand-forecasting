"""Training tests — Spark and model are mocked (no Databricks / MLflow on GitHub)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "Model_tranining_and_Prediction"))
sys.path.insert(0, "src/Model_tranining_and_Prediction")

from Training.model_training import DemandForecastTraining


def _sample_pdf():
    return pd.DataFrame(
        {
            "feature_uid": ["a", "b"],
            "transaction_date": ["2024-01-01", "2024-01-02"],
            "product_name": ["oil", "gas"],
            "destination_city": ["Mumbai", "Delhi"],
            "demand_lag_1": [1.0, 2.0],
            "total_demand": [10.0, 20.0],
        }
    )


def _fake_spark_df(pdf):
    fake = MagicMock()
    fake.toPandas.return_value = pdf.copy()
    return fake


@pytest.fixture
def trainer(mock_spark, mock_logger):
    config = {
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
            "ts_feature_cols": ["demand_lag_1"],
            "cat_cols": ["product_name", "destination_city"],
            "drop_cols": [
                "feature_uid",
                "transaction_date",
                "total_demand",
                "product_name",
                "destination_city",
            ],
        },
        "split": {"train": 0.64, "val": 0.16},
        "rf": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    }
    return DemandForecastTraining(mock_spark, mock_logger, "oil_gas_demo", config)


class TestPrepareFeatures:

    def test_creates_X_and_y(self, trainer):
        pdf = _sample_pdf()
        trainer.train_df = _fake_spark_df(pdf)
        trainer.val_df = _fake_spark_df(pdf)
        trainer.test_df = _fake_spark_df(pdf)

        trainer.prepare_features()

        assert len(trainer.y_train) == 2
        assert list(trainer.y_train) == [10.0, 20.0]

    def test_drops_target_from_X(self, trainer):
        pdf = _sample_pdf()
        trainer.train_df = _fake_spark_df(pdf)
        trainer.val_df = _fake_spark_df(pdf)
        trainer.test_df = _fake_spark_df(pdf)

        trainer.prepare_features()

        assert "total_demand" not in trainer.X_train.columns


class TestEvaluateModel:

    def test_perfect_predictions_mae_is_zero(self, trainer):
        trainer.X_val = pd.DataFrame({"x": [1, 2, 3]})
        trainer.X_test = pd.DataFrame({"x": [1, 2, 3]})
        trainer.y_val = pd.Series([10.0, 20.0, 30.0])
        trainer.y_test = pd.Series([10.0, 20.0, 30.0])

        trainer.model = MagicMock()
        trainer.model.predict.return_value = np.array([10.0, 20.0, 30.0])

        trainer.evaluate_model()

        assert trainer.val_metrics["mae"] == 0.0
        assert trainer.test_metrics["r2"] == 1.0

    def test_rmse_matches_sqrt_mse(self, trainer):
        trainer.X_val = pd.DataFrame({"x": [1, 2]})
        trainer.X_test = pd.DataFrame({"x": [1, 2]})
        trainer.y_val = pd.Series([10.0, 20.0])
        trainer.y_test = pd.Series([10.0, 20.0])

        trainer.model = MagicMock()
        trainer.model.predict.return_value = np.array([12.0, 18.0])

        trainer.evaluate_model()

        assert trainer.val_metrics["rmse"] == pytest.approx(trainer.val_metrics["mse"] ** 0.5)
