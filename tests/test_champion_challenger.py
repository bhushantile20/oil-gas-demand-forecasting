"""Unit tests for ChampionChallengerEvaluator.

These tests exercise the compare_performance and promote_challenger
methods without connecting to MLflow, Spark, or Unity Catalog.  Spark
DataFrames are replaced by pandas-backed stubs and the MLflow client is
mocked.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "src/Model_tranining_and_Prediction")

from tests.conftest import FakeSparkDataFrame


# ---------------------------------------------------------------------------
# Helper to bypass __init__
# ---------------------------------------------------------------------------


def _make_evaluator_obj(config, logger, **overrides):
    """Create a ChampionChallengerEvaluator instance bypassing __init__."""
    from Champion_Challenger.champion_challenger import ChampionChallengerEvaluator

    obj = ChampionChallengerEvaluator.__new__(ChampionChallengerEvaluator)
    obj.logger = logger
    obj.spark = MagicMock()
    obj.catalog_name = "oil_gas_demo"
    obj.config = config
    obj.model_name = f"oil_gas_demo.ml_model.random_forest_demand"
    obj.validation_table = f"oil_gas_demo.ml_model.validation_data"
    obj.champion_alias = config["model"]["champion_alias"]
    obj.challenger_alias = config["model"]["challenger_alias"]
    obj.client = MagicMock()
    obj.champion_version_number = None
    obj.challenger_version_number = None
    obj.comparison_df = None
    obj.champion_predictions = None
    obj.challenger_predictions = None
    for key, value in overrides.items():
        setattr(obj, key, value)
    return obj


# ===========================================================================
# 4. compare_performance
# ===========================================================================


class TestComparePerformance:
    """Tests for ChampionChallengerEvaluator.compare_performance."""

    def test_metric_values_correct(self, champion_challenger_config, mock_logger):
        """MAE, RMSE, and R2 are computed correctly for known predictions."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        champion_preds = np.array([110.0, 190.0, 310.0, 390.0])
        challenger_preds = np.array([105.0, 195.0, 305.0, 395.0])

        val_pdf = pd.DataFrame({"total_demand": y_true})
        obj = _make_evaluator_obj(
            champion_challenger_config, mock_logger,
            validation_df=FakeSparkDataFrame(val_pdf),
            champion_predictions=champion_preds,
            challenger_predictions=challenger_preds,
        )

        obj.compare_performance()

        df = obj.comparison_df
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 4)
        assert list(df.columns) == ["Model", "MAE", "RMSE", "R2"]

        # Verify champion metrics manually
        expected_champion_mae = mean_absolute_error(y_true, champion_preds)
        expected_champion_rmse = mean_squared_error(y_true, champion_preds) ** 0.5
        expected_champion_r2 = r2_score(y_true, champion_preds)

        champion_row = df[df["Model"] == "Champion"].iloc[0]
        assert pytest.approx(champion_row["MAE"]) == expected_champion_mae
        assert pytest.approx(champion_row["RMSE"]) == expected_champion_rmse
        assert pytest.approx(champion_row["R2"]) == expected_champion_r2

        # Verify challenger metrics manually
        expected_challenger_mae = mean_absolute_error(y_true, challenger_preds)
        expected_challenger_rmse = mean_squared_error(y_true, challenger_preds) ** 0.5
        expected_challenger_r2 = r2_score(y_true, challenger_preds)

        challenger_row = df[df["Model"] == "Challenger"].iloc[0]
        assert pytest.approx(challenger_row["MAE"]) == expected_challenger_mae
        assert pytest.approx(challenger_row["RMSE"]) == expected_challenger_rmse
        assert pytest.approx(challenger_row["R2"]) == expected_challenger_r2

    def test_dataframe_structure(self, champion_challenger_config, mock_logger):
        """comparison_df has exactly 2 rows, 4 columns, with correct Model labels."""
        y_true = np.array([1.0, 2.0])
        obj = _make_evaluator_obj(
            champion_challenger_config, mock_logger,
            validation_df=FakeSparkDataFrame(pd.DataFrame({"total_demand": y_true})),
            champion_predictions=np.array([1.1, 2.1]),
            challenger_predictions=np.array([0.9, 1.9]),
        )

        obj.compare_performance()

        df = obj.comparison_df
        assert df.shape == (2, 4)
        assert set(df["Model"].tolist()) == {"Champion", "Challenger"}
        assert list(df.columns) == ["Model", "MAE", "RMSE", "R2"]

    def test_challenger_better_when_lower_mae(self, champion_challenger_config, mock_logger):
        """When challenger MAE < champion MAE, the comparison reflects that."""
        y_true = np.array([100.0, 200.0])
        obj = _make_evaluator_obj(
            champion_challenger_config, mock_logger,
            validation_df=FakeSparkDataFrame(pd.DataFrame({"total_demand": y_true})),
            champion_predictions=np.array([120.0, 180.0]),  # higher error
            challenger_predictions=np.array([101.0, 199.0]),  # lower error
        )

        obj.compare_performance()

        df = obj.comparison_df
        champion_mae = df.loc[df["Model"] == "Champion", "MAE"].values[0]
        challenger_mae = df.loc[df["Model"] == "Challenger", "MAE"].values[0]
        assert challenger_mae < champion_mae


# ===========================================================================
# 5. promote_challenger
# ===========================================================================


class TestPromoteChallenger:
    """Tests for ChampionChallengerEvaluator.promote_challenger."""

    def test_promote_when_challenger_has_lower_mae(self, champion_challenger_config, mock_logger):
        """When challenger MAE < champion MAE, the challenger is promoted."""
        comparison_df = pd.DataFrame(
            {
                "Model": ["Champion", "Challenger"],
                "MAE": [20.0, 10.0],
                "RMSE": [25.0, 15.0],
                "R2": [0.80, 0.85],
            }
        )
        obj = _make_evaluator_obj(
            champion_challenger_config, mock_logger,
            comparison_df=comparison_df,
            champion_version_number=1,
            challenger_version_number=2,
        )
        # Mock the client's alias operations
        old_champion_mock = MagicMock()
        old_champion_mock.version = 1
        obj.client.get_model_version_by_alias.return_value = old_champion_mock

        obj.promote_challenger()

        # The source code calls set_registered_model_alias with positional args:
        #   set_registered_model_alias(model_name, alias, version)
        promote_calls = obj.client.set_registered_model_alias.call_args_list
        assert len(promote_calls) == 2
        # First call: demote old champion to previous-champion
        first_args = promote_calls[0].args
        assert first_args[1] == "previous-champion"
        assert first_args[2] == 1
        # Second call: promote challenger to champion
        second_args = promote_calls[1].args
        assert second_args[1] == "champion"
        assert second_args[2] == 2

    def test_keep_champion_when_challenger_has_higher_mae(self, champion_challenger_config, mock_logger):
        """When challenger MAE >= champion MAE, the champion is kept."""
        comparison_df = pd.DataFrame(
            {
                "Model": ["Champion", "Challenger"],
                "MAE": [10.0, 20.0],
                "RMSE": [15.0, 25.0],
                "R2": [0.85, 0.80],
            }
        )
        obj = _make_evaluator_obj(
            champion_challenger_config, mock_logger,
            comparison_df=comparison_df,
            champion_version_number=1,
            challenger_version_number=2,
        )

        obj.promote_challenger()

        # Verify no promotion happened
        obj.client.set_registered_model_alias.assert_not_called()

    def test_keep_champion_on_equal_mae(self, champion_challenger_config, mock_logger):
        """When MAE values are equal, the champion is kept (strict less-than comparison)."""
        comparison_df = pd.DataFrame(
            {
                "Model": ["Champion", "Challenger"],
                "MAE": [15.0, 15.0],
                "RMSE": [20.0, 20.0],
                "R2": [0.82, 0.82],
            }
        )
        obj = _make_evaluator_obj(
            champion_challenger_config, mock_logger,
            comparison_df=comparison_df,
            champion_version_number=1,
            challenger_version_number=2,
        )

        obj.promote_challenger()

        # Verify no promotion happened (equal is not strictly less)
        obj.client.set_registered_model_alias.assert_not_called()