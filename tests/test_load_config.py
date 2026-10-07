"""Tests for load_config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "Model_tranining_and_Prediction"))
sys.path.insert(0, "src/Model_tranining_and_Prediction")

from model_training_prediction import load_config


def test_load_config_returns_catalog_name(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "model_training_prediction:\n"
        "  catalog_name: oil_gas_demo\n"
    )

    result = load_config(str(config_file))

    assert result["model_training_prediction"]["catalog_name"] == "oil_gas_demo"


def test_load_config_reads_nested_keys(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(
        "model_training_prediction:\n"
        "  model_training:\n"
        "    rf:\n"
        "      n_estimators: 200\n"
    )

    result = load_config(str(config_file))

    assert result["model_training_prediction"]["model_training"]["rf"]["n_estimators"] == 200