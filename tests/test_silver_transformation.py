"""Tests for Silver layer deduplication logic."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path so we can import Silver module
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "Preparation"))
sys.path.insert(0, "src/Preparation")

from tests.conftest import FakeSparkDataFrame

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def silver_obj(mock_spark, mock_logger):
    """Create a SilverTransformer instance with mocked dependencies."""
    from Silver.silver_transformation import SilverTransformer

    obj = object.__new__(SilverTransformer)
    obj.spark = mock_spark
    obj.logger = mock_logger
    obj.catalog_name = "oil_gas_demo"
    obj.bronze_schema_name = "bronze"
    obj.bronze_table_name = "raw_data"
    obj.silver_schema_name = "silver"
    obj.silver_table_name = "cleaned_data"
    obj.source_table = "oil_gas_demo.bronze.raw_data"
    obj.target_table = "oil_gas_demo.silver.cleaned_data"
    return obj


# ---------------------------------------------------------------------------
# Test: remove_duplicates
# ---------------------------------------------------------------------------


class TestRemoveDuplicates:
    """Test Silver deduplication: keep latest record per transaction_id."""

    def test_keeps_latest_record_when_duplicates_exist(self, silver_obj):
        """When multiple records share a transaction_id, keep the one with the latest _ingest_ts."""
        # Two records with transaction_id = "txn_001", different ingest timestamps
        silver_obj.df = FakeSparkDataFrame(
            pd.DataFrame(
                {
                    "transaction_id": ["txn_001", "txn_001", "txn_002"],
                    "_ingest_ts": [
                        pd.Timestamp("2024-01-01 10:00:00"),
                        pd.Timestamp("2024-01-01 12:00:00"),  # Latest for txn_001
                        pd.Timestamp("2024-01-01 11:00:00"),
                    ],
                    "value": [100, 200, 300],
                }
            )
        )

        silver_obj.remove_duplicates()

        # After deduplication, should have 2 rows: latest txn_001 (value=200) + txn_002
        result = silver_obj.df.toPandas()
        assert len(result) == 2
        assert result[result["transaction_id"] == "txn_001"]["value"].iloc[0] == 200
        assert result[result["transaction_id"] == "txn_002"]["value"].iloc[0] == 300

    def test_no_duplicates_returns_all_rows(self, silver_obj):
        """When all transaction_ids are unique, all rows are kept."""
        silver_obj.df = FakeSparkDataFrame(
            pd.DataFrame(
                {
                    "transaction_id": ["txn_001", "txn_002", "txn_003"],
                    "_ingest_ts": [
                        pd.Timestamp("2024-01-01 10:00:00"),
                        pd.Timestamp("2024-01-01 11:00:00"),
                        pd.Timestamp("2024-01-01 12:00:00"),
                    ],
                    "value": [100, 200, 300],
                }
            )
        )

        silver_obj.remove_duplicates()

        result = silver_obj.df.toPandas()
        assert len(result) == 3

    def test_three_duplicates_keeps_only_latest(self, silver_obj):
        """When a transaction_id appears 3 times, keep only the latest."""
        silver_obj.df = FakeSparkDataFrame(
            pd.DataFrame(
                {
                    "transaction_id": ["txn_001", "txn_001", "txn_001"],
                    "_ingest_ts": [
                        pd.Timestamp("2024-01-01 08:00:00"),
                        pd.Timestamp("2024-01-01 10:00:00"),
                        pd.Timestamp("2024-01-01 12:00:00"),  # Latest
                    ],
                    "value": [100, 200, 300],
                }
            )
        )

        silver_obj.remove_duplicates()

        result = silver_obj.df.toPandas()
        assert len(result) == 1
        assert result["value"].iloc[0] == 300
