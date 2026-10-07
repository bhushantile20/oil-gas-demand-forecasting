"""Silver tests — real local Spark, as required for this layer."""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "Preparation"))
sys.path.insert(0, "src/Preparation")

from Silver.silver_transformation import SilverTransformer


@pytest.fixture
def silver_obj(spark, mock_logger):
    return SilverTransformer(
        spark,
        mock_logger,
        "oil_gas_demo",
        "bronze",
        "raw_data",
        "silver",
        "cleaned_data",
    )


class TestRemoveDuplicates:

    def test_keeps_latest_record_when_duplicates_exist(self, silver_obj, spark):
        data = [
            ("txn_001", datetime(2024, 1, 1, 10, 0, 0), 100),
            ("txn_001", datetime(2024, 1, 1, 12, 0, 0), 200),  # latest
            ("txn_002", datetime(2024, 1, 1, 11, 0, 0), 300),
        ]
        silver_obj.df = spark.createDataFrame(
            data, ["transaction_id", "_ingest_ts", "value"]
        )

        silver_obj.remove_duplicates()

        result = silver_obj.df.toPandas()
        assert len(result) == 2
        txn_001 = result[result["transaction_id"] == "txn_001"]
        assert txn_001["value"].iloc[0] == 200

    def test_no_duplicates_keeps_all_rows(self, silver_obj, spark):
        data = [
            ("txn_001", datetime(2024, 1, 1, 10, 0, 0), 100),
            ("txn_002", datetime(2024, 1, 1, 11, 0, 0), 200),
            ("txn_003", datetime(2024, 1, 1, 12, 0, 0), 300),
        ]
        silver_obj.df = spark.createDataFrame(
            data, ["transaction_id", "_ingest_ts", "value"]
        )

        silver_obj.remove_duplicates()

        assert silver_obj.df.count() == 3
