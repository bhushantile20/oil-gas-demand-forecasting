"""Shared pytest fixtures for unit tests.

These fixtures build lightweight Spark DataFrame stubs backed by real
pandas DataFrames so tests never need a live Databricks connection.
"""

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helper: a minimal fake Spark DataFrame that wraps a pandas DataFrame.
# Supports methods used by Silver layer deduplication tests:
#   .count(), .columns, .toPandas(), .withColumns(), .filter(), .drop()
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

    def drop(self, col):
        pdf = self._pdf.drop(columns=[col], errors="ignore")
        return FakeSparkDataFrame(pdf)

    def withColumns(self, col_dict):
        """Add or replace columns. For deduplication tests, handles _row_num window function."""
        pdf = self._pdf.copy()
        for col_name, col_expr in col_dict.items():
            if col_name == "_row_num":
                # Special case: simulate row_number() window function for deduplication
                # The Silver transformation uses: partition by transaction_id, order by _ingest_ts desc
                pdf["_row_num"] = (
                    pdf.sort_values("_ingest_ts", ascending=False)
                    .groupby("transaction_id", sort=False)
                    .cumcount() + 1
                )
            else:
                # For other columns, just assign the value
                pdf[col_name] = col_expr
        return FakeSparkDataFrame(pdf)

    def filter(self, condition):
        # condition can be:
        #   1. A string SQL expression (e.g., "_row_num = 1") — evaluate it.
        #   2. A pandas Series (boolean mask) — used directly.
        #   3. A callable — apply it to the pandas DataFrame.
        if isinstance(condition, str):
            # Convert SQL-like condition to pandas boolean mask
            # Handle simple equality: "column = value"
            if " = " in condition:
                col_name, value = condition.split(" = ")
                col_name = col_name.strip()
                value = value.strip()
                # Convert value to int if it's numeric
                try:
                    value = int(value)
                except ValueError:
                    pass
                return FakeSparkDataFrame(self._pdf[self._pdf[col_name] == value].copy())
            # For other complex conditions, try pandas query (with == instead of =)
            return FakeSparkDataFrame(self._pdf.query(condition.replace(" = ", " == ")).copy())
        if isinstance(condition, pd.Series):
            return FakeSparkDataFrame(self._pdf[condition.values].copy())
        if callable(condition):
            mask = condition(self._pdf)
            return FakeSparkDataFrame(self._pdf[mask].copy())
        raise TypeError(
            f"Unsupported filter condition type: {type(condition)}."
        )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_spark():
    """A mock SparkSession for tests that don't need real Spark."""
    spark = MagicMock()
    return spark


@pytest.fixture
def mock_logger():
    """A no-op logger that silently absorbs .info() calls."""
    logger = MagicMock(spec=logging.Logger)
    return logger
