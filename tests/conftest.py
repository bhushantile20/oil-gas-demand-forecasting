"""
Shared pytest fixtures. A single session-scoped local SparkSession is reused
across every test file that needs one — starting a JVM per test would make
the suite painfully slow.

Only Silver uses this real Spark. Training tests use mock_spark / MagicMock
because GitHub has no Databricks / MLflow.
"""

import logging
from unittest.mock import MagicMock

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark_session = (
        SparkSession.builder
        .master("local[1]")
        .appName("unit-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()


@pytest.fixture
def mock_spark():
    return MagicMock()


@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)