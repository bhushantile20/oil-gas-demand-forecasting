import logging
from pathlib import Path

import yaml
from pyspark.sql import SparkSession

from Bronze.bronze_ingestion import BronzeLoader
from Feature_store.feature_store import DemandFeatureStore
from Gold.gold_aggregations import DailyDemandAggregator, DemandFeatureBuilder
from Silver.silver_transformation import SilverTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def run_bronze(config, spark, logging):
    catalog_name = config["data_preparation"]["catalog_name"]
    raw_schema_name = config["data_preparation"]["bronze"]["source"]["schema_name"]
    volume_name = config["data_preparation"]["bronze"]["source"]["volume_name"]
    file_name = config["data_preparation"]["bronze"]["source"]["file_name"]
    bronze_schema_name = config["data_preparation"]["bronze"]["target"]["schema_name"]
    bronze_table_name = config["data_preparation"]["bronze"]["target"]["table_name"]

    bronze_obj = BronzeLoader(
        spark=spark,
        logging=logging,
        catalog_name=catalog_name,
        bronze_schema_name=bronze_schema_name,
        raw_schema_name=raw_schema_name,
        volume_name=volume_name,
        bronze_table_name=bronze_table_name,
        file_name=file_name,
    )

    bronze_obj.read_csv()
    bronze_obj.add_txn_date()
    bronze_obj.add_audit_columns()
    bronze_obj.write_to_bronze()

def run_silver(config, spark, logging):
    catalog_name = config["data_preparation"]["catalog_name"]
    bronze_schema_name = config["data_preparation"]["silver"]["source"]["bronze_schema_name"]
    bronze_table_name = config["data_preparation"]["silver"]["source"]["bronze_table_name"]
    silver_schema_name = config["data_preparation"]["silver"]["target"]["silver_schema_name"]
    silver_table_name = config["data_preparation"]["silver"]["target"]["silver_table_name"]

    silver_obj = SilverTransformer(
        spark=spark,
        logging=logging,
        catalog_name=catalog_name,
        bronze_schema_name=bronze_schema_name,
        bronze_table_name=bronze_table_name,
        silver_schema_name=silver_schema_name,
        silver_table_name=silver_table_name,
    )

    silver_obj.load_bronze()
    silver_obj.remove_duplicates()
    silver_obj.write_to_silver()

def run_gold(config, spark, logging):
    catalog_name = config["data_preparation"]["catalog_name"]
    silver_schema_name = config["data_preparation"]["gold"]["source"]["silver_schema_name"]
    silver_table_name = config["data_preparation"]["gold"]["source"]["silver_table_name"]
    gold_schema_name = config["data_preparation"]["gold"]["target"]["gold_schema_name"]
    aggregated_gold_table_name = config["data_preparation"]["gold"]["target"]["aggregated_gold_table_name"]
    featured_gold_table_name = config["data_preparation"]["gold"]["target"]["featured_gold_table_name"]

    aggregator = DailyDemandAggregator(
        spark=spark,
        logging=logging,
        catalog_name=catalog_name,
        silver_schema_name=silver_schema_name,
        silver_table_name=silver_table_name,
        gold_schema_name=gold_schema_name,
        aggregated_gold_table_name=aggregated_gold_table_name,
    )

    aggregator.write_aggregated_table()

    builder = DemandFeatureBuilder(
        spark=spark,
        logging=logging,
        catalog_name=catalog_name,
        gold_schema_name=gold_schema_name,
        aggregated_gold_table_name=aggregated_gold_table_name,
        featured_gold_table_name=featured_gold_table_name,
    )

    builder.write_feature_table()

def run_feature_store(config, spark, logging):
    catalog_name = config["data_preparation"]["catalog_name"]
    gold_schema_name = config["data_preparation"]["feature_store"]["source"]["gold_schema_name"]
    featured_gold_table_name = config["data_preparation"]["feature_store"]["source"]["featured_gold_table_name"]
    feature_schema = config["data_preparation"]["feature_store"]["target"]["feature_schema"]
    feature_table_name = config["data_preparation"]["feature_store"]["target"]["feature_table_name"]
    label_table_name = config["data_preparation"]["feature_store"]["target"]["label_table_name"]

    feature_store_obj = DemandFeatureStore(
        spark=spark,
        logging=logging,
        catalog_name=catalog_name,
        gold_schema_name=gold_schema_name,
        featured_gold_table_name=featured_gold_table_name,
        feature_schema=feature_schema,
        feature_table_name=feature_table_name,
        label_table_name=label_table_name,
    )

    feature_store_obj.read_gold_table()
    feature_store_obj.add_primary_key()
    feature_store_obj.register_feature_table()
    feature_store_obj.save_label_data()

if __name__ == "__main__":
    import inspect
    import sys
    spark = SparkSession.builder.getOrCreate()

    # Allow custom config path via command-line argument
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yml", ".yaml")):
        config_path = sys.argv[1]
    else:
        # Derive config path from script location using inspect.
        # inspect.currentframe().f_code.co_filename works in both normal Python
        # and exec() context (Databricks serverless spark_python_task) where
        # __file__ is not defined.
        frame = inspect.currentframe()
        assert frame is not None
        script_path = frame.f_code.co_filename
        config_path = str(Path(script_path).resolve().parents[2] / "config" / "config.yml")
    config = load_config(config_path)

    logging.info("Bronze - raw data ingestion started. 🔃")
    run_bronze(config, spark, logger)
    logging.info("Bronze layer completed. ✅")

    logging.info("Silver - transformation started on the bronze raw data. 🔃")
    run_silver(config, spark, logger)
    logging.info("Silver layer completed. ✅")

    logging.info("Gold - aggregations & features engineering started on the transformed data. 🔃")
    run_gold(config, spark, logger)
    logging.info("Gold layer completed. ✅")

    logging.info("Feature Store - started registering the feature data in the feature store. 🔃")
    run_feature_store(config, spark, logger)
    logging.info("Feature Store layer completed. ✅")
