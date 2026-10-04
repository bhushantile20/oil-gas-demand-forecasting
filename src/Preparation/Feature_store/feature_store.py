"""Feature Store layer: register features and save labels."""

# Dependency: databricks-feature-engineering (pip install databricks-feature-engineering "protobuf>=5.29.4")

from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient


class DemandFeatureStore:
    """Registers features in Feature Store and saves labels separately.

    Args:
        spark: Active SparkSession.
        logging: Logger instance for logging messages.
        catalog_name: Unity Catalog catalog name.
        gold_schema_name: Schema name of the gold feature source table.
        featured_gold_table_name: Name of the gold featured source table.
        feature_schema: Schema name for feature store target tables.
        feature_table_name: Name of the feature store table.
        label_table_name: Name of the label table.
    """

    def __init__(self, spark, logging, catalog_name, gold_schema_name,
                 featured_gold_table_name, feature_schema, feature_table_name,
                 label_table_name):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.gold_schema_name = gold_schema_name
        self.featured_gold_table_name = featured_gold_table_name
        self.feature_schema = feature_schema
        self.feature_table_name = feature_table_name
        self.label_table_name = label_table_name

        # Derive fully qualified table names
        self.source_table = f"{catalog_name}.{gold_schema_name}.{featured_gold_table_name}"
        self.feature_table = f"{catalog_name}.{feature_schema}.{feature_table_name}"
        self.label_table = f"{catalog_name}.{feature_schema}.{label_table_name}"

        self.fe = FeatureEngineeringClient()
        self.data = None

    def read_gold_table(self):
        """Step 1: Load the gold feature table."""
        self.logger.info(f"Loading: {self.source_table}")
        self.data = self.spark.table(self.source_table)
        self.logger.info(f"Loaded {self.data.count():,} rows")

    def add_primary_key(self):
        """Step 2: Add feature_uid as primary key."""
        self.logger.info("Adding primary key: feature_uid")
        self.data = self.data.withColumns({
            "feature_uid": F.concat_ws(
                "_",
                F.col("transaction_date").cast("string"),
                F.col("product_name"),
                F.col("destination_city"),
            ),
        })
        self.logger.info("Primary key added: feature_uid")

    def register_feature_table(self):
        """Step 3: Register features (no label) in Feature Store."""
        feature_data = self.data.select(
            "feature_uid", "transaction_date", "product_name", "destination_city",
            "avg_unit_price", "total_inventory",
            "day_of_week", "month",
            "demand_lag_1", "demand_lag_7", "rolling_avg_7",
        )

        if not self.spark.catalog.tableExists(self.feature_table):
            self.logger.info(f"Creating feature table: {self.feature_table}")
            self.fe.create_table(
                name=self.feature_table,
                primary_keys=["feature_uid"],
                df=feature_data,
                description="Features for demand forecasting. Label stored separately.",
                tags={
                    "name": "daily_demand_features",
                    "domain": "oil&gas",
                    "key_columns": "feature_uid, transaction_date, product_name, destination_city",
                    "categorical_columns": "product_name, destination_city",
                    "numeric_columns": "total_inventory, day_of_week, month, demand_lag_1, demand_lag_7, rolling_avg_7",
                },
            )
            self.logger.info("Feature table created.")
        else:
            self.logger.info(f"Feature table already exists: {self.feature_table}")

        self.logger.info(f"Writing features via merge: {self.feature_table}")
        self.fe.write_table(name=self.feature_table, df=feature_data, mode="merge")
        self.logger.info(f"Features written ({feature_data.count():,} rows)")

    def save_label_data(self):
        """Step 4: Save total_demand as the label."""
        label_data = self.data.select("feature_uid", "total_demand")

        self.logger.info(f"Writing labels to: {self.label_table}")
        label_data.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(self.label_table)
        self.logger.info(f"Labels written ({label_data.count():,} rows)")
