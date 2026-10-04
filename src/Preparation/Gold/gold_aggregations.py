"""Gold layer: aggregate silver data and build ML-ready features."""


class DailyDemandAggregator:
    """Aggregates silver data into daily demand by product and city.

    Args:
        spark: Active SparkSession.
        logging: Logger instance for logging messages.
        catalog_name: Unity Catalog catalog name.
        silver_schema_name: Schema name of the silver source table.
        silver_table_name: Name of the silver source table.
        gold_schema_name: Schema name for gold target table.
        aggregated_gold_table_name: Name of the aggregated gold table.
    """

    def __init__(self, spark, logging, catalog_name, silver_schema_name,
                 silver_table_name, gold_schema_name, aggregated_gold_table_name):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.silver_schema_name = silver_schema_name
        self.silver_table_name = silver_table_name
        self.gold_schema_name = gold_schema_name
        self.aggregated_gold_table_name = aggregated_gold_table_name

        # Derive fully qualified table names
        self.source_table = f"{catalog_name}.{silver_schema_name}.{silver_table_name}"
        self.target_table = f"{catalog_name}.{gold_schema_name}.{aggregated_gold_table_name}"

    def write_aggregated_table(self):
        """Create or replace the gold aggregate table using SQL.

        Aggregates silver data into daily demand by product and city.
        """
        self.logger.info(f"Aggregating {self.source_table} -> {self.target_table}")
        self.spark.sql(f"""
            CREATE OR REPLACE TABLE {self.target_table} AS
            SELECT
                txn_date AS transaction_date,
                product_name,
                destination_city,
                ROUND(SUM(CAST(demand_quantity AS DOUBLE)), 2) AS total_demand,
                ROUND(AVG(CAST(unit_price_usd AS DOUBLE)), 2) AS avg_unit_price,
                SUM(CAST(available_inventory AS DOUBLE)) AS total_inventory,
                COUNT(*) AS transaction_count
            FROM {self.source_table}
            GROUP BY txn_date, product_name, destination_city
        """)
        count = self.spark.table(self.target_table).count()
        self.logger.info(f"Gold aggregate table written: {self.target_table} ({count:,} rows)")


class DemandFeatureBuilder:
    """Builds ML-ready features from gold daily aggregates using SQL.

    Args:
        spark: Active SparkSession.
        logging: Logger instance for logging messages.
        catalog_name: Unity Catalog catalog name.
        gold_schema_name: Schema name for gold tables.
        aggregated_gold_table_name: Name of the aggregated gold source table.
        featured_gold_table_name: Name of the featured gold target table.
    """

    def __init__(self, spark, logging, catalog_name, gold_schema_name,
                 aggregated_gold_table_name, featured_gold_table_name):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.gold_schema_name = gold_schema_name
        self.aggregated_gold_table_name = aggregated_gold_table_name
        self.featured_gold_table_name = featured_gold_table_name

        # Derive fully qualified table names
        self.source_table = f"{catalog_name}.{gold_schema_name}.{aggregated_gold_table_name}"
        self.target_table = f"{catalog_name}.{gold_schema_name}.{featured_gold_table_name}"

    def write_feature_table(self):
        """Create or replace the feature table with lags, rolling avg, and calendar fields."""
        self.logger.info(f"Building features: {self.source_table} -> {self.target_table}")
        self.spark.sql(f"""
            CREATE OR REPLACE TABLE {self.target_table} AS
            SELECT * FROM (
                SELECT
                    transaction_date,
                    product_name,
                    destination_city,
                    total_demand,
                    avg_unit_price,
                    total_inventory,
                    dayofweek(transaction_date) AS day_of_week,
                    month(transaction_date) AS month,
                    ROUND(LAG(total_demand, 1) OVER (
                        PARTITION BY product_name, destination_city ORDER BY transaction_date
                    ), 2) AS demand_lag_1,
                    ROUND(LAG(total_demand, 7) OVER (
                        PARTITION BY product_name, destination_city ORDER BY transaction_date
                    ), 2) AS demand_lag_7,
                    ROUND(AVG(total_demand) OVER (
                        PARTITION BY product_name, destination_city
                        ORDER BY transaction_date
                        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
                    ), 2) AS rolling_avg_7
                FROM {self.source_table}
            )
            WHERE demand_lag_7 IS NOT NULL
              AND rolling_avg_7 IS NOT NULL
        """)
        count = self.spark.table(self.target_table).count()
        self.logger.info(f"Feature table written: {self.target_table} ({count:,} rows)")
