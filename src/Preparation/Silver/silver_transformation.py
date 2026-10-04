"""Silver layer: load from bronze, remove duplicates, and write to silver."""

from pyspark.sql import functions as F
from pyspark.sql.window import Window


class SilverTransformer:
    """Transforms bronze data into silver by removing duplicates.

    (Renamed from Silver to match original notebook class name.)

    Args:
        spark: Active SparkSession.
        logging: Logger instance for logging messages.
        catalog_name: Unity Catalog catalog name.
        bronze_schema_name: Schema name of the bronze source table.
        bronze_table_name: Name of the bronze source table.
        silver_schema_name: Schema name for the silver target table.
        silver_table_name: Name of the silver target table.
    """

    def __init__(self, spark, logging, catalog_name, bronze_schema_name,
                 bronze_table_name, silver_schema_name, silver_table_name):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.bronze_schema_name = bronze_schema_name
        self.bronze_table_name = bronze_table_name
        self.silver_schema_name = silver_schema_name
        self.silver_table_name = silver_table_name

        # Derive fully qualified table names
        self.source_table = f"{catalog_name}.{bronze_schema_name}.{bronze_table_name}"
        self.target_table = f"{catalog_name}.{silver_schema_name}.{silver_table_name}"

        self.df = None

    def load_bronze(self):
        """Read the bronze table."""
        self.logger.info(f"Loading bronze: {self.source_table}")
        self.df = self.spark.table(self.source_table)
        self.logger.info(f"Loaded {self.df.count()} rows")

    def remove_duplicates(self):
        """Keep the latest record per transaction_id (by _ingest_ts desc)."""
        self.logger.info("Removing duplicates by transaction_id")
        window = Window.partitionBy("transaction_id").orderBy(F.col("_ingest_ts").desc())
        self.df = (
            self.df
            .withColumns({"_row_num": F.row_number().over(window)})
            .filter("_row_num = 1")
            .drop("_row_num")
        )

    def write_to_silver(self):
        """Write the deduplicated DataFrame to the silver Delta table."""
        count = self.df.count()
        self.logger.info(f"Writing {count:,} rows to silver: {self.target_table}")
        (
            self.df.write
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.target_table)
        )
        self.logger.info(f"Silver table written: {self.target_table} ({count:,} rows)")
