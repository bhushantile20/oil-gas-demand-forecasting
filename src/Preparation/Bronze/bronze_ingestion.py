"""Bronze layer: load raw CSV data into a Delta bronze table with audit columns."""

from pyspark.sql import functions as F


class BronzeLoader:
    """Loads raw CSV data into the bronze layer with audit columns.

    Args:
        spark: Active SparkSession.
        logging: Logger instance for logging messages.
        catalog_name: Unity Catalog catalog name.
        bronze_schema_name: Schema name for the bronze target table.
        raw_schema_name: Schema name where the source volume lives.
        volume_name: Volume name containing the raw CSV file.
        bronze_table_name: Target bronze table name.
        file_name: Name of the source CSV file.
    """

    def __init__(self, spark, logging, catalog_name, bronze_schema_name,
                 raw_schema_name, volume_name, bronze_table_name, file_name):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.bronze_schema_name = bronze_schema_name
        self.raw_schema_name = raw_schema_name
        self.volume_name = volume_name
        self.bronze_table_name = bronze_table_name
        self.file_name = file_name

        # Derive fully qualified paths
        self.file_path = f"/Volumes/{catalog_name}/{raw_schema_name}/{volume_name}/{file_name}"
        self.table_name = f"{catalog_name}.{bronze_schema_name}.{bronze_table_name}"

        self.df = None

    def read_csv(self):
        """Read CSV file with all columns as strings (Spark default)."""
        self.logger.info(f"Reading CSV from: {self.file_path}")
        self.df = (
            self.spark.read
            .option("header", True)
            .option("multiLine", True)
            .csv(self.file_path)
        )
        self.logger.info(f"Loaded {self.df.count()} rows from CSV")

    def add_txn_date(self):
        """Parse transaction_date into txn_date (M/d/yyyy with ISO fallback)."""
        self.logger.info("Parsing transaction_date -> txn_date")
        self.df = self.df.withColumns({
            "txn_date": F.coalesce(
                F.expr("try_to_date(transaction_date, 'M/d/yyyy')"),
                F.expr("try_to_date(transaction_date, 'yyyy-MM-dd')"),
            ),
        })


    def add_audit_columns(self):
        """Add ingestion timestamp and source file path."""
        self.logger.info("Adding audit columns: _ingest_ts, _source_file")
        self.df = self.df.withColumns({
            "_ingest_ts": F.current_timestamp(),
            "_source_file": F.lit(self.file_name),
        })


    def write_to_bronze(self):
        """Write the DataFrame to the bronze Delta table."""
        self.logger.info(f"Writing to bronze table: {self.table_name}")
        (
            self.df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(self.table_name)
        )
        self.logger.info(f"Bronze table written successfully: {self.table_name}")
