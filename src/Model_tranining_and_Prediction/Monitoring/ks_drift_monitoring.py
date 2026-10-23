
from datetime import datetime, timezone

import numpy as np
from scipy.stats import ks_2samp
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


REFERENCE_TABLE = "oil_gas_demo.ml_model.training_data"
CURRENT_TABLE = "oil_gas_demo.ml_model.testing_data"

FEATURES = [
    "avg_unit_price",
    "total_inventory",
    "demand_lag_1",
    "demand_lag_7",
    "rolling_avg_7",
    "day_of_week",
]

P_VALUE_THRESHOLD = 0.05

def calculate_ks_drift(spark: SparkSession):
    # 1. Read reference and current datasets
    reference_df = spark.table(REFERENCE_TABLE)
    current_df = spark.table(CURRENT_TABLE)

    # 2. Collect only the required features
    reference_pd = reference_df.select(*FEATURES).toPandas()
    current_pd = current_df.select(*FEATURES).toPandas()

    results = []

    # 3. Compare each feature
    for feature in FEATURES:
        ref_values = reference_pd[feature].to_numpy(dtype=float)
        cur_values = current_pd[feature].to_numpy(dtype=float)

        # Remove null, NaN and infinite values
        ref_values = ref_values[np.isfinite(ref_values)]
        cur_values = cur_values[np.isfinite(cur_values)]

        if len(ref_values) == 0 or len(cur_values) == 0:
            raise ValueError(
                f"No valid values found for feature: {feature}"
            )

        # 4. Run the two-sample KS test
        ks_result = ks_2samp(
            ref_values,
            cur_values,
            alternative="two-sided",
            method="auto",
        )

        p_value = float(ks_result.pvalue)

        status = (
            "DRIFT"
            if p_value < P_VALUE_THRESHOLD
            else "STABLE"
        )

        results.append({
            "feature_name": feature,
            "reference_count": len(ref_values),
            "current_count": len(cur_values),
            "ks_stat": float(ks_result.statistic),
            "p_value": p_value,
            "drift_status": status,
        })

    # 5. Convert the results into a Spark DataFrame
    results_df = spark.createDataFrame(results)

    
    # Save results in the monitoring Delta table
    monitoring_schema = "oil_gas_demo.monitoring"
    result_table = f"{monitoring_schema}.ks_drift_results"

    # Create the schema if it doesn't exist
    spark.sql(
        f"CREATE SCHEMA IF NOT EXISTS {monitoring_schema}"
    )

    # Add a run timestamp and source-table columns so we can track each monitoring run
    run_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    results_df = results_df.withColumns({
        "run_timestamp": F.lit(run_timestamp).cast("timestamp"),
        "reference_table": F.lit(REFERENCE_TABLE),
        "current_table": F.lit(CURRENT_TABLE),
    })

    # First run creates the table; later runs append new results
    if not spark.catalog.tableExists(result_table):
        (
            results_df.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(result_table)
        )
    else:
        (
            results_df.write
            .format("delta")
            .mode("append")
            .saveAsTable(result_table)
        )

    print(f"KS drift results saved to {result_table}")

    return results_df



if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()

    results_df = calculate_ks_drift(spark)
    results_df.orderBy("feature_name").show(truncate=False)
