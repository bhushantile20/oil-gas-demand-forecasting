"""Drift monitoring: compare training data with prediction data and track model error.

Writes four tables to the monitoring schema:
    feature_drift      - KS statistic, p-value, PSI and mean shift per model feature
    target_drift       - weekly average demand vs the training-period average
    performance_drift  - weekly MAE / RMSE / bias per product and city vs baseline MAE
    drift_summary      - one row per run with the overall status
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyspark.sql import SparkSession
from scipy.stats import ks_2samp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

STATUS_ORDER = ["STABLE", "WARNING", "DRIFT"]


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


# ---------------------------------------------------------------------------
# Pure helper functions (no Spark) - easy to unit test
# ---------------------------------------------------------------------------

def calculate_psi(reference, current, bins=10):
    """Population Stability Index between a reference and a current sample.

    Numeric features are binned on reference quantiles. Features with few distinct
    values (such as day_of_week) are compared by the share of each value.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    # PSI is noisy when bins hold few rows, so use fewer bins for small current samples
    bins = int(max(3, min(bins, len(cur) // 20)))

    if len(np.unique(ref)) <= bins:
        values = np.unique(np.concatenate([ref, cur]))
        ref_pct = np.array([(ref == v).mean() for v in values])
        cur_pct = np.array([(cur == v).mean() for v in values])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0] = -np.inf
        edges[-1] = np.inf
        ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
        cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)

    eps = 1e-4
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def calculate_ks(reference, current):
    """Two-sample Kolmogorov-Smirnov test. Returns (ks_stat, p_value); NaN when a sample is empty."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan"), float("nan")
    result = ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def threshold_status(value, warning, drift):
    """Map a value to STABLE / WARNING / DRIFT. NaN gives UNKNOWN."""
    if value is None or np.isnan(value):
        return "UNKNOWN"
    if value > drift:
        return "DRIFT"
    if value >= warning:
        return "WARNING"
    return "STABLE"


def worst_status(statuses):
    """Return the most severe status in a list (UNKNOWN is ignored)."""
    known = [s for s in statuses if s in STATUS_ORDER]
    if not known:
        return "UNKNOWN"
    return max(known, key=STATUS_ORDER.index)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Calculates feature, target and performance drift and saves the results.

    Args:
        spark: Active SparkSession.
        logging: Logger instance.
        catalog_name: Unity Catalog name.
        config: The drift_monitoring section of config.yml.
    """

    def __init__(self, spark, logging, catalog_name, config):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.config = config

        src = config["source"]
        tgt = config["target"]
        self.training_table = f"{catalog_name}.{src['ml_schema']}.{src['training_table']}"
        self.prediction_table = f"{catalog_name}.{src['ml_schema']}.{src['prediction_table']}"
        self.predictions_table = f"{catalog_name}.{src['ml_schema']}.{src['predictions_table']}"
        self.metrics_table = f"{catalog_name}.{src['ml_schema']}.{src['metrics_table']}"

        self.monitoring_schema = f"{catalog_name}.{tgt['monitoring_schema']}"
        self.feature_drift_table = f"{self.monitoring_schema}.{tgt['feature_drift_table']}"
        self.target_drift_table = f"{self.monitoring_schema}.{tgt['target_drift_table']}"
        self.performance_table = f"{self.monitoring_schema}.{tgt['performance_drift_table']}"
        self.summary_table = f"{self.monitoring_schema}.{tgt['summary_table']}"

        self.features = config["features"]
        self.psi_bins = config["psi_bins"]
        self.ks_alpha = config["thresholds"]["ks_alpha"]
        th = config["thresholds"]
        self.psi_warning = th["psi_warning"]
        self.psi_drift = th["psi_drift"]
        self.mae_warning = th["mae_ratio_warning"]
        self.mae_drift = th["mae_ratio_drift"]

        self.ref_df = None
        self.cur_df = None
        self.run_date = pd.Timestamp.now().date()

        self.feature_drift = None
        self.target_drift = None
        self.performance_drift = None
        self.summary = None

    def load_data(self):
        """Reference = training dataset. Current = the dataset the model makes predictions on."""
        self.logger.info(f"Loading training data from: {self.training_table}")
        self.ref_df = self.spark.table(self.training_table).toPandas()
        self.logger.info(f"Loading prediction data from: {self.prediction_table}")
        self.cur_df = self.spark.table(self.prediction_table).toPandas()
        for df in (self.ref_df, self.cur_df):
            df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        self.logger.info(f"Training rows: {len(self.ref_df):,} | Prediction rows: {len(self.cur_df):,}")

    def compute_feature_drift(self):
        """PSI and mean shift for every monitored feature."""
        rows = []
        for feature in self.features:
            ref_values = self.ref_df[feature]
            cur_values = self.cur_df[feature]
            psi = calculate_psi(ref_values, cur_values, self.psi_bins)
            ks_stat, p_value = calculate_ks(ref_values, cur_values)
            ref_mean = float(ref_values.mean())
            cur_mean = float(cur_values.mean())
            shift = (cur_mean - ref_mean) / ref_mean * 100 if ref_mean != 0 else float("nan")
            rows.append({
                "run_date": self.run_date,
                "feature_name": feature,
                "ks_stat": round(ks_stat, 4) if not np.isnan(ks_stat) else ks_stat,
                "p_value": p_value,
                "ks_drift": bool(p_value < self.ks_alpha) if not np.isnan(p_value) else False,
                "psi": round(psi, 4) if not np.isnan(psi) else psi,
                "ref_mean": round(ref_mean, 2),
                "cur_mean": round(cur_mean, 2),
                "mean_shift_pct": round(shift, 1) if not np.isnan(shift) else shift,
                "drift_status": threshold_status(psi, self.psi_warning, self.psi_drift),
            })
        self.feature_drift = pd.DataFrame(rows)

    def compute_target_drift(self):
        """Weekly average demand in the prediction data vs the training average."""
        ref_avg = float(self.ref_df["total_demand"].mean())
        weekly = self.cur_df.copy()
        weekly["week_start"] = weekly["transaction_date"].dt.to_period("W-SUN").dt.start_time
        weekly = weekly.groupby("week_start", as_index=False)["total_demand"].mean()
        weekly = weekly.rename(columns={"total_demand": "avg_demand"})
        weekly["avg_demand"] = weekly["avg_demand"].round(2)
        weekly["ref_avg_demand"] = round(ref_avg, 2)
        weekly["shift_pct"] = ((weekly["avg_demand"] - ref_avg) / ref_avg * 100).round(1) if ref_avg != 0 else np.nan
        weekly["week_start"] = weekly["week_start"].dt.date
        weekly.insert(0, "run_date", self.run_date)
        self.target_drift = weekly

    def get_baseline_mae(self):
        """Latest model's test MAE from the model_metrics table."""
        metrics = self.spark.table(self.metrics_table).toPandas()
        test = metrics[metrics["split"] == "test"].sort_values("run_ts")
        if test.empty:
            raise ValueError(f"No test metrics found in {self.metrics_table}")
        return float(test.iloc[-1]["mae"])

    def compute_performance_drift(self):
        """Weekly MAE, RMSE and bias per product and city on data the model did not train on."""
        preds = self.spark.table(self.predictions_table).toPandas()
        if "split" not in preds.columns:
            raise ValueError(
                f"{self.predictions_table} has no 'split' column. "
                "Apply the pipeline changes (split / model_version / model_metrics) and re-run the job first."
            )
        preds = preds[preds["split"].isin(["val", "test"])].copy()
        preds["transaction_date"] = pd.to_datetime(preds["transaction_date"])
        preds["error"] = preds["predicted_demand"] - preds["total_demand"]
        preds["abs_error"] = preds["error"].abs()
        preds["sq_error"] = preds["error"] ** 2
        preds["week_start"] = preds["transaction_date"].dt.to_period("W-SUN").dt.start_time

        baseline_mae = self.get_baseline_mae()
        grouped = preds.groupby(["week_start", "product_name", "destination_city"], as_index=False).agg(
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
            bias=("error", "mean"),
            n_rows=("error", "size"),
        )
        grouped["rmse"] = np.sqrt(grouped["mse"])
        grouped["baseline_mae"] = baseline_mae
        grouped["mae_ratio"] = grouped["mae"] / baseline_mae
        grouped["status"] = [threshold_status(r, self.mae_warning, self.mae_drift) for r in grouped["mae_ratio"]]
        grouped["week_start"] = grouped["week_start"].dt.date

        cols = ["week_start", "product_name", "destination_city", "mae", "rmse", "bias",
                "baseline_mae", "mae_ratio", "status", "n_rows"]
        result = grouped[cols].copy()
        for col in ["mae", "rmse", "bias", "baseline_mae", "mae_ratio"]:
            result[col] = result[col].round(2)
        self.performance_drift = result

    def compute_summary(self):
        """One row: worst of feature status and latest-week performance status."""
        fd = self.feature_drift
        worst = fd.loc[fd["psi"].idxmax()] if fd["psi"].notna().any() else None

        perf = self.performance_drift
        latest_week = perf["week_start"].max()
        latest = perf[perf["week_start"] == latest_week]
        latest_mae = float((latest["mae"] * latest["n_rows"]).sum() / latest["n_rows"].sum())
        latest_ratio = latest_mae / float(perf["baseline_mae"].iloc[0])
        perf_status = threshold_status(latest_ratio, self.mae_warning, self.mae_drift)

        feature_status = worst_status(fd["drift_status"].tolist())
        self.summary = pd.DataFrame([{
            "run_date": self.run_date,
            "overall_status": worst_status([feature_status, perf_status]),
            "drifted_feature_count": int((fd["drift_status"] == "DRIFT").sum()),
            "warning_feature_count": int((fd["drift_status"] == "WARNING").sum()),
            "ks_drifted_feature_count": int(fd["ks_drift"].sum()),
            "worst_feature": worst["feature_name"] if worst is not None else None,
            "worst_psi": float(worst["psi"]) if worst is not None else float("nan"),
            "latest_mae_ratio": round(latest_ratio, 2),
            "performance_status": perf_status,
        }])

    def _write(self, df, table, replace_today=False):
        """Overwrite the table, or append after removing today's rows so re-runs do not duplicate."""
        sdf = self.spark.createDataFrame(df)
        if replace_today:
            if self.spark.catalog.tableExists(table):
                self.spark.sql(f"DELETE FROM {table} WHERE run_date = DATE'{self.run_date}'")
            sdf.write.mode("append").option("mergeSchema", "true").saveAsTable(table)
        else:
            sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
        self.logger.info(f"Saved {len(df):,} rows to: {table}")

    def save_results(self):
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {self.monitoring_schema}")
        self._write(self.feature_drift, self.feature_drift_table, replace_today=True)
        self._write(self.summary, self.summary_table, replace_today=True)
        self._write(self.target_drift, self.target_drift_table, replace_today=True)
        self._write(self.performance_drift, self.performance_table)

    def run(self):
        """Run all monitoring steps."""
        self.load_data()
        self.compute_feature_drift()
        self.compute_target_drift()
        self.compute_performance_drift()
        self.compute_summary()
        self.save_results()


def run_drift_monitoring(config, spark, logging):
    dm = config["drift_monitoring"]
    monitor = DriftMonitor(
        spark=spark,
        logging=logging,
        catalog_name=dm["catalog_name"],
        config=dm,
    )
    monitor.run()


# main execution part
if __name__ == "__main__":
    import inspect
    import sys
    spark = SparkSession.builder.getOrCreate()

    # Allow custom config path via command-line argument
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yml", ".yaml")):
        config_path = sys.argv[1]
    else:
        # Same approach as the other scripts: derive the config path from the script location.
        frame = inspect.currentframe()
        assert frame is not None
        script_path = frame.f_code.co_filename
        config_path = str(Path(script_path).resolve().parents[2] / "config" / "config.yml")
    config = load_config(config_path)

    logging.info("Drift Monitoring - started.")
    run_drift_monitoring(config, spark, logger)
    logging.info("Drift Monitoring completed.")
