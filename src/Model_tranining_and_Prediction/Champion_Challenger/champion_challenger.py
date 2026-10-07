"""Champion-Challenger Evaluation: compare models and promote if better."""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ChampionChallengerEvaluator:
    """Evaluates champion vs challenger models on validation data and promotes if better.

    Args:
        spark: Active SparkSession.
        logging: Logger instance for logging messages.
        model_name: Fully qualified registered model name.
        champion_alias: Alias for the current champion model.
        challenger_alias: Alias for the challenger model.
        validation_table: Fully qualified validation data table name.
    """

    def __init__(self, spark, logging, catalog_name, config):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.config = config

        # Read from config dict
        ml_schema = config["source"]["ml_schema"]
        validation_table_name = config["source"]["validation_table"]
        model_name = config["model"]["model_name"]
        self.champion_alias = config["model"]["champion_alias"]
        self.challenger_alias = config["model"]["challenger_alias"]

        # Derive fully qualified names
        self.model_name = f"{catalog_name}.{ml_schema}.{model_name}"
        self.validation_table = f"{catalog_name}.{ml_schema}.{validation_table_name}"

        self.client = MlflowClient()

        self.champion_version_number = None
        self.champion_run_id = None
        self.challenger_version_number = None
        self.challenger_run_id = None
        self.champion_encoder = None
        self.challenger_encoder = None
        self.validation_df = None
        self.champion_predictions = None
        self.challenger_predictions = None
        self.comparison_df = None

    def evaluate(self):
        """Run all evaluation steps: retrieve, load, predict, compare, promote."""
        self.retrieve_model_versions()
        self.load_encoders()
        self.load_validation_data()
        self.generate_predictions()
        self.compare_performance()
        self.promote_challenger()

    def retrieve_model_versions(self):
        """Step 1: Retrieve champion and challenger model versions by alias."""
        self.logger.info("Retrieving model versions")
        champion_version = self.client.get_model_version_by_alias(self.model_name, self.champion_alias)
        challenger_version = self.client.get_model_version_by_alias(self.model_name, self.challenger_alias)

        self.champion_version_number = champion_version.version
        self.champion_run_id = champion_version.run_id
        self.challenger_version_number = challenger_version.version
        self.challenger_run_id = challenger_version.run_id

        self.logger.info(f"Champion: version {self.champion_version_number} | run {self.champion_run_id}")
        self.logger.info(f"Challenger: version {self.challenger_version_number} | run {self.challenger_run_id}")

    def load_encoders(self):
        """Step 2: Download encoders for both models from MLflow."""
        self.logger.info("Loading model-specific encoders")
        champion_encoder_path = mlflow.artifacts.download_artifacts(
            run_id=self.champion_run_id,
            artifact_path="encoder/encoder.joblib",
        )
        self.champion_encoder = joblib.load(champion_encoder_path)
        self.logger.info("Champion encoder loaded")

        challenger_encoder_path = mlflow.artifacts.download_artifacts(
            run_id=self.challenger_run_id,
            artifact_path="encoder/encoder.joblib",
        )
        self.challenger_encoder = joblib.load(challenger_encoder_path)
        self.logger.info("Challenger encoder loaded")

    def load_validation_data(self):
        """Step 3: Load validation dataset."""
        self.logger.info(f"Loading validation data from: {self.validation_table}")
        self.validation_df = self.spark.table(self.validation_table)
        self.logger.info(f"Validation rows: {self.validation_df.count():,}")

    def generate_predictions(self):
        """Step 4: Prepare features and predict with both models."""
        self.logger.info("Generating predictions for champion and challenger")
        val_pd = self.validation_df.toPandas()

        # Handle missing values in time-series feature columns
        for col_name in ["demand_lag_1", "demand_lag_7", "rolling_avg_7"]:
            if col_name in val_pd.columns:
                val_pd = val_pd.fillna({col_name: 0.0})

        cat_cols = ["product_name", "destination_city"]
        drop_cols = ["feature_uid", "transaction_date", "total_demand", "product_name", "destination_city"]
        drop_present = [c for c in drop_cols if c in val_pd.columns]

        # Champion features
        champion_encoded = self.champion_encoder.transform(val_pd[cat_cols])
        champion_encoded_cols = self.champion_encoder.get_feature_names_out(cat_cols).tolist()
        champion_encoded_df = pd.DataFrame(champion_encoded, columns=champion_encoded_cols, index=val_pd.index)
        champion_features = pd.concat([val_pd.drop(columns=drop_present), champion_encoded_df], axis=1)

        # Challenger features
        challenger_encoded = self.challenger_encoder.transform(val_pd[cat_cols])
        challenger_encoded_cols = self.challenger_encoder.get_feature_names_out(cat_cols).tolist()
        challenger_encoded_df = pd.DataFrame(challenger_encoded, columns=challenger_encoded_cols, index=val_pd.index)
        challenger_features = pd.concat([val_pd.drop(columns=drop_present), challenger_encoded_df], axis=1)

        # Load models and predict
        champion_model = mlflow.sklearn.load_model(f"runs:/{self.champion_run_id}/random_forest_demand_model")
        challenger_model = mlflow.sklearn.load_model(f"runs:/{self.challenger_run_id}/random_forest_demand_model")

        self.champion_predictions = champion_model.predict(champion_features)
        self.challenger_predictions = challenger_model.predict(challenger_features)

        self.logger.info(f"Champion predictions: {len(self.champion_predictions):,}")
        self.logger.info(f"Challenger predictions: {len(self.challenger_predictions):,}")

    def compare_performance(self):
        """Step 5: Compare MAE, RMSE, R2 for both models."""
        self.logger.info("Comparing model performance")
        y_true = self.validation_df.select("total_demand").toPandas()["total_demand"]

        def evaluate(name, predictions):
            return {
                "Model": name,
                "MAE": mean_absolute_error(y_true, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
                "R2": r2_score(y_true, predictions),
            }

        self.comparison_df = pd.DataFrame([
            evaluate("Champion", self.champion_predictions),
            evaluate("Challenger", self.challenger_predictions),
        ])

        for _, row in self.comparison_df.iterrows():
            self.logger.info(f"{row['Model']}: MAE={row['MAE']:.2f} | RMSE={row['RMSE']:.2f} | R2={row['R2']:.4f}")

    def promote_challenger(self):
        """Step 6: Promote challenger if it has lower MAE than champion."""
        champion_mae = self.comparison_df.loc[self.comparison_df["Model"] == "Champion", "MAE"].values[0]
        challenger_mae = self.comparison_df.loc[self.comparison_df["Model"] == "Challenger", "MAE"].values[0]

        decision = "PROMOTE" if challenger_mae < champion_mae else "KEEP_CHAMPION"

        self.logger.info(f"Champion MAE: {champion_mae:.2f} | Challenger MAE: {challenger_mae:.2f}")
        self.logger.info(f"Decision: {decision}")

        if decision == "PROMOTE":
            old_champion = self.client.get_model_version_by_alias(self.model_name, "champion")
            self.client.set_registered_model_alias(self.model_name, "previous-champion", old_champion.version)
            self.client.set_registered_model_alias(self.model_name, "champion", self.challenger_version_number)
            self.logger.info(f"Version {self.challenger_version_number} promoted to Champion!")
        else:
            self.logger.info(f"Challenger not promoted. Champion remains version {self.champion_version_number}.")
