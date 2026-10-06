"""Model Training: train RandomForestRegressor on feature store data."""

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature


class DemandForecastTraining:
    """Trains a RandomForestRegressor on the feature store data.
    """

    def __init__(self, spark, logging, catalog_name, config):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.config = config

        # Read from config dict
        feature_schema = config["source"]["feature_schema"]
        feature_table_name = config["source"]["feature_table"]
        label_schema = config["source"]["label_schema"]
        label_table_name = config["source"]["label_table"]
        ml_schema = config["target"]["ml_schema"]
        train_table_name = config["target"]["train_table"]
        val_table_name = config["target"]["val_table"]
        test_table_name = config["target"]["test_table"]
        predictions_table_name = config["target"]["predictions_table"]
        model_name = config["model"]["model_name"]
        self.experiment_name = config["model"]["experiment_name"]
        self.ts_feature_cols = config["features"]["ts_feature_cols"]
        self.cat_cols = config["features"]["cat_cols"]
        self.drop_cols = config["features"]["drop_cols"]
        self.split_train = config["split"]["train"]
        self.split_val = config["split"]["val"]
        self.n_estimators = config["rf"]["n_estimators"]
        self.random_state = config["rf"]["random_state"]
        self.n_jobs = config["rf"]["n_jobs"]

        # Derive fully qualified names
        self.feature_table = f"{catalog_name}.{feature_schema}.{feature_table_name}"
        self.label_table = f"{catalog_name}.{label_schema}.{label_table_name}"
        self.train_table = f"{catalog_name}.{ml_schema}.{train_table_name}"
        self.val_table = f"{catalog_name}.{ml_schema}.{val_table_name}"
        self.test_table = f"{catalog_name}.{ml_schema}.{test_table_name}"
        self.predictions_table = f"{catalog_name}.{ml_schema}.{predictions_table_name}"
        self.model_name = f"{catalog_name}.{ml_schema}.{model_name}"

        self.features_df = None
        self.labels_df = None
        self.joined = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.train_pd = None
        self.val_pd = None
        self.test_pd = None
        self.encoder = None
        self.encoded_cols = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.val_metrics = {}
        self.test_metrics = {}
        self.run_id = None
        self.model_info = None
        self.registered_version = None

    def load_data(self):
        """Load feature and label tables."""
        self.logger.info(f"Loading features from: {self.feature_table}")
        self.features_df = self.spark.table(self.feature_table)
        self.logger.info(f"Loaded {self.features_df.count():,} feature rows")

        self.logger.info(f"Loading labels from: {self.label_table}")
        self.labels_df = self.spark.table(self.label_table)
        self.logger.info(f"Loaded {self.labels_df.count():,} label rows")

    def start_training(self):
        """Run all training steps: join, clean, split, encode, train, evaluate, log."""
        self.join_data()
        self.handle_missing_values()
        self.chronological_split()
        self.prepare_features()
        self.train_model()
        self.evaluate_model()
        self.log_to_mlflow()
        self.create_prediction_table()

    def join_data(self):
        """Join features + labels on feature_uid."""
        self.logger.info("Joining features + labels on feature_uid")
        labels_renamed = self.labels_df.withColumnRenamed("feature_uid", "label_feature_uid")
        self.joined = (
            self.features_df.join(labels_renamed, self.features_df.feature_uid == labels_renamed.label_feature_uid, "inner")
            .drop("label_feature_uid")
        )
        self.logger.info(f"Joined: {self.joined.count():,} rows")

    def handle_missing_values(self):
        """Fill nulls in time-series feature columns with 0.0."""
        self.logger.info("Handling missing values in time-series feature columns")
        for col_name in self.ts_feature_cols:
            if col_name in self.joined.columns:
                self.joined = self.joined.fillna({col_name: 0.0})
        self.logger.info("Missing values handled")

    def chronological_split(self):
        """Split data chronologically into train/val/test sets."""
        self.logger.info("Performing chronological split")
        dates = [row.transaction_date for row in self.joined.select("transaction_date").distinct().orderBy("transaction_date").collect()]

        n_dates = len(dates)
        train_end = int(n_dates * self.split_train)
        val_end = int(n_dates * (self.split_train + self.split_val))

        train_dates = set(dates[:train_end])
        val_dates = set(dates[train_end:val_end])
        test_dates = set(dates[val_end:])

        self.train_df = self.joined.filter(F.col("transaction_date").isin(list(train_dates)))
        self.val_df = self.joined.filter(F.col("transaction_date").isin(list(val_dates)))
        self.test_df = self.joined.filter(F.col("transaction_date").isin(list(test_dates)))

        self.logger.info(f"Train: {self.train_df.count():,} | Val: {self.val_df.count():,} | Test: {self.test_df.count():,}")

    def prepare_features(self):
        """Convert to pandas, OneHotEncode categoricals, prepare X/y."""
        self.logger.info("Converting to pandas and encoding features")
        self.train_pd = self.train_df.toPandas()
        self.val_pd = self.val_df.toPandas()
        self.test_pd = self.test_df.toPandas()

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(self.train_pd[self.cat_cols])

        train_encoded = self.encoder.transform(self.train_pd[self.cat_cols])
        val_encoded = self.encoder.transform(self.val_pd[self.cat_cols])
        test_encoded = self.encoder.transform(self.test_pd[self.cat_cols])

        self.encoded_cols = self.encoder.get_feature_names_out(self.cat_cols).tolist()

        train_encoded_df = pd.DataFrame(train_encoded, columns=self.encoded_cols, index=self.train_pd.index)
        val_encoded_df = pd.DataFrame(val_encoded, columns=self.encoded_cols, index=self.val_pd.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=self.encoded_cols, index=self.test_pd.index)

        drop_cols_present = [c for c in self.drop_cols if c in self.train_pd.columns]

        self.X_train = pd.concat([self.train_pd.drop(columns=drop_cols_present), train_encoded_df], axis=1)
        self.X_val = pd.concat([self.val_pd.drop(columns=drop_cols_present), val_encoded_df], axis=1)
        self.X_test = pd.concat([self.test_pd.drop(columns=drop_cols_present), test_encoded_df], axis=1)

        self.y_train = self.train_pd["total_demand"]
        self.y_val = self.val_pd["total_demand"]
        self.y_test = self.test_pd["total_demand"]

        self.logger.info(f"X_train: {self.X_train.shape} | X_val: {self.X_val.shape} | X_test: {self.X_test.shape}")

    def train_model(self):
        """Train RandomForestRegressor."""
        self.logger.info(f"Training RandomForestRegressor (n_estimators={self.n_estimators})")
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(self.X_train, self.y_train)
        self.logger.info("Model trained")

    def evaluate_model(self):
        """Evaluate on validation and test sets."""
        self.logger.info("Evaluating model")
        val_preds = self.model.predict(self.X_val)
        test_preds = self.model.predict(self.X_test)

        self.val_metrics = {
            "mae": mean_absolute_error(self.y_val, val_preds),
            "mse": mean_squared_error(self.y_val, val_preds),
            "rmse": mean_squared_error(self.y_val, val_preds) ** 0.5,
            "r2": r2_score(self.y_val, val_preds),
        }
        self.test_metrics = {
            "mae": mean_absolute_error(self.y_test, test_preds),
            "mse": mean_squared_error(self.y_test, test_preds),
            "rmse": mean_squared_error(self.y_test, test_preds) ** 0.5,
            "r2": r2_score(self.y_test, test_preds),
        }

        self.logger.info(f"Val  - MAE: {self.val_metrics['mae']:.2f} | RMSE: {self.val_metrics['rmse']:.2f} | R2: {self.val_metrics['r2']:.4f}")
        self.logger.info(f"Test - MAE: {self.test_metrics['mae']:.2f} | RMSE: {self.test_metrics['rmse']:.2f} | R2: {self.test_metrics['r2']:.4f}")

    def log_to_mlflow(self):
        """Log params, metrics, encoder artifact, and model to MLflow."""
        self.logger.info(f"Logging to MLflow experiment: {self.experiment_name}")
        mlflow.set_experiment(self.experiment_name)

        signature = infer_signature(self.X_train.head(100), self.model.predict(self.X_train.head(100)))

        with mlflow.start_run() as run:
            self.run_id = run.info.run_id

            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("n_jobs", self.n_jobs)
            mlflow.log_param("train_rows", len(self.train_pd))
            mlflow.log_param("val_rows", len(self.val_pd))
            mlflow.log_param("test_rows", len(self.test_pd))

            mlflow.log_metric("val_mae", self.val_metrics["mae"])
            mlflow.log_metric("val_mse", self.val_metrics["mse"])
            mlflow.log_metric("val_rmse", self.val_metrics["rmse"])
            mlflow.log_metric("val_r2", self.val_metrics["r2"])
            mlflow.log_metric("test_mae", self.test_metrics["mae"])
            mlflow.log_metric("test_mse", self.test_metrics["mse"])
            mlflow.log_metric("test_rmse", self.test_metrics["rmse"])
            mlflow.log_metric("test_r2", self.test_metrics["r2"])

            joblib.dump(self.encoder, "/tmp/encoder.joblib")
            mlflow.log_artifact("/tmp/encoder.joblib", artifact_path="encoder")

            self.model_info = mlflow.sklearn.log_model(
                self.model,
                name="random_forest_demand_model",
                signature=signature,
                input_example=self.X_train.head(3),
            )

        self.logger.info(f"MLflow run: {self.run_id}")

    def register_model(self):
        """Register model in Unity Catalog."""
        self.logger.info(f"Registering model: {self.model_name}")
        result = mlflow.register_model(
            model_uri=self.model_info.model_uri,
            name=self.model_name,
        )
        self.registered_version = result.version
        self.logger.info(f"Registered: {self.model_name} version {self.registered_version}")

    def create_prediction_table(self):
        """Create final prediction table with all data predictions."""
        self.logger.info(f"Creating prediction table: {self.predictions_table}")
        all_pd = pd.concat([self.train_pd, self.val_pd, self.test_pd], ignore_index=True)

        all_encoded = self.encoder.transform(all_pd[self.cat_cols])
        all_encoded_df = pd.DataFrame(all_encoded, columns=self.encoded_cols, index=all_pd.index)
        all_drop = [c for c in self.drop_cols if c in all_pd.columns]
        X_all = pd.concat([all_pd.drop(columns=all_drop), all_encoded_df], axis=1)

        all_pd["predicted_demand"] = self.model.predict(X_all)

        pred_cols = ["feature_uid", "transaction_date", "product_name", "destination_city", "total_demand", "predicted_demand"]
        predictions_pd = all_pd[[c for c in pred_cols if c in all_pd.columns]]

        self.spark.createDataFrame(predictions_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(self.predictions_table)
        self.logger.info(f"Predictions saved to: {self.predictions_table} ({len(predictions_pd):,} rows)")
















        # Args:
        # spark: Active SparkSession.
        # logging: Logger instance for logging messages.
        # feature_table: Fully qualified feature store table name.
        # label_table: Fully qualified label table name.
        # train_table: Fully qualified training data table name.
        # val_table: Fully qualified validation data table name.
        # test_table: Fully qualified testing data table name.
        # predictions_table: Fully qualified predictions table name.
        # model_name: Fully qualified registered model name.
        # experiment_name: MLflow experiment name.
        # ts_feature_cols: List of time-series feature columns that may have nulls.
        # cat_cols: List of categorical columns to OneHotEncode.
        # drop_cols: List of columns to drop from model features.
        # split_train: Train split ratio.
        # split_val: Validation split ratio.
        # n_estimators: Number of trees in the forest.
        # random_state: Random seed.
        # n_jobs: Number of parallel jobs.
