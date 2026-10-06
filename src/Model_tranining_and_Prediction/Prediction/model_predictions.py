"""Model Predictions & Evaluation: load test data, predict, evaluate, save results."""

import mlflow
import joblib
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelPredictions:
    """Loads test data, encodes categoricals, predicts demand, and evaluates.

    Args:
        spark: Active SparkSession.
        logging: Logger instance for logging messages.
        test_table: Fully qualified testing data table name.
        model_name: Fully qualified registered model name.
        output_table: Fully qualified output predictions table name.
    """

    def __init__(self, spark, logging, catalog_name, config):
        self.spark = spark
        self.logger = logging
        self.catalog_name = catalog_name
        self.config = config

        # Read from config dict
        ml_schema = config["source"]["ml_schema"]
        test_table_name = config["source"]["test_table"]
        output_table_name = config["target"]["output_table"]
        model_name = config["model"]["model_name"]

        # Derive fully qualified names
        self.test_table = f"{catalog_name}.{ml_schema}.{test_table_name}"
        self.model_name = f"{catalog_name}.{ml_schema}.{model_name}"
        self.output_table = f"{catalog_name}.{ml_schema}.{output_table_name}"

        self.client = MlflowClient()
        self.data = None
        self.data_pd = None
        self.encoder = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.latest_version = None

    def load_data(self):
        """Step 1: Load the saved testing dataset."""
        self.logger.info(f"Loading test data from: {self.test_table}")
        self.data = self.spark.table(self.test_table)
        self.logger.info(f"Loaded {self.data.count():,} rows")

    def load_encoder(self):
        """Step 2: Load the OneHotEncoder saved during model training."""
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        self.latest_version = latest.version
        run_id = latest.run_id

        self.logger.info(f"Downloading encoder from run: {run_id}")
        encoder_path = self.client.download_artifacts(run_id=run_id, path="encoder/encoder.joblib")
        self.encoder = joblib.load(encoder_path)
        self.logger.info(f"Encoder loaded from: {encoder_path}")

    def encode_categories(self):
        """Step 3: Convert product_name and destination_city into numerical columns."""
        cat_cols = ["product_name", "destination_city"]

        self.data_pd = self.data.toPandas()
        encoded_data = self.encoder.transform(self.data_pd[cat_cols])
        encoded_cols = self.encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=self.data_pd.index)

        drop_cols = ["feature_uid", "transaction_date", "total_demand", "product_name", "destination_city"]
        self.X_test = pd.concat([self.data_pd.drop(columns=drop_cols), encoded_df], axis=1)
        self.y_test = self.data_pd["total_demand"]

        self.logger.info(f"X_test: {self.X_test.shape} | y_test: {self.y_test.shape}")

    def predict_demand(self):
        """Run all prediction steps: load, encode, predict, evaluate, save."""
        self.load_data()
        self.load_encoder()
        self.encode_categories()
        self._generate_predictions()
        self.evaluate()
        self.save_results()

    def _generate_predictions(self):
        """Step 4: Load the registered Random Forest model and generate predictions."""
        model_uri = f"models:/{self.model_name}/{self.latest_version}"
        self.logger.info(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        self.predictions = model.predict(self.X_test)
        self.logger.info(f"Generated {len(self.predictions):,} predictions")

    def evaluate(self):
        """Step 5: Compare predictions with actual total_demand."""
        mae = mean_absolute_error(self.y_test, self.predictions)
        mse = mean_squared_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)

        self.logger.info(f"--- Evaluation Metrics ---")
        self.logger.info(f"MAE:  {mae:.2f}")
        self.logger.info(f"MSE:  {mse:.2f}")
        self.logger.info(f"R2:   {r2:.4f}")

        return mae, mse, r2

    def save_results(self):
        """Step 6: Save predictions + actuals to a Delta table."""
        output_pd = self.data_pd[[
            "feature_uid", "transaction_date", "product_name",
            "destination_city", "total_demand",
        ]].copy()
        output_pd["predicted_demand"] = self.predictions

        self.logger.info(f"Saving {len(output_pd):,} rows to: {self.output_table}")
        self.spark.createDataFrame(output_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(self.output_table)
        self.logger.info(f"Results saved to: {self.output_table}")
