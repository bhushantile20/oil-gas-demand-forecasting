# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Model Predictions & Evaluation
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Imports + Config
import mlflow
import joblib
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Config
CATALOG = "oil_gas_demo"
TEST_TABLE = f"{CATALOG}.ml_model.testing_data"
MODEL_NAME = f"{CATALOG}.ml_model.random_forest_demand"
OUTPUT_TABLE = f"{CATALOG}.ml_model.model_predictions_eval"
MODEL_ALIAS = "dev"

client = MlflowClient()
print("Config set.")

# COMMAND ----------

# DBTITLE 1,ModelPredictions class + execute
class ModelPredictions:
    """Loads test data, encodes categoricals, predicts demand, and evaluates."""

    def __init__(self, test_table, model_name, output_table, alias):
        self.test_table = test_table
        self.model_name = model_name
        self.output_table = output_table
        self.alias = alias
        self.data = None
        self.data_pd = None
        self.encoder = None
        self.X_test = None
        self.y_test = None
        self.predictions = None

    def load_data(self):
        """Step 1: Load the saved testing dataset."""
        print(f"Loading test data from: {self.test_table}")
        self.data = spark.table(self.test_table)
        print(f"Loaded {self.data.count():,} rows")

    def load_encoder(self):
        """Step 2: Load the OneHotEncoder saved during model training."""
        model_version = client.get_model_version_by_alias(self.model_name, self.alias)
        run_id = model_version.run_id

        print(f"Downloading encoder from run: {run_id}")
        encoder_path = client.download_artifacts(run_id=run_id, path="encoder/encoder.joblib")
        self.encoder = joblib.load(encoder_path)
        print(f"Encoder loaded from: {encoder_path}")

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

        print(f"X_test: {self.X_test.shape} | y_test: {self.y_test.shape}")
        print(f"Feature columns: {list(self.X_test.columns)}")

    def predict_demand(self):
        """Step 4: Load the registered Random Forest model and generate predictions."""
        model_uri = f"models:/{self.model_name}@{self.alias}"
        print(f"Loading model from: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        self.predictions = model.predict(self.X_test)
        print(f"Generated {len(self.predictions):,} predictions")

    def evaluate(self):
        """Step 5: Compare predictions with actual total_demand."""
        mae = mean_absolute_error(self.y_test, self.predictions)
        mse = mean_squared_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)

        print("--- Evaluation Metrics ---")
        print(f"MAE:  {mae:.2f}")
        print(f"MSE:  {mse:.2f}")
        print(f"R2:   {r2:.4f}")

        return mae, mse, r2

    def save_results(self):
        """Step 6: Save predictions + actuals to a Delta table."""
        output_pd = self.data_pd[[
            "feature_uid", "transaction_date", "product_name",
            "destination_city", "total_demand",
        ]].copy()
        output_pd["predicted_demand"] = self.predictions

        print(f"Saving {len(output_pd):,} rows to: {self.output_table}")
        spark.createDataFrame(output_pd).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(self.output_table)
        print(f"Results saved to: {self.output_table}")

    def run(self):
        """Execute all steps in order."""
        self.load_data()
        self.load_encoder()
        self.encode_categories()
        self.predict_demand()
        mae, mse, r2 = self.evaluate()
        self.save_results()

        # print("\n" + "=" * 50)
        # print("PREDICTION & EVALUATION COMPLETE")
        # print("=" * 50)
        # print(f"Test rows: {len(self.y_test):,}")
        # print(f"MAE:  {mae:.2f}")
        # print(f"MSE:  {mse:.2f}")
        # print(f"R2:   {r2:.4f}")
        # print(f"Model: {self.model_name}@{self.alias}")
        # print(f"Output: {self.output_table}")
        # print("=" * 50)


# Execute
predictor = ModelPredictions(
    test_table=TEST_TABLE,
    model_name=MODEL_NAME,
    output_table=OUTPUT_TABLE,
    alias=MODEL_ALIAS,
)
predictor.run()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# Loads the saved testing dataset, downloads the fitted OneHotEncoder from MLflow, encodes categorical columns, loads the registered Random Forest model by alias, generates predictions, and evaluates against actual `total_demand`.

# **Pipeline:** Load test data → Download encoder → Encode → Load model → Predict → Evaluate → Save results