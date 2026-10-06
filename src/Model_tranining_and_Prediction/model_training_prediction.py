import yaml
import logging
from pathlib import Path
from pyspark.sql import SparkSession

from Training.model_training import DemandForecastTraining
from Prediction.model_predictions import ModelPredictions

# logging configurations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def run_model_training(config, spark, logging):
    catalog_name = config["model_training_prediction"]["catalog_name"]
    mt = config["model_training_prediction"]["model_training"]

    training_obj = DemandForecastTraining(
        spark=spark,
        logging=logging,
        catalog_name=catalog_name,
        config=mt,
    )

    training_obj.load_data()
    training_obj.start_training()
    training_obj.register_model()


def run_model_predictions(config, spark, logging):
    catalog_name = config["model_training_prediction"]["catalog_name"]
    mp = config["model_training_prediction"]["model_predictions"]

    predictions_obj = ModelPredictions(
        spark=spark,
        logging=logging,
        catalog_name=catalog_name,
        config=mp,
    )

    predictions_obj.predict_demand()


# main execution part
if __name__ == "__main__":
    import sys
    import inspect
    spark = SparkSession.builder.getOrCreate()

    # Allow custom config path via command-line argument
    if len(sys.argv) > 1 and sys.argv[1].endswith((".yml", ".yaml")):
        config_path = sys.argv[1]
    else:
        # Derive config path from script location using inspect.
        # inspect.currentframe().f_code.co_filename works in both normal Python
        # and exec() context (Databricks serverless spark_python_task) where
 
        script_path = inspect.currentframe().f_code.co_filename
        config_path = str(Path(script_path).resolve().parents[2] / "config" / "config.yml")
    config = load_config(config_path)

    logging.info("Model Training - started training the RandomForest model. ")
    run_model_training(config, spark, logger)
    logging.info("Model Training completed. ")

    logging.info("Model Predictions & Evaluation - started generating predictions. ")
    run_model_predictions(config, spark, logger)
    logging.info("Model Predictions & Evaluation completed. ")
