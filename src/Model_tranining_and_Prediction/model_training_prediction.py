import yaml
import logging
from pathlib import Path
from pyspark.sql import SparkSession

from Training.model_training import DemandForecastTraining
from Prediction.model_predictions import ModelPredictions
from Champion_Challenger.champion_challenger import ChampionChallengerEvaluator

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


def run_champion_challenger(config, spark, logging):
    catalog_name = config["model_training_prediction"]["catalog_name"]
    cc = config["model_training_prediction"]["champion_challenger"]

    evaluator = ChampionChallengerEvaluator(
        spark=spark,
        logging=logging,
        catalog_name=catalog_name,
        config=cc,
    )

    evaluator.evaluate()


# main execution part
if __name__ == "__main__":
    import sys
    spark = SparkSession.builder.getOrCreate()
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = Path(__file__).resolve().parents[2] / "config" / "config.yml"
    config = load_config(config_path)

    logging.info("Model Training - started training the RandomForest model. ")
    run_model_training(config, spark, logger)
    logging.info("Model Training completed. ")

    logging.info("Model Predictions & Evaluation - started generating predictions. ")
    run_model_predictions(config, spark, logger)
    logging.info("Model Predictions & Evaluation completed. ")

    logging.info("Champion-Challenger Evaluation - started comparing models. ")
    run_champion_challenger(config, spark, logger)
    logging.info("Champion-Challenger Evaluation completed.")
