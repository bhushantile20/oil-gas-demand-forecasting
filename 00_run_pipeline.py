# Databricks notebook source
# DBTITLE 1,Pipeline Entry Point
# MAGIC %md
# MAGIC # Oil & Gas Demand Forecasting - Pipeline Entry Point

# COMMAND ----------

# DBTITLE 1,Run full pipeline
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Pipeline steps in order (paths relative to project root)
PIPELINE = [
    {"name": "Bronze Load",       "path": "Bronze/02_bronze"},
    {"name": "Silver Transform",   "path": "Silver/03_silver"},
    {"name": "Gold Aggregate",     "path": "Gold/04_gold_aggregate"},
    {"name": "Gold Features",      "path": "Gold/05_gold_features"},
    {"name": "Feature Store",     "path": "ML/06_feature_store"},
    # {"name": "Create 1-Day Target", "path": "notebooks/07_create_1day_target"},
    {"name": "Train RF Model (v1)",  "path": "ML/08_train_random_forest_1day"},
    {"name": "Demand Forecast Training", "path": "ML/Demand Forecast Model Training"},
    {"name": "Predictions & Evaluation", "path": "ML/Model Predictions and Evaluation"},
]

# Run each notebook in sequence
for i, step in enumerate(PIPELINE, 1):
    name = step["name"]
    path = step["path"]

    logger.info(f"[{i}/{len(PIPELINE)}] Starting: {name}")
    start = time.time()

    try:
        dbutils.notebook.run(path, 2400)
        elapsed = time.time() - start
        logger.info(f"[{i}/{len(PIPELINE)}] Completed: {name} ({elapsed:.1f}s)")
    except Exception as e:
        logger.error(f"[{i}/{len(PIPELINE)}] FAILED: {name} - {e}")
        raise

logger.info("=" * 50)
logger.info("Pipeline completed successfully!")
logger.info("=" * 50)

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %run 

# COMMAND ----------

