# Oil & Gas Demand Forecasting

This project builds an end-to-end demand forecasting pipeline for oil and gas supply chain data using Databricks, Spark, MLflow, and a feature store workflow.

## Overview

The pipeline performs the following stages:

- Bronze layer: raw CSV ingestion into a bronze table
- Silver layer: data cleaning and transformations
- Gold layer: aggregation and feature engineering
- Feature store: registration of features and labels for model training
- Model training: Random Forest forecasting model
- Model evaluation: prediction and champion/challenger comparison

## Project Structure

```text
oil-gas-demand-forecasting/
├── config/
│   └── config.yml
├── resources/
│   └── oil_gas_demand.yml
├── src/
│   ├── Model_tranining_and_Prediction/
│   │   ├── Champion_Challenger/
│   │   ├── Prediction/
│   │   └── Training/
│   └── Preparation/
├── databricks.yml
├── requirements.txt
├── readme.md
└── .gitignore
```

## Tech Stack

- Python
- PySpark
- Databricks
- MLflow
- scikit-learn
- Databricks Feature Store
- YAML configuration files

## Setup

1. Create or open a Databricks workspace.
2. Upload the project to your Databricks workspace or clone it locally.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Update the workspace path and Databricks host in the following files:

- `databricks.yml`
- `src/Preparation/preparation.py`
- `src/Model_tranining_and_Prediction/model_training_prediction.py`

5. Ensure the catalog, schema, and volume names from `config/config.yml` exist in Databricks.

## Configuration

The pipeline configuration is managed in:

- `config/config.yml`

This file defines:

- bronze, silver, gold, and feature store table names
- model training parameters
- feature columns and split configuration
- Random Forest model settings

## Run the Pipeline

### Databricks Job

Deploy and run the Databricks bundle:

```bash
databricks bundle deploy
databricks bundle run
```

### Manual Execution

Run the preparation pipeline:

```bash
python src/Preparation/preparation.py
```

Run the model pipeline:

```bash
python src/Model_tranining_and_Prediction/model_training_prediction.py
```

## Notes

- This project is designed for Databricks environments and uses Unity Catalog-style table naming.
- The script paths and workspace paths should be adjusted to match your actual Databricks workspace.

## License

This project is for educational and demonstration purposes.

## Author

Bhushan Tile

