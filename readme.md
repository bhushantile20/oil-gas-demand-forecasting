# Oil & Gas Demand Forecasting

An end-to-end demand forecasting pipeline for oil and gas supply chain data — built on Databricks with PySpark, MLflow, scikit-learn, and a Unity Catalog feature store. The project follows a **medallion architecture** (Bronze → Silver → Gold) for data engineering and a two-stage ML pipeline for model training and prediction. CI/CD is automated through GitHub Actions and Declarative Automation Bundles (DABs).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Project Structure](#project-structure)
4. [Tech Stack](#tech-stack)
5. [Prerequisites](#prerequisites)
6. [Configuration](#configuration)
7. [Unity Catalog Layout](#unity-catalog-layout)
8. [How to Run](#how-to-run)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Testing](#testing)
11. [Development](#development)
12. [License](#license)
13. [Author](#author)

---

## Architecture Overview

```
                          ┌─────────────┐
                          │  CSV Source  │
                          │ (UC Volume)  │
                          └──────┬──────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │     STAGE 1: Data Prep       │
                  │   preparation.py             │
                  │                              │
                  │  Bronze  →  Silver  →  Gold  │
                  │  (ingest)  (clean)   (agg+feat)│
                  │                │             │
                  │         Feature Store         │
                  │   (register features+labels) │
                  └──────────────┬──────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │  STAGE 2: Training & Pred    │
                  │  model_training_prediction.py │
                  │                              │
                  │  Train → Evaluate → Register │
                  │         (MLflow + UC)         │
                  │                │             │
                  │    Predictions + Evaluation  │
                  └──────────────┬──────────────┘
                                 │
                          ┌──────▼──────┐
                          │ Predictions  │
                          │   Table      │
                          └─────────────┘
```

The pipeline runs as two sequential tasks in a Databricks serverless job:

1. **Data Preparation** (`preparation.py`) — ingests raw CSV, cleans and transforms it through Bronze/Silver/Gold layers, and registers features in the Unity Catalog feature store.
2. **Model Training & Prediction** (`model_training_prediction.py`) — loads features and labels, trains a Random Forest model, registers it in MLflow, and generates predictions.

---

## Pipeline Stages

### Stage 1: Data Preparation

| Step | Module | Class | What It Does |
| --- | --- | --- | --- |
| Bronze | `Bronze/bronze_ingestion.py` | `BronzeLoader` | Reads CSV from a UC volume, adds `transaction_date` and audit columns (`_ingest_ts`), writes to a Delta bronze table |
| Silver | `Silver/silver_transformation.py` | `SilverTransformer` | Loads bronze data, deduplicates by `transaction_id` (keeping latest `_ingest_ts`), writes to a Delta silver table |
| Gold | `Gold/gold_aggregations.py` | `DailyDemandAggregator` + `DemandFeatureBuilder` | Aggregates daily demand, engineers time-series features (`demand_lag_1`, `demand_lag_7`, `rolling_avg_7`), writes to Delta gold tables |
| Feature Store | `Feature_store/feature_store.py` | `DemandFeatureStore` | Adds a primary key (`feature_uid`), registers the feature table in Unity Catalog, and saves label data (`total_demand`) |

Orchestrator: `src/Preparation/preparation.py`

### Stage 2: Model Training & Prediction

| Step | Module | Class | What It Does |
| --- | --- | --- | --- |
| Training | `Training/model_training.py` | `DemandForecastTraining` | Joins features + labels, handles missing values, splits chronologically (64/16/20%), OneHotEncodes categoricals, trains `RandomForestRegressor`, evaluates (MAE, MSE, RMSE, R2), logs to MLflow, registers model in UC |
| Prediction | `Prediction/model_predictions.py` | `ModelPredictions` | Loads the latest model version, encodes categories with the saved encoder, predicts demand, evaluates predictions, saves results to a predictions table |

Orchestrator: `src/Model_tranining_and_Prediction/model_training_prediction.py`

> **Note:** The `Champion_Challenger/` module is dead code — it is no longer imported by the orchestrator and can be safely deleted.

---

## Project Structure

```
oil-gas-demand-forecasting/
├── .github/
│   └── workflows/
│       ├── Ci.yml                  # CI: lint (ruff) → typecheck (mypy) → test (pytest)
│       └── deploy.yaml             # CD: validate + deploy bundle on push to main
├── config/
│   └── config.yml                  # All pipeline config (table names, model params, splits)
├── resources/
│   └── oil_gas_demand.yml          # DABs job definition (2 tasks, serverless, daily schedule)
├── src/
│   ├── Preparation/                # Stage 1: Bronze → Silver → Gold → Feature Store
│   │   ├── Bronze/
│   │   │   ├── __init__.py
│   │   │   └── bronze_ingestion.py
│   │   ├── Silver/
│   │   │   ├── __init__.py
│   │   │   └── silver_transformation.py
│   │   ├── Gold/
│   │   │   ├── __init__.py
│   │   │   └── gold_aggregations.py
│   │   ├── Feature_store/
│   │   │   ├── __init__.py
│   │   │   └── feature_store.py
│   │   └── preparation.py           # Entry point for Stage 1
│   └── Model_tranining_and_Prediction/   # Stage 2: Train → Predict
│       ├── Training/
│       │   ├── __init__.py
│       │   └── model_training.py
│       ├── Prediction/
│       │   ├── __init__.py
│       │   └── model_predictions.py
│       ├── Champion_Challenger/     # Dead code (not imported)
│       │   ├── __init__.py
│       │   └── champion_challenger.py
│       └── model_training_prediction.py  # Entry point for Stage 2
├── tests/
│   ├── conftest.py                 # Shared fixtures (Spark session, mocks)
│   ├── test_load_config.py         # Config loading tests
│   ├── test_model_training.py      # Feature prep + model evaluation tests
│   └── test_silver_transformation.py  # Silver deduplication tests
├── databricks.yml                  # Bundle config (name, targets, sync rules)
├── pyproject.toml                  # Ruff, mypy, pytest configuration
├── requirements.txt                # Runtime dependencies (installed on the job)
├── requirements-dev.txt            # Dev dependencies (CI: ruff, mypy, pytest, etc.)
├── .gitignore
└── readme.md
```

---

## Tech Stack

| Category | Technology |
| --- | --- |
| Cloud Platform | Databricks (serverless compute) |
| Data Processing | Apache Spark / PySpark |
| Storage | Delta Lake, Unity Catalog |
| Feature Management | Databricks Feature Store (`databricks-feature-engineering`) |
| ML Framework | scikit-learn (`RandomForestRegressor`) |
| Experiment Tracking | MLflow |
| Model Registry | Unity Catalog Models |
| CI/CD | GitHub Actions, Declarative Automation Bundles (DABs) |
| Code Quality | Ruff (lint), Mypy (type check), Pytest (unit tests) |
| Config | YAML |

---

## Prerequisites

1. **Databricks workspace** with Unity Catalog enabled
2. **GitHub repository** with the following secrets configured (Settings → Secrets and variables → Actions):
   - `DATABRICKS_HOST` — your workspace URL (e.g. `https://<workspace>.cloud.databricks.com`)
   - `DATABRICKS_CLIENT_ID` — service principal OAuth client ID
   - `DATABRICKS_CLIENT_SECRET` — service principal OAuth client secret
3. **Databricks CLI** installed locally (for manual bundle deploy/validation)
4. **Python 3.12** and **Java 17** (for running tests locally with Spark)

---

## Configuration

All pipeline configuration lives in **`config/config.yml`**. The file has two main sections:

### Data Preparation (`data_preparation`)

Controls the Bronze → Silver → Gold → Feature Store pipeline:

| Section | Key | Example Value | Description |
| --- | --- | --- | --- |
| `catalog_name` | — | `oil_gas_demo` | Unity Catalog name |
| `bronze.source` | `schema_name` | `raw` | Schema containing the source volume |
| `bronze.source` | `volume_name` | `raw_volume` | UC Volume with the raw CSV |
| `bronze.source` | `file_name` | `supply_chain_dataset.csv` | Source CSV file name |
| `bronze.target` | `table_name` | `supply_chain_transactions` | Bronze Delta table name |
| `silver.target` | `silver_table_name` | `supply_chain_transactions` | Silver Delta table name |
| `gold.target` | `aggregated_gold_table_name` | `daily_demand_agg` | Gold aggregation table |
| `gold.target` | `featured_gold_table_name` | `demand_features` | Gold feature table |
| `feature_store.target` | `feature_table_name` | `demand_feature_store` | Registered feature table |
| `feature_store.target` | `label_table_name` | `demand_labels` | Label table (`total_demand`) |

### Model Training & Prediction (`model_training_prediction`)

Controls the training and prediction pipeline:

| Section | Key | Example Value | Description |
| --- | --- | --- | --- |
| `catalog_name` | — | `oil_gas_demo` | Unity Catalog name |
| `model_training.features` | `ts_feature_cols` | `["demand_lag_1", ...]` | Time-series feature columns (nulls filled with 0.0) |
| `model_training.features` | `cat_cols` | `["product_name", "destination_city"]` | Categorical columns to OneHotEncode |
| `model_training.features` | `drop_cols` | `["feature_uid", ...]` | Columns dropped before training |
| `model_training.split` | `train` / `val` | `0.64` / `0.16` | Chronological split ratios (test = remaining) |
| `model_training.rf` | `n_estimators` | `200` | Number of trees in the Random Forest |
| `model_training.rf` | `random_state` | `42` | Random seed for reproducibility |
| `model_training.model` | `model_name` | `random_forest_demand` | Registered model name in UC |
| `model_training.model` | `experiment_name` | `/Shared/demand_forecasting_experiment` | MLflow experiment path |

---

## Unity Catalog Layout

The pipeline uses a single catalog with four schemas:

```
oil_gas_demo (catalog)
├── raw                    # Source volume: raw_volume/supply_chain_dataset.csv
├── bronze                 # supply_chain_transactions (raw ingested data)
├── silver                 # supply_chain_transactions (deduplicated, cleaned)
├── gold                   # daily_demand_agg, demand_features,
│                         # demand_feature_store, demand_labels
└── ml_model               # training_data, validation_data, testing_data,
                          # predictions, model_predictions_eval,
                          # random_forest_demand (registered model)
```

---

## How to Run

### Option 1: Databricks Job (Recommended)

The pipeline is deployed as a DABs bundle job with two sequential tasks:

```bash
# Validate the bundle configuration
databricks bundle validate --target dev

# Deploy the bundle (creates/updates the job in Databricks)
databricks bundle deploy --target dev
```

The job runs automatically every day at **6:00 AM IST** (cron: `0 0 6 * * ?`, timezone: `Asia/Kolkata`). To change the schedule, edit `resources/oil_gas_demand.yml` and redeploy.

### Option 2: CI/CD (Automatic on Push)

Pushing to the `main` branch triggers the `deploy.yaml` workflow, which validates and deploys the bundle automatically. No manual steps needed.

### Option 3: Manual Execution

From a Databricks notebook or interactive cluster:

```bash
# Stage 1: Data preparation
python src/Preparation/preparation.py

# Stage 2: Model training & prediction (run after Stage 1 completes)
python src/Model_tranining_and_Prediction/model_training_prediction.py
```

You can also pass a custom config path as a command-line argument:

```bash
python src/Preparation/preparation.py /path/to/custom-config.yml
```

---

## CI/CD Pipeline

The project uses two GitHub Actions workflows:

### CI — `.github/workflows/Ci.yml`

Triggers on **pull requests to `main`** and `workflow_dispatch`. Runs three sequential jobs:

| Job | Tool | Command | Checks |
| --- | --- | --- | --- |
| `lint` | Ruff | `ruff check src tests` | Import sorting, unused vars, style |
| `typecheck` | Mypy | `mypy src` | Static type analysis (needs `lint` to pass) |
| `test` | Pytest | `pytest -v` | Unit tests (needs `typecheck` to pass) |

The `test` job sets up Java 17 and `SPARK_LOCAL_IP=127.0.0.1` so Spark can run locally on the GitHub runner.

### CD — `.github/workflows/deploy.yaml`

Triggers on **push to `main`**. Runs two steps:

1. `databricks bundle validate --target dev` — validates the bundle configuration
2. `databricks bundle deploy --target dev` — deploys the updated job to Databricks

The deploy uses a **service principal** (machine-to-machine OAuth) for authentication, so no interactive login is needed.

---

## Testing

Tests are designed to run **without a live Databricks connection** — they use mocks and a local Spark session.

### Test Files

| File | Tests | What It Covers |
| --- | --- | --- |
| `test_load_config.py` | 2 | Config loading from YAML (catalog name, nested keys) |
| `test_model_training.py` | 4 | Feature preparation (X/y split, target dropped) and model evaluation (MAE, RMSE) |
| `test_silver_transformation.py` | 2 | Silver deduplication logic (keeps latest record per `transaction_id`) |

### Running Tests Locally

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest -v

# Run a specific test file
pytest -v tests/test_model_training.py
```

### Test Infrastructure

- `conftest.py` provides a session-scoped local `SparkSession` (for Silver tests) and `mock_spark` / `mock_logger` MagicMock fixtures (for training tests that do not need real Spark)
- Training tests mock the Spark DataFrame with `MagicMock` — no Spark/MLflow/Databricks needed
- Silver tests use a real local Spark session for DataFrame operations
- Config: `pyproject.toml` sets `pythonpath` to include both `src/` subdirectories so tests can import modules directly

---

## Development

### Code Quality

| Tool | Config | Purpose |
| --- | --- | --- |
| Ruff | `pyproject.toml` `[tool.ruff]` | Linting (E, F, W, I rules, line-length 120) |
| Mypy | `pyproject.toml` `[tool.mypy]` | Type checking (Python 3.12, `ignore_missing_imports`) |
| Pytest | `pyproject.toml` `[tool.pytest.ini_options]` | Test runner (`testpaths=["tests"]`, pythonpath includes `src/`) |

### Local Linting

```bash
ruff check src tests          # Lint
ruff check --fix src tests    # Auto-fix
mypy src                      # Type check
pytest -v                     # Run tests
```

### DABs Bundle

The bundle is defined in:

- `databricks.yml` — bundle name, sync rules, targets
- `resources/oil_gas_demand.yml` — job definition (2 tasks, serverless compute, daily schedule)

The job is `UI_LOCKED` — all changes must be made through the bundle YAML, not the Databricks UI.

### Important Notes for New Starters

* Folder names in `src/` are **case-sensitive** (e.g. `Bronze/` not `bronze/`) — imports use `from Bronze.bronze_ingestion import BronzeLoader`
* Config paths are derived from the script location using `inspect.currentframe()` (not `__file__`) because Databricks serverless runs scripts via `exec()` where `__file__` is undefined
* The `requirements.txt` is installed on the job's serverless environment via the `environments` block in the bundle resource file
* The CI service principal needs `USE SCHEMA` + `MODIFY` grants on all schemas in `oil_gas_demo` for pipeline writes to succeed

---

## License

This project is for educational and demonstration purposes.

---

## Author

**Bhushan Tile**
