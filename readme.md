# Oil & Gas Demand Forecasting

[![CI](https://github.com/bhushantile20/oil-gas-demand-forecasting/actions/workflows/Ci.yml/badge.svg)](https://github.com/bhushantile20/oil-gas-demand-forecasting/actions/workflows/Ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Databricks](https://img.shields.io/badge/platform-Databricks-FF3621)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)

An end-to-end demand forecasting solution for oil and gas supply chain data, built on Databricks with PySpark, MLflow, scikit-learn and a Unity Catalog feature store. It covers the full lifecycle: a **medallion data pipeline** (Bronze → Silver → Gold), **model training and registration**, **champion–challenger model promotion**, **data drift monitoring**, and an **interactive AI/BI dashboard**. Everything is deployed and scheduled as code with Declarative Automation Bundles (DABs) and GitHub Actions.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Dashboard](#dashboard)
3. [Architecture](#architecture)
4. [Project Flow](#project-flow)
5. [Pipeline Stages](#pipeline-stages)
6. [Data Drift Monitoring](#data-drift-monitoring)
7. [Project Structure](#project-structure)
8. [Tech Stack](#tech-stack)
9. [Prerequisites](#prerequisites)
10. [Configuration](#configuration)
11. [Unity Catalog Layout](#unity-catalog-layout)
12. [How to Run](#how-to-run)
13. [CI/CD](#cicd)
14. [Testing](#testing)
15. [Development Notes](#development-notes)
16. [Roadmap](#roadmap)
18. [Author](#author)

---

## Highlights

- **Medallion architecture** with Delta Lake: raw CSV → Bronze → Silver (deduplicated) → Gold (daily demand and time-series features).
- **Feature store** in Unity Catalog with a registered feature table and a separate label table.
- **Random Forest forecasting model** trained on a chronological 64 / 16 / 20 split, tracked in MLflow and registered in Unity Catalog.
- **Champion–challenger evaluation**: every newly trained version is compared with the current champion on validation data and promoted only if its MAE is lower.
- **Data drift monitoring** using the two-sample Kolmogorov–Smirnov test on every model input feature.
- **Interactive dashboard** for demand analysis, forecast accuracy and drift status, with filters and Genie for natural-language questions.
- **Fully automated**: a daily scheduled Databricks job, CI (lint, type check, tests) on pull requests, and automatic deployment on merge to `main`.

---

## Dashboard

The Databricks AI/BI dashboard has three tabs: **Demand Overview**, **DriftWatch** and **Data Drift Monitoring**.

### Demand Overview

Shows how well the model forecasts demand and where demand comes from.

Demand Overview dashboard
<img width="941" height="431" alt="dashboard_demand_overview" src="https://github.com/user-attachments/assets/214d900d-a5d1-4919-89bf-454d6ee401b4" />

| Component | What it shows |
| --- | --- |
| Global filters | Year, Month, Day, Product and Destination City filters that apply across the dashboard |
| KPI cards | Total predictions made, MAE, RMSE and R² |
| Actual vs Predicted Demand | Line chart of actual and predicted demand over time (weekly), to see how closely the forecast follows reality |
| Demand by Destination City | Bar chart of total demand for each city |
| Product × City Demand | Pivot table of total demand for every product and destination city combination |
| Ask Genie | Natural-language questions about the data directly from the dashboard |

### Data Drift Monitoring

Shows whether the data the model predicts on still looks like the data it was trained on.
<img width="956" height="369" alt="dashboard_data_drift_monitoring" src="https://github.com/user-attachments/assets/bf8f3a13-1315-451a-8234-7ebee0255222" />

| Component | What it shows |
| --- | --- |
| Summary cards | Total features monitored, drifted features, stable features, maximum KS statistic and minimum p-value |
| KS Statistic by Feature | Size of the distribution difference for each feature |
| P-value by Feature | Statistical significance for each feature in the latest run, colored by drift status |
| Drift Status Distribution | Share of features that are stable vs drifted |
| KS Statistic Over Time | Trend of the KS statistic per feature, which fills in as monitoring runs accumulate |
| Feature-Level Drift Results | Table of `ks_stat`, `p_value`, `drift_status`, reference row count, current row count and run timestamp |

> **Note:** the **DriftWatch** tab is part of the same dashboard and is not described here.

---

## Architecture
<img width="2006" height="1242" alt="architecture" src="https://github.com/user-attachments/assets/7f4d0ce9-6ceb-41c8-a3a2-c27924294580" />

The diagram shows the main building blocks:

- **Data source:** a raw CSV file stored in a Unity Catalog volume.
- **Databricks workflow job:** runs daily on serverless compute and executes the pipeline tasks. Data lands in Delta tables under the `oil_gas_demo` catalog, models are tracked in MLflow and registered in Unity Catalog, and drift results are saved to a table.
- **Consumption:** an AI/BI dashboard (with Genie) reads the predictions and drift results.
- **CI/CD:** pull requests are checked by GitHub Actions (lint, type check, tests). Merging to `main` validates and deploys the Databricks Asset Bundle, which creates or updates the job.

The pipeline is deployed as a Databricks serverless job with three tasks:

| Order | Task key | Entry point | Purpose |
| --- | --- | --- | --- |
| 1 | `data_preparation` | `src/Preparation/preparation.py` | Ingest, clean and transform data; register features and labels |
| 2 | `model_training_prediction` | `src/Model_tranining_and_Prediction/model_training_prediction.py` | Train, evaluate, register the model and generate predictions |
| 3 | `champion_challenger_evaluation` | `src/Model_tranining_and_Prediction/Champion_Challenger/champion_challenger.py` | Compare the new model version with the champion and promote it if it is better |

The job runs daily at **6:00 AM IST** and sends email notifications on start, success and failure.

---

## Project Flow

<img width="2006" height="1300" alt="project_flow" src="https://github.com/user-attachments/assets/68909a90-70dd-467d-b5d8-190ef8271d94" />

| Steps | Stage | What happens |
| --- | --- | --- |
| 1 to 6 | Data preparation | The CSV is ingested into Bronze, deduplicated into Silver, aggregated into daily demand (Gold), enriched with lag and rolling features, and registered in the feature store with a separate label table |
| 7 to 9 | Training | Features and labels are joined, split chronologically (64 / 16 / 20), encoded, used to train a Random Forest, and evaluated with MAE, MSE, RMSE and R² |
| 10 to 11 | Registration and prediction | The model is registered in Unity Catalog as a new `challenger` version, and predictions are written to the results tables |
| 12 to 13 | Champion–challenger | The challenger is compared with the champion on validation data and promoted only if its MAE is lower |
| 14 to 16 | Monitoring and reporting | A KS drift test compares training and prediction data, the results are stored in a table, and the dashboard shows demand, forecast accuracy and drift |

Steps 13 and 14 happen independently of each other, after steps 12 and 11 respectively.

---

## Pipeline Stages

### Stage 1: Data Preparation

| Step | Module | Class | What it does |
| --- | --- | --- | --- |
| Bronze | `Bronze/bronze_ingestion.py` | `BronzeLoader` | Reads the CSV from a UC volume, adds `transaction_date` and audit columns (`_ingest_ts`), writes a Delta bronze table |
| Silver | `Silver/silver_transformation.py` | `SilverTransformer` | Deduplicates by `transaction_id`, keeping the latest `_ingest_ts`, and writes a Delta silver table |
| Gold | `Gold/gold_aggregations.py` | `DailyDemandAggregator`, `DemandFeatureBuilder` | Aggregates daily demand per product and city and builds `demand_lag_1`, `demand_lag_7` and `rolling_avg_7` |
| Feature store | `Feature_store/feature_store.py` | `DemandFeatureStore` | Adds the primary key `feature_uid`, registers the feature table in Unity Catalog and saves the label table (`total_demand`) |

### Stage 2: Model Training and Prediction

| Step | Module | Class | What it does |
| --- | --- | --- | --- |
| Training | `Training/model_training.py` | `DemandForecastTraining` | Joins features and labels, handles missing values, splits chronologically (64 / 16 / 20), one-hot encodes `product_name` and `destination_city`, trains a `RandomForestRegressor`, logs MAE, MSE, RMSE and R² to MLflow, registers the model in Unity Catalog and sets the `challenger` alias (bootstrapping `champion` on the first run) |
| Prediction | `Prediction/model_predictions.py` | `ModelPredictions` | Loads the latest model version and the saved encoder, predicts demand on the test set, evaluates it and saves the results table |


### Stage 3: Champion–Challenger Evaluation

`Champion_Challenger/champion_challenger.py` (`ChampionChallengerEvaluator`) loads the models behind the `champion` and `challenger` aliases, predicts on the validation set with both, and compares MAE, RMSE and R². If the challenger has the lower MAE, it is promoted to `champion` and the old champion keeps the `previous-champion` alias. Otherwise the champion stays in place.

---

## Data Drift Monitoring

Model quality depends on the data staying similar to what the model learned from. The monitoring step compares a **reference** dataset (the training data) with a **current** dataset (the data the model makes predictions on) for each model input feature.

**Monitored features (6):** `avg_unit_price`, `total_inventory`, `demand_lag_1`, `demand_lag_7`, `rolling_avg_7`, `day_of_week`.

**Method:** the two-sample Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`) is run for every feature and produces:

| Field | Meaning |
| --- | --- |
| `ks_stat` | Largest gap between the two cumulative distributions. 0 means identical, 1 means completely different |
| `p_value` | Probability of seeing a difference this large by chance if the distributions were the same |
| `drift_status` | `DRIFT` when `p_value < 0.05`, otherwise `STABLE` |
| `reference_count`, `current_count` | Number of rows in each dataset |
| `run_timestamp` | When the monitoring run happened |

Results are saved to a Delta table that the dashboard reads.

### Example run (22 Oct 2026)

Reference rows: 46,950. Current rows: 15,025. One feature of six was flagged.

| Feature | KS statistic | P-value | Status |
| --- | --- | --- | --- |
| `rolling_avg_7` | 0.0219 | 0.000035 | DRIFT |
| `demand_lag_1` | 0.0122 | 0.068445 | STABLE |
| `demand_lag_7` | 0.0097 | 0.237233 | STABLE |
| `total_inventory` | 0.0076 | 0.528527 | STABLE |
| `avg_unit_price` | 0.0070 | 0.638449 | STABLE |
| `day_of_week` | 0.0058 | 0.836231 | STABLE |

### Reading the results

- With large datasets, even a very small shift gives a small p-value. In the example above, `rolling_avg_7` is flagged as drift, but its KS statistic of 0.0219 means the distributions are very close. Use `ks_stat` to judge **how big** a shift is, and the p-value to judge **whether** it is statistically detectable.
- The "KS Statistic Over Time" chart shows a single point until more monitoring runs are recorded.

---

## Project Structure

```
oil-gas-demand-forecasting/
├── .github/
│   └── workflows/
│       ├── Ci.yml                       # CI: lint (ruff) -> type check (mypy) -> tests (pytest)
│       └── deploy.yaml                  # CD: validate and deploy the bundle on push to main
├── config/
│   └── config.yml                       # Pipeline configuration (tables, model, splits, aliases)
├── docs/
│   └── images/                          # Dashboard screenshots and diagrams used in this README
├── resources/
│   └── oil_gas_demand.yml               # DABs job definition (serverless, daily schedule)
├── src/
│   ├── Preparation/                     # Stage 1: Bronze -> Silver -> Gold -> Feature Store
│   │   ├── Bronze/bronze_ingestion.py
│   │   ├── Silver/silver_transformation.py
│   │   ├── Gold/gold_aggregations.py
│   │   ├── Feature_store/feature_store.py
│   │   └── preparation.py               # Entry point for Stage 1
│   └── Model_tranining_and_Prediction/  # Stages 2 and 3
│       ├── Training/model_training.py
│       ├── Prediction/model_predictions.py
│       ├── Champion_Challenger/champion_challenger.py
│       └── model_training_prediction.py # Entry point for Stage 2
├── tests/
│   ├── conftest.py                      # Shared fixtures (local Spark session, mocks)
│   ├── test_load_config.py
│   ├── test_model_training.py
│   └── test_silver_transformation.py
├── databricks.yml                       # Bundle config (name, targets, sync rules)
├── pyproject.toml                       # Ruff, mypy and pytest configuration
├── requirements.txt                     # Runtime dependencies for the job
├── requirements-dev.txt                 # CI and development dependencies
└── readme.md
```

---

## Tech Stack

| Category | Technology |
| --- | --- |
| Platform | Databricks (serverless compute) |
| Data processing | Apache Spark / PySpark |
| Storage | Delta Lake, Unity Catalog |
| Feature management | Databricks Feature Store (`databricks-feature-engineering`) |
| Modeling | scikit-learn (`RandomForestRegressor`) |
| Statistics | SciPy (Kolmogorov–Smirnov test) |
| Experiment tracking and registry | MLflow, Unity Catalog Models |
| Dashboard | Databricks AI/BI dashboards and Genie |
| CI/CD | GitHub Actions, Declarative Automation Bundles |
| Code quality | Ruff, Mypy, Pytest |
| Configuration | YAML |

---

## Prerequisites

1. A **Databricks workspace** with Unity Catalog enabled.
2. A **GitHub repository** with these Actions secrets (Settings → Secrets and variables → Actions):
   - `DATABRICKS_HOST` – the workspace URL
   - `DATABRICKS_CLIENT_ID` – service principal OAuth client ID
   - `DATABRICKS_CLIENT_SECRET` – service principal OAuth client secret
3. The **Databricks CLI** (for manual bundle validation and deployment).
4. **Python 3.12** and **Java 17** (to run the tests locally with Spark).

---

## Configuration

All pipeline settings live in `config/config.yml`.

### Data preparation (`data_preparation`)

| Section | Key | Example | Description |
| --- | --- | --- | --- |
| `catalog_name` | – | `oil_gas_demo` | Unity Catalog name |
| `bronze.source` | `schema_name`, `volume_name`, `file_name` | `raw`, `raw_volume`, `supply_chain_dataset.csv` | Location of the source CSV |
| `bronze.target` | `table_name` | `supply_chain_transactions` | Bronze table |
| `silver.target` | `silver_table_name` | `supply_chain_transactions` | Silver table |
| `gold.target` | `aggregated_gold_table_name` | `daily_demand_agg` | Daily aggregation table |
| `gold.target` | `featured_gold_table_name` | `demand_features` | Feature table |
| `feature_store.target` | `feature_table_name`, `label_table_name` | `demand_feature_store`, `demand_labels` | Registered features and labels |

### Model training and prediction (`model_training_prediction`)

| Section | Key | Example | Description |
| --- | --- | --- | --- |
| `model_training.features` | `ts_feature_cols` | `demand_lag_1`, `demand_lag_7`, `rolling_avg_7` | Time-series features (nulls filled with 0.0) |
| `model_training.features` | `cat_cols` | `product_name`, `destination_city` | Categorical columns that are one-hot encoded |
| `model_training.split` | `train`, `val` | `0.64`, `0.16` | Chronological split (the rest is the test set) |
| `model_training.rf` | `n_estimators`, `random_state` | `200`, `42` | Random Forest settings |
| `model_training.model` | `model_name`, `experiment_name` | `random_forest_demand`, `/Shared/demand_forecasting_experiment` | Registered model and MLflow experiment |
| `model_training.model` | `champion_alias`, `challenger_alias` | `champion`, `challenger` | Model registry aliases |
| `champion_challenger` | `validation_table`, aliases | `validation_data` | Settings for the evaluation task |

---

## Unity Catalog Layout

```
oil_gas_demo (catalog)
├── raw          # Source volume with supply_chain_dataset.csv
├── bronze       # supply_chain_transactions (raw ingested data)
├── silver       # supply_chain_transactions (deduplicated)
├── gold         # daily_demand_agg, demand_features, demand_feature_store, demand_labels
└── ml_model     # predictions, model_predictions_eval,
                 # training_data / validation_data / testing_data (dataset splits),
                 # random_forest_demand (registered model)
```

---

## How to Run

### Option 1: Databricks job (recommended)

```bash
databricks bundle validate --target dev
databricks bundle deploy --target dev
```

The job runs every day at 6:00 AM IST (`0 0 6 * * ?`, timezone `Asia/Kolkata`). To change the schedule, edit `resources/oil_gas_demand.yml` and redeploy.

### Option 2: Automatic deployment

Merging or pushing to `main` triggers `deploy.yaml`, which validates and deploys the bundle. No manual steps are needed.

### Option 3: Manual execution

From a Databricks notebook or an interactive cluster:

```bash
python src/Preparation/preparation.py
python src/Model_tranining_and_Prediction/model_training_prediction.py
python src/Model_tranining_and_Prediction/Champion_Challenger/champion_challenger.py
```

Each script accepts an optional custom config path as its first argument:

```bash
python src/Preparation/preparation.py /path/to/custom-config.yml
```

---

## CI/CD

### Continuous integration: `.github/workflows/Ci.yml`

Runs on pull requests to `main` and on manual dispatch:

| Job | Tool | Command |
| --- | --- | --- |
| `lint` | Ruff | `ruff check src tests` |
| `typecheck` | Mypy | `mypy src` (after `lint` passes) |
| `test` | Pytest | `pytest -v` (after `typecheck` passes) |

The test job sets up Java 17 and `SPARK_LOCAL_IP=127.0.0.1` so Spark can run on the GitHub runner.

### Continuous deployment: `.github/workflows/deploy.yaml`

Runs on every push to `main`:

1. `databricks bundle validate --target dev`
2. `databricks bundle deploy --target dev`

Authentication uses a service principal (machine-to-machine OAuth), so no interactive login is needed.

---

## Testing

Tests run without a live Databricks connection, using mocks and a local Spark session.

| File | Tests | Covers |
| --- | --- | --- |
| `test_load_config.py` | 2 | Config loading from YAML |
| `test_model_training.py` | 4 | Feature preparation and model evaluation |
| `test_silver_transformation.py` | 2 | Silver deduplication (latest record per `transaction_id`) |

```bash
pip install -r requirements-dev.txt
pytest -v
```

---

## Development Notes

- **Code quality:** `ruff check src tests`, `mypy src` and `pytest -v` must pass before merging. Settings are in `pyproject.toml` (line length 120, Python 3.12).
- **Folder names are case-sensitive.** Imports look like `from Bronze.bronze_ingestion import BronzeLoader`.
- **Config path discovery** uses `inspect.currentframe()` rather than `__file__`, because Databricks serverless runs scripts through `exec()` where `__file__` is undefined.
- **Job dependencies** come from `requirements.txt`, installed through the `environments` block in `resources/oil_gas_demand.yml`.
- **Locked job:** the job is `UI_LOCKED`, so change it through the bundle YAML, not the Databricks UI.
- **Permissions:** the deployment service principal needs `USE SCHEMA` and `MODIFY` on the schemas in `oil_gas_demo`.

---

## Roadmap

- Run drift monitoring as a task in the bundle job so results refresh automatically after every training run.
- Add PSI and weekly model-error tracking (MAE against the training baseline) alongside the KS test.
- Send an alert when a feature drifts or the model error rises.
- Trigger retraining when drift persists across several runs.
- Add unit tests for the Gold and Prediction layers.

---

## Author
GitHub: [@bhushantile20](https://github.com/bhushantile20)
