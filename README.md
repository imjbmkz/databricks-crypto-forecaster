# Crypto Forecaster

An end-to-end cryptocurrency price forecasting system built on Databricks, leveraging Unity Catalog, Declarative Automation Bundles (DABs), and machine learning to predict cryptocurrency market movements.

## Overview

This project implements a complete MLOps pipeline for cryptocurrency forecasting:

* **Data Ingestion**: Automated ingestion of CoinGecko API data from Unity Catalog Volumes
* **Feature Engineering**: Advanced feature creation from historical price data (OHLC)
* **Model Training**: Machine learning models for price prediction
* **Orchestration**: Declarative Automation Bundle (DAB) for workflow management
* **Visualization**: Interactive dashboards for market analysis and model performance monitoring

## Architecture

```
Volume Storage (CoinGecko JSON)
        ↓
┌─────────────────────────────┐
│   Data Ingestion Layer      │
│  (SQL - Unity Catalog)      │
├─────────────────────────────┤
│ • ingest_volume_to_raw      │
│ • ingest_historical_chart   │
│ • ingest_ohlc_data          │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  Feature Engineering        │
│  (PySpark)                  │
├─────────────────────────────┤
│ • create_features.py        │
│ • feature_engineering_utils │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  Model Training             │
│  (PySpark + MLflow)         │
├─────────────────────────────┤
│ • model_training.py         │
│ • model_training_utils      │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│  Visualization              │
├─────────────────────────────┤
│ • Market Analysis Dashboard │
│ • Performance Control Chart │
└─────────────────────────────┘
```

## Project Structure

```
databricks-demo/
│
├── databricks.yml                      # DAB configuration
│
├── src/
│   ├── data_pipeline/                  # Data ingestion scripts
│   │   ├── ingest_volume_to_raw.sql   # Ingest raw JSON from Volume
│   │   ├── ingest_historical_chart_data.sql
│   │   └── ingest_ohlc_data.sql       # OHLC (Open, High, Low, Close) data
│   │
│   ├── features/                       # Feature engineering
│   │   ├── create_features.py         # Main feature creation script
│   │   ├── feature_engineering_utils.py
│   │   └── feature_engineering.ipynb  # Development notebook
│   │
│   ├── model_train/                    # Model training
│   │   ├── model_training.py          # Main training script
│   │   └── model_training_utils.py
│   │
│   ├── 00_init_catalogs_schemas       # Initialization query
│   └── playground                      # Experimentation query
│
├── experiments/                        # MLflow experiment tracking
│
├── Crypto Forecaster - Market Analysis    # Dashboard
└── Model Performance Control Chart         # Dashboard
```

## Prerequisites

* Databricks workspace with Unity Catalog enabled
* Unity Catalog volume with CoinGecko API data: `/Volumes/{catalog}/raw/vlm/coingecko/`
* SQL warehouse (warehouse_id: `afed8d7d8795bf0d` or update in `databricks.yml`)
* Databricks CLI installed (for bundle deployment)

## Setup & Deployment

### 1. Initialize Catalogs and Schemas

Run the initialization query to create the required Unity Catalog structure:

```sql
-- Run: src/00_init_catalogs_schemas
```

This creates:
* `{catalog}.raw` - Raw ingested data
* `{catalog}.features` - Feature engineered datasets
* `{catalog}.models` - Model outputs and forecasts

### 2. Configure the Bundle

Edit `databricks.yml` to set your catalog:

```yaml
variables:
  catalog:
    default: your_catalog_name
```

### 3. Deploy the Bundle

```bash
# Validate bundle configuration
databricks bundle validate

# Deploy to development environment
databricks bundle deploy -t dev

# Deploy to production environment
databricks bundle deploy -t prod
```

### 4. Run the Pipeline

```bash
# Run the entire crypto forecasting job
databricks bundle run crypto_forecasting_job -t dev
```

Or trigger from the Databricks UI: **Workflows** → **crypto_forecasting_job** → **Run now**

## Pipeline Overview

The DAB orchestrates 5 sequential tasks:

### Task 1: Ingest Volume to Raw
**File**: `src/data_pipeline/ingest_volume_to_raw.sql`
* Reads CoinGecko JSON files from Unity Catalog Volume
* Creates `{catalog}.raw.coingecko` table
* Captures metadata (file path, size, modification time)

### Task 2: Ingest Historical Chart Data
**File**: `src/data_pipeline/ingest_historical_chart_data.sql`
* Processes historical price charts
* Depends on Task 1

### Task 3: Ingest OHLC Data
**File**: `src/data_pipeline/ingest_ohlc_data.sql`
* Extracts Open, High, Low, Close price data
* Depends on Task 1

### Task 4: Create Features
**File**: `src/features/create_features.py`
* Engineers features for ML models
* Creates training and test datasets
* Outputs:
  * `{catalog}.features.cg_coin_forecast_features_train`
  * `{catalog}.features.cg_coin_forecast_features_test`
* Depends on Task 2

### Task 5: Model Training
**File**: `src/model_train/model_training.py`
* Trains forecasting models per cryptocurrency
* Logs experiments to MLflow
* Outputs:
  * `{catalog}.models.cg_coin_model_performance`
  * `{catalog}.models.cg_coin_forecasts`
* Depends on Task 4

## Usage

### Running Individual Tasks

You can execute tasks independently for development:

```bash
# Test data ingestion
databricks bundle run crypto_forecasting_job --task ingest_volume_to_raw -t dev

# Test feature engineering
databricks bundle run crypto_forecasting_job --task create_features -t dev

# Test model training
databricks bundle run crypto_forecasting_job --task model_training -t dev
```

### Monitoring & Debugging

* **Job Runs**: Navigate to Workflows → crypto_forecasting_job → Runs
* **MLflow Experiments**: Check `experiments/` folder for model tracking
* **Data Quality**: Query Unity Catalog tables directly

## Dashboards

### 1. Crypto Forecaster - Market Analysis
Interactive dashboard for:
* Price trends across cryptocurrencies
* Volume analysis
* Market sentiment indicators
* Forecast visualizations

### 2. Model Performance Control Chart
Monitors model quality:
* Prediction accuracy metrics
* Training/test performance comparison
* Model drift detection
* Performance trends over time

## Development

### Local Development

1. **Feature Engineering**: Use `src/features/feature_engineering.ipynb` for interactive development
2. **SQL Queries**: Test queries in `src/playground`
3. **Testing**: Validate changes with `databricks bundle validate`

### Environment Management

The bundle supports multiple deployment targets:

* **dev**: Development environment (`dbx_joshdevph_dev` catalog)
* **prod**: Production environment (`dbx_joshdevph_prd` catalog)

Switch environments using the `-t` flag:
```bash
databricks bundle deploy -t prod
```

### Compute Configuration

* **SQL Tasks**: Use serverless SQL warehouse
* **Python Tasks**: Run on serverless compute (environment version 4)
* No cluster management required

## Key Technologies

* **Databricks**: Unified analytics platform
* **Unity Catalog**: Data governance and cataloging
* **Declarative Automation Bundles (DABs)**: Infrastructure as code
* **Delta Lake**: Reliable data lake storage
* **MLflow**: Machine learning lifecycle management
* **PySpark**: Distributed data processing
* **Databricks SQL**: Data warehousing and analytics

## Data Sources

* **CoinGecko API**: Cryptocurrency market data
* **Volume Location**: `/Volumes/dbx_joshdevph_dev/raw/vlm/coingecko/`
* **Data Format**: JSON

## Contributing

When making changes:

1. Develop and test in the `dev` environment
2. Validate bundle configuration: `databricks bundle validate`
3. Deploy to dev: `databricks bundle deploy -t dev`
4. Test thoroughly before promoting to production
5. Deploy to prod: `databricks bundle deploy -t prod`

## Troubleshooting

### Common Issues

**Issue**: Task fails with "Table not found"
* **Solution**: Run `00_init_catalogs_schemas` to create required schemas

**Issue**: "Volume not found" error
* **Solution**: Verify CoinGecko data exists in `/Volumes/{catalog}/raw/vlm/coingecko/`

**Issue**: Model training fails
* **Solution**: Check feature tables contain data; verify training/test split

**Issue**: Bundle deployment fails
* **Solution**: Ensure you have proper permissions on Unity Catalog and workspace

## License

This project is designed for demonstration and educational purposes.

---

**Project**: joshdevph_crypto_forecaster  
**Catalog**: dbx_joshdevph_dev (dev) / dbx_joshdevph_prd (prod)  
**Environment**: Databricks Serverless