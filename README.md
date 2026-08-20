# Crypto Forecaster - Declarative Automation Bundle (DAB)

## Project Structure (DAB Best Practices)

```
databricks-crypto-forecaster/
├── databricks.yml              # DAB bundle configuration (to be created)
├── init_catalog.sql            # One-time catalog setup
├── README.md                   # This file
│
├── src/                        # Source code (SQL, Python, notebooks)
│   ├── data_pipeline/          # Data ingestion and ETL
│   ├── feature_engineering/    # ML feature creation
│   └── model_training/         # Model training and deployment
│
├── resources/                  # DAB resource definitions (YAML)
│   └── (job and pipeline definitions will go here)
│
├── tests/                      # Unit and integration tests
│   └── (test files will go here)
│
└── config/                     # Configuration files
    └── (environment configs will go here)
```

## What is DAB?

Declarative Automation Bundles (DAB) is Databricks' framework for:
- **Infrastructure as Code**: Define jobs, pipelines, models as YAML
- **Version Control**: Everything in Git
- **CI/CD**: Deploy via `databricks bundle deploy`
- **Environment Management**: Dev, staging, prod configurations
- **Testing**: Integrated testing framework

## Quick Start

### 1. Initialize Catalog (One-time)
```bash
databricks bundle run init_catalog
```

### 2. Deploy Bundle (Future)
```bash
# Development environment
databricks bundle deploy --target dev

# Production environment
databricks bundle deploy --target prod
```

### 3. Run Workflows
```bash
# Run ingestion job
databricks bundle run ingestion_job

# Run full pipeline
databricks bundle run crypto_forecaster_pipeline
```

## Development Workflow

### Local Development
1. Edit code in `src/`
2. Test locally using `tests/`
3. Commit changes to Git

### Deployment
1. `databricks bundle validate` - Check bundle configuration
2. `databricks bundle deploy --target dev` - Deploy to dev environment
3. Test in dev environment
4. `databricks bundle deploy --target prod` - Deploy to production

## Folder Purposes

### `src/` - Source Code
All executable code lives here:
- **data_pipeline/**: SQL scripts for data ingestion and ETL
- **feature_engineering/**: Feature creation scripts
- **model_training/**: Python/SQL for ML training

**Convention**: Use relative paths from project root in DAB configs

### `resources/` - DAB Definitions
YAML files defining Databricks resources:
- Job definitions
- Pipeline definitions
- Model serving endpoints
- Permissions

**Example**: `resources/jobs/ingestion_job.yml`

### `tests/` - Testing
Unit and integration tests:
- SQL query tests
- Python unit tests
- End-to-end pipeline tests

### `config/` - Configuration
Environment-specific configurations:
- API endpoints
- Catalog names per environment
- Feature flags

## Architecture

### Data Flow
1. **Ingest**: Volume → `raw.coingecko` table
2. **Process**: Raw JSON → `processed.ohlc_data` table
3. **Features**: Processed → `features.ml_features` table
4. **Train**: Features → ML models
5. **Deploy**: Model → Serving endpoint
6. **Monitor**: Predictions → Dashboards

### Unity Catalog Structure
- **Catalog**: `dbx_joshy_demo` (dev), `dbx_joshy_prod` (prod)
- **Schemas**: `raw`, `processed`, `features`, `models`, `monitoring`

## Current Status

- [x] Folder structure organized for DAB
- [x] Data ingestion working
- [ ] Create `databricks.yml` bundle configuration
- [ ] Define jobs in `resources/`
- [ ] Add tests in `tests/`
- [ ] Set up CI/CD pipeline

## Next Steps

1. **Create Bundle Config**: Define `databricks.yml`
2. **Define Jobs**: Create job definitions in `resources/`
3. **Add Tests**: Write tests for SQL transformations
4. **CI/CD**: Set up GitHub Actions for deployment

## References

- [DAB Documentation](https://docs.databricks.com/dev-tools/bundles/index.html)
- [DAB Best Practices](https://docs.databricks.com/dev-tools/bundles/best-practices.html)
- [Bundle Settings Reference](https://docs.databricks.com/dev-tools/bundles/settings.html)
