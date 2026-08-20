# Resources

## Purpose
DAB resource definitions (YAML files) for Databricks resources.

## Structure

```
resources/
├── jobs/                    # Job definitions
│   ├── ingestion_job.yml
│   ├── feature_engineering_job.yml
│   └── model_training_job.yml
│
├── pipelines/               # Pipeline definitions
│   └── crypto_forecaster_pipeline.yml
│
└── models/                  # Model serving endpoints
    └── crypto_model_endpoint.yml
```

## Job Definitions

Jobs define scheduled or triggered tasks.

**Example**: `jobs/ingestion_job.yml`
```yaml
resources:
  jobs:
    ingestion_job:
      name: "Crypto Ingestion Job"
      tasks:
        - task_key: ingest_data
          sql_task:
            file:
              path: ../src/data_pipeline/ingestion.sql
          existing_cluster_id: ${var.cluster_id}
      schedule:
        quartz_cron_expression: "0 0 * * * ?"  # Every hour
        timezone_id: "UTC"
```

## Pipeline Definitions

Pipelines define end-to-end workflows.

**Example**: `pipelines/crypto_forecaster_pipeline.yml`
```yaml
resources:
  pipelines:
    crypto_forecaster:
      name: "Crypto Forecaster Pipeline"
      target: "dbx_joshy_demo"
      libraries:
        - notebook:
            path: ../src/data_pipeline/ingestion.sql
        - notebook:
            path: ../src/feature_engineering/time_series_features.sql
      configuration:
        "spark.sql.shuffle.partitions": "8"
```

## Model Endpoints

Define model serving endpoints.

**Example**: `models/crypto_model_endpoint.yml`
```yaml
resources:
  model_serving_endpoints:
    crypto_forecaster_endpoint:
      name: "crypto-forecaster"
      config:
        served_models:
          - model_name: "crypto_forecaster"
            model_version: "1"
            workload_size: "Small"
            scale_to_zero_enabled: true
```

## Environment Variables

Reference variables from `databricks.yml`:
- `${var.cluster_id}` - Cluster ID
- `${var.catalog_name}` - Catalog name
- `${workspace.host}` - Workspace URL

## Validation

Validate resource definitions:
```bash
databricks bundle validate
```

## Best Practices

1. **One resource per file**: Easier to manage
2. **Clear naming**: `<resource_type>_<name>.yml`
3. **Comments**: Explain non-obvious configurations
4. **Variables**: Use variables for environment-specific values
