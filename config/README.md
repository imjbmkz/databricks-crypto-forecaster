# Configuration

## Purpose
Environment-specific configuration files.

## Structure

```
config/
├── dev.yml              # Development environment
├── staging.yml          # Staging environment
├── prod.yml             # Production environment
└── local.yml            # Local development
```

## Configuration Files

### `dev.yml`
Development environment settings:
```yaml
catalog_name: dbx_joshy_demo_dev
volume_path: /Volumes/dbx_joshy_demo_dev/coingecko/coingecko/coingecko/
cluster_id: 1234-567890-abcdef
api_endpoint: https://dev-api.example.com
```

### `prod.yml`
Production environment settings:
```yaml
catalog_name: dbx_joshy_demo_prod
volume_path: /Volumes/dbx_joshy_demo_prod/coingecko/coingecko/coingecko/
cluster_id: 9876-543210-fedcba
api_endpoint: https://api.example.com
```

## Usage in DAB

Reference in `databricks.yml`:
```yaml
targets:
  dev:
    variables:
      catalog_name: ${include.dev.catalog_name}
  prod:
    variables:
      catalog_name: ${include.prod.catalog_name}

include:
  - file: config/dev.yml
    target: dev
  - file: config/prod.yml
    target: prod
```

## Best Practices

1. **No secrets**: Use Databricks secrets, not config files
2. **Environment parity**: Keep structure consistent
3. **Validation**: Validate configs before deployment
4. **Documentation**: Comment non-obvious values
