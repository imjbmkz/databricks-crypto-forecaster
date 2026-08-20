# Crypto Forecaster - SQL-Based DevOps Workflow

## Overview
This project uses SQL scripts instead of notebooks for better DevOps practices:
- Version control friendly (plain SQL files)
- CI/CD pipeline compatible
- Database-first approach
- Streaming ingestion from Unity Catalog Volumes

## Architecture

### Data Flow
1. **Upload**: CoinGecko API responses (JSON files) → Unity Catalog Volume
2. **Ingest**: Streaming table reads from Volume → `dbx_joshy_demo.raw.coingecko`
3. **Process**: Transform raw data → processed tables
4. **Feature Engineering**: Create ML-ready features
5. **Model Training**: Train forecasting models
6. **Monitor**: Dashboard and metrics

### Volume Path
```
/Volumes/dbx_joshy_demo/coingecko/coingecko/coingecko/
```

This is where you upload your CoinGecko JSON response files.

## Files

### SQL Scripts
1. **`init_catalog.sql`** - Initialize catalog and schemas
2. **`ingestion.sql`** - Create streaming table for data ingestion

### Legacy (Deprecated)
- `init_catalog.py` - Old notebook version
- `ingestion.py` - Old notebook version (had network restrictions)

## Setup Instructions

### Step 1: Initialize Catalog and Schemas
Run the initialization script to create the Unity Catalog structure:

```sql
-- In Databricks SQL Editor or notebook SQL cell
%run ./init_catalog.sql
```

Or execute directly:
```bash
databricks sql execute --file init_catalog.sql
```

This creates:
- Catalog: `dbx_joshy_demo`
- Schemas: `raw`, `processed`, `features`, `models`, `monitoring`

### Step 2: Upload CoinGecko Data to Volume

#### Option A: Manual Upload via UI
1. Navigate to Catalog Explorer
2. Go to: `dbx_joshy_demo` → `coingecko` → `coingecko` → `coingecko`
3. Upload your JSON files

#### Option B: Upload via Python/Notebook
```python
import json
import requests
from datetime import datetime

# Fetch data from CoinGecko API
response = requests.get(
    "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
    params={"vs_currency": "usd", "days": "1"},
    headers={"x-cg-demo-api-key": "YOUR_API_KEY"}
)

# Save to Volume
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = f"/Volumes/dbx_joshy_demo/coingecko/coingecko/coingecko/market_chart_{timestamp}.json"

with open(file_path, 'w') as f:
    json.dump(response.json(), f)

print(f"✓ Uploaded: {file_path}")
```

#### Option C: Upload via CLI
```bash
# Save API response locally
curl "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1" \
  -H "x-cg-demo-api-key: YOUR_API_KEY" \
  -o market_chart.json

# Upload to Volume (requires databricks CLI)
databricks fs cp market_chart.json \
  dbfs:/Volumes/dbx_joshy_demo/coingecko/coingecko/coingecko/
```

### Step 3: Create Streaming Table
Run the ingestion script to create the streaming table:

```sql
-- In Databricks SQL Editor or notebook SQL cell
%run ./ingestion.sql
```

Or execute directly:
```bash
databricks sql execute --file ingestion.sql
```

This creates a **streaming table** that automatically ingests new JSON files from the Volume.

### Step 4: Verify Data
```sql
-- Check row count
SELECT COUNT(*) as total_records 
FROM dbx_joshy_demo.raw.coingecko;

-- View latest records
SELECT 
  etl_timestamp,
  file_name,
  file_size,
  file_modification_time,
  LEFT(data, 100) as data_preview
FROM dbx_joshy_demo.raw.coingecko
ORDER BY file_modification_time DESC
LIMIT 10;

-- Check unique files
SELECT 
  file_name,
  COUNT(*) as record_count,
  MAX(file_modification_time) as latest_modification
FROM dbx_joshy_demo.raw.coingecko
GROUP BY file_name
ORDER BY latest_modification DESC;
```

## Table Schema

### `dbx_joshy_demo.raw.coingecko`

| Column | Type | Description |
|--------|------|-------------|
| `etl_timestamp` | TIMESTAMP | When the record was ingested |
| `data` | STRING | Raw JSON content from file |
| `file_path` | STRING | Full path to source file |
| `file_name` | STRING | Source file name |
| `file_size` | LONG | File size in bytes |
| `file_modification_time` | TIMESTAMP | When file was last modified |

## Streaming Table Behavior

The streaming table automatically:
- Monitors the Volume path for new files
- Ingests new JSON files as they arrive
- Processes each file exactly once
- Maintains checkpoints for reliability

**To trigger ingestion:**
1. Upload new JSON files to the Volume
2. The streaming table will automatically detect and ingest them

**To refresh/reprocess:**
```sql
-- Refresh the streaming table
REFRESH STREAMING TABLE dbx_joshy_demo.raw.coingecko;
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Deploy Crypto Forecaster

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install Databricks CLI
        run: pip install databricks-cli
      
      - name: Initialize Catalog
        run: databricks sql execute --file init_catalog.sql
      
      - name: Create Streaming Table
        run: databricks sql execute --file ingestion.sql
```

## Advantages Over Notebooks

| Aspect | Notebooks | SQL Scripts |
|--------|-----------|-------------|
| Version Control | Binary format, hard to diff | Plain text, easy to diff |
| CI/CD | Complex | Simple |
| Testing | Limited | Easy to unit test |
| Reusability | Requires %run | Standard SQL import |
| Dependencies | Python packages needed | Pure SQL, no dependencies |
| Execution | Requires compute | Can run anywhere |
| Modularity | Cell-based | File-based |

## Next Steps

1. ✅ **Data Ingestion** (Current stage)
2. **Data Processing** - Parse JSON and extract OHLC data
3. **Feature Engineering** - Create time-series features
4. **Model Training** - Build forecasting model
5. **Monitoring** - Create dashboards

## Troubleshooting

### Issue: Streaming table not ingesting data
**Solution:** Check if files exist in the Volume
```sql
SELECT * FROM read_files('/Volumes/dbx_joshy_demo/coingecko/coingecko/coingecko/');
```

### Issue: Files not appearing in Volume
**Solution:** Verify the Volume exists
```sql
SHOW VOLUMES IN dbx_joshy_demo.coingecko;
```

### Issue: Permission denied
**Solution:** Grant permissions
```sql
GRANT ALL PRIVILEGES ON VOLUME dbx_joshy_demo.coingecko.coingecko 
TO `user@example.com`;
```

## Additional Resources

- [Databricks Streaming Tables](https://docs.databricks.com/structured-streaming/streaming-tables.html)
- [Unity Catalog Volumes](https://docs.databricks.com/data-governance/unity-catalog/volumes.html)
- [read_files() Function](https://docs.databricks.com/sql/language-manual/functions/read_files.html)
