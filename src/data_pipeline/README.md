# Data Pipeline

## Purpose
SQL scripts for data ingestion, ETL, and raw data processing.

## Pipeline Flow

```
Volume (JSON files)
    ↓
[ingestion.sql]
    ↓
raw.coingecko (Raw JSON)
    ↓
    ├─→ [process_ohlc.sql] ──→ processed.cg_coin_ohlc_chart
    └─→ [process_market_chart.sql] ──→ processed.cg_coin_historical_chart_data
```

## Scripts

### 1. `ingestion.sql` ✓
**Purpose**: Ingest CoinGecko JSON data from Unity Catalog Volume

**Input**: JSON files in `/Volumes/dbx_joshy_demo/coingecko/coingecko/coingecko/`

**Output**: `dbx_joshy_demo.raw.coingecko`

**Schema**:
- `etl_timestamp` - Ingestion timestamp
- `data` - Raw JSON content
- `file_path` - Source file path
- `file_name` - Source file name
- `file_size` - File size in bytes
- `file_modification_time` - File modification timestamp

**Run**:
```sql
%run ./src/data_pipeline/ingestion.sql
```

---

### 2. `process_ohlc.sql` ✓
**Purpose**: Parse OHLC (Open, High, Low, Close) data from raw JSON

**Input**: `raw.coingecko` (filters for `*ohlc*` files)

**Output**: `processed.cg_coin_ohlc_chart`

**Schema**:
- `etl_timestamp` - When raw data was ingested
- `file_name` - Source file name
- `file_modification_time` - File modification time
- `timestamp_unix_ms` - Unix timestamp in milliseconds (original)
- `timestamp` - Converted timestamp
- `open_price` - Opening price
- `high_price` - Highest price
- `low_price` - Lowest price
- `close_price` - Closing price

**Transformation**:
- Parses JSON array: `[[timestamp, open, high, low, close], ...]`
- Keeps Unix timestamp (ms) AND adds converted timestamp
- Explodes array into individual records
- Filters for valid OHLC records (5 elements)

**Run**:
```sql
%run ./src/data_pipeline/process_ohlc.sql
```

---

### 3. `process_market_chart.sql` ✓
**Purpose**: Parse market chart data (prices, market caps, volumes) from raw JSON

**Input**: `raw.coingecko` (filters for `*market_chart*` files)

**Output**: `processed.cg_coin_historical_chart_data`

**Schema**:
- `etl_timestamp` - When raw data was ingested
- `file_name` - Source file name
- `file_modification_time` - File modification time
- `timestamp_unix_ms` - Unix timestamp in milliseconds (original)
- `timestamp` - Converted timestamp
- `price` - Cryptocurrency price
- `market_cap` - Total market capitalization
- `total_volume` - Trading volume

**Transformation**:
- Parses JSON: `{"prices": [[ts, price]], "market_caps": [[ts, mc]], "total_volumes": [[ts, vol]]}`
- Explodes each array independently
- Joins all three on `timestamp_unix_ms`
- Keeps Unix timestamp (ms) AND adds converted timestamp

**Run**:
```sql
%run ./src/data_pipeline/process_market_chart.sql
```

---

## Data Quality Features

### OHLC Processing
- ✓ Filters for OHLC-specific files by name pattern
- ✓ Validates record structure (requires 5 elements)
- ✓ Handles NULL data gracefully
- ✓ Includes verification query with stats

### Market Chart Processing
- ✓ Filters for market_chart-specific files by name pattern
- ✓ Joins prices, market caps, and volumes on timestamp
- ✓ LEFT JOINs preserve price records even if caps/volumes missing
- ✓ Includes verification query with completeness stats

## Running the Full Pipeline

```sql
-- Step 1: Ingest raw data from Volume
%run ./src/data_pipeline/ingestion.sql

-- Step 2: Process OHLC data
%run ./src/data_pipeline/process_ohlc.sql

-- Step 3: Process market chart data
%run ./src/data_pipeline/process_market_chart.sql
```

## Timestamp Handling

Both processing scripts handle timestamps by:
1. **Preserving original**: `timestamp_unix_ms` (BIGINT) - Unix timestamp in milliseconds
2. **Adding converted**: `timestamp` (TIMESTAMP) - Human-readable timestamp via `from_unixtime(unix_ms / 1000)`

This allows:
- Exact timestamp matching between tables
- Easy time-based queries and filtering
- Human-readable display
- Compatibility with time-series models

## Verification Queries

### Check OHLC Data
```sql
SELECT 
  COUNT(*) as total_records,
  MIN(timestamp) as earliest,
  MAX(timestamp) as latest,
  COUNT(DISTINCT DATE(timestamp)) as unique_days
FROM dbx_joshy_demo.processed.cg_coin_ohlc_chart;
```

### Check Market Chart Data
```sql
SELECT 
  COUNT(*) as total_records,
  MIN(timestamp) as earliest,
  MAX(timestamp) as latest,
  SUM(CASE WHEN price IS NOT NULL THEN 1 ELSE 0 END) as with_price,
  SUM(CASE WHEN market_cap IS NOT NULL THEN 1 ELSE 0 END) as with_market_cap,
  SUM(CASE WHEN total_volume IS NOT NULL THEN 1 ELSE 0 END) as with_volume
FROM dbx_joshy_demo.processed.cg_coin_historical_chart_data;
```

### Compare Data Coverage
```sql
-- OHLC vs Market Chart timestamps
SELECT 
  'OHLC' as source,
  COUNT(*) as records,
  MIN(timestamp) as earliest,
  MAX(timestamp) as latest
FROM dbx_joshy_demo.processed.cg_coin_ohlc_chart
UNION ALL
SELECT 
  'Market Chart' as source,
  COUNT(*) as records,
  MIN(timestamp) as earliest,
  MAX(timestamp) as latest
FROM dbx_joshy_demo.processed.cg_coin_historical_chart_data
ORDER BY source;
```

## Upcoming Scripts

### `validate_processed_data.sql`
Data quality checks:
- Duplicate detection
- Missing value analysis
- Price anomaly detection
- Timestamp continuity checks

### `deduplicate.sql`
Remove duplicates based on:
- Timestamp + endpoint combination
- Latest file_modification_time wins
