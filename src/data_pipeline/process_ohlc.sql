-- ============================================================
-- Crypto Forecaster - Process OHLC Data
-- ============================================================
-- Purpose: Parse raw CoinGecko OHLC JSON into structured table
-- Input: raw.coingecko (JSON data from OHLC endpoint)
-- Output: processed.cg_coin_ohlc_chart
-- ============================================================

CREATE OR REPLACE TABLE dbx_joshy_demo.processed.cg_coin_ohlc_chart
COMMENT 'Parsed OHLC (Open, High, Low, Close) data from CoinGecko API'
AS
WITH parsed_wrapper AS (
  SELECT
    etl_timestamp,
    file_name,
    file_modification_time,
    from_json(data, 'struct<endpoint:string, parameters:string, response:string>') as wrapper
  FROM dbx_joshy_demo.raw.coingecko
  WHERE file_name LIKE '%ohlc%'  -- Filter for OHLC files only
    AND data IS NOT NULL
),
parsed_json AS (
  SELECT
    etl_timestamp,
    file_name,
    file_modification_time,
    wrapper.endpoint as endpoint,
    -- Extract coin_id from endpoint (value after second slash, before third slash)
    split(wrapper.endpoint, '/')[2] as coin_id,
    -- Parse parameters JSON to extract vs_currency and days
    get_json_object(wrapper.parameters, '$.vs_currency') as vs_currency,
    get_json_object(wrapper.parameters, '$.days') as days,
    from_json(wrapper.response, 'array<array<double>>') as ohlc_array
  FROM parsed_wrapper
  WHERE wrapper.response IS NOT NULL
),
exploded_data AS (
  SELECT
    etl_timestamp,
    file_name,
    file_modification_time,
    endpoint,
    coin_id,
    vs_currency,
    days,
    explode(ohlc_array) as ohlc_record
  FROM parsed_json
  WHERE ohlc_array IS NOT NULL
)
SELECT
  etl_timestamp,
  file_name,
  file_modification_time,
  endpoint,
  coin_id,
  vs_currency,
  days,
  CAST(ohlc_record[0] AS BIGINT) as timestamp_unix_ms,
  from_unixtime(ohlc_record[0] / 1000) as timestamp,
  ohlc_record[1] as open_price,
  ohlc_record[2] as high_price,
  ohlc_record[3] as low_price,
  ohlc_record[4] as close_price
FROM exploded_data
WHERE size(ohlc_record) >= 5  -- Ensure valid OHLC records
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY 
    coin_id, 
    vs_currency, 
    CAST(ohlc_record[0] AS BIGINT),  -- timestamp
    ohlc_record[1],  -- open_price
    ohlc_record[2],  -- high_price
    ohlc_record[3],  -- low_price
    ohlc_record[4]   -- close_price
  ORDER BY etl_timestamp ASC  -- Keep first occurrence when all data values are identical
) = 1;

-- Verification (commented - uncomment to run)
-- SELECT 
--   COUNT(*) as total_records,
--   MIN(timestamp) as earliest_timestamp,
--   MAX(timestamp) as latest_timestamp,
--   COUNT(DISTINCT DATE(timestamp)) as unique_days
-- FROM dbx_joshy_demo.processed.cg_coin_ohlc_chart;
