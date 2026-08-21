-- ============================================================
-- Crypto Forecaster - Process Market Chart Data
-- ============================================================
-- Purpose: Parse raw CoinGecko market_chart JSON into structured table
-- Input: raw.coingecko (JSON data from market_chart endpoint)
-- Output: processed.cg_coin_historical_chart_data
-- ============================================================
CREATE OR REPLACE TABLE IDENTIFIER(:catalog || '.processed.cg_coin_historical_chart_data')
COMMENT 'Parsed historical market data (prices, market caps, volumes) from CoinGecko API'
AS
WITH parsed_wrapper AS (
  SELECT
    etl_timestamp,
    file_name,
    file_modification_time,
    from_json(data, 'struct<endpoint:string, parameters:string, response:string>') as wrapper
  FROM dbx_joshdevph_dev.raw.coingecko
  WHERE contains(file_name, 'market_chart') -- Filter for market_chart files only
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
    from_json(wrapper.response, 'struct<prices:array<array<double>>, market_caps:array<array<double>>, total_volumes:array<array<double>>>') as market_data
  FROM parsed_wrapper
  WHERE wrapper.response IS NOT NULL
),
prices_exploded AS (
  SELECT
    etl_timestamp,
    file_name,
    file_modification_time,
    endpoint,
    coin_id,
    vs_currency,
    days,
    CAST(price_record[0] AS BIGINT) as timestamp_unix_ms,
    price_record[1] as price
  FROM parsed_json
  LATERAL VIEW explode(market_data.prices) AS price_record
  WHERE market_data.prices IS NOT NULL
),
market_caps_exploded AS (
  SELECT
    CAST(mc_record[0] AS BIGINT) as timestamp_unix_ms,
    mc_record[1] as market_cap
  FROM parsed_json
  LATERAL VIEW explode(market_data.market_caps) AS mc_record
  WHERE market_data.market_caps IS NOT NULL
),
volumes_exploded AS (
  SELECT
    CAST(vol_record[0] AS BIGINT) as timestamp_unix_ms,
    vol_record[1] as total_volume
  FROM parsed_json
  LATERAL VIEW explode(market_data.total_volumes) AS vol_record
  WHERE market_data.total_volumes IS NOT NULL
)
SELECT
  p.etl_timestamp,
  p.file_name,
  p.file_modification_time,
  p.endpoint,
  p.coin_id,
  p.vs_currency,
  p.days,
  p.timestamp_unix_ms,
  from_unixtime(p.timestamp_unix_ms / 1000) as timestamp,
  p.price,
  mc.market_cap,
  v.total_volume
FROM prices_exploded p
LEFT JOIN market_caps_exploded mc 
  ON p.timestamp_unix_ms = mc.timestamp_unix_ms
LEFT JOIN volumes_exploded v 
  ON p.timestamp_unix_ms = v.timestamp_unix_ms
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY 
    p.coin_id,
    p.vs_currency,
    p.timestamp_unix_ms,
    p.price  -- Only deduplicate when timestamp AND price are identical
  ORDER BY p.etl_timestamp ASC  -- Keep first occurrence when all data values match
) = 1;

