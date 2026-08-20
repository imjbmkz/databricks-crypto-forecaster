-- ============================================================
-- Crypto Forecaster - Data Ingestion Script (SQL)
-- ============================================================
-- Purpose: Ingest CoinGecko JSON data from Unity Catalog Volume
-- Creates a streaming table that continuously ingests new files
-- ============================================================

-- Create Table for CoinGecko Raw Data
-- Note: Using regular table instead of streaming table due to serverless compute limitations
-- Run this script again to refresh data when new files are added to the volume
CREATE OR REPLACE TABLE dbx_joshy_demo.raw.coingecko
COMMENT 'CoinGecko API responses from Volume (refresh manually or via schedule)'
AS SELECT
  current_timestamp() as etl_timestamp,
  value as data,
  _metadata.file_path,
  _metadata.file_name,
  _metadata.file_size,
  _metadata.file_modification_time
FROM read_files(
  '/Volumes/dbx_joshy_demo/coingecko/coingecko/coingecko/',
  format => 'text'
);

-- Verification Query
-- Run this after the streaming table is created to verify data
-- SELECT 
--   etl_timestamp,
--   file_name,
--   file_size,
--   file_modification_time,
--   LEFT(data, 100) as data_preview
-- FROM dbx_joshy_demo.raw.coingecko
-- ORDER BY file_modification_time DESC
-- LIMIT 10;
