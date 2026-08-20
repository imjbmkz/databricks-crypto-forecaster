-- ============================================================
-- Crypto Forecaster - Initialization Script (SQL)
-- ============================================================
-- Purpose: Initialize Unity Catalog infrastructure
-- Creates catalog and schemas for the crypto forecaster project
-- ============================================================

-- Create Catalog
CREATE CATALOG IF NOT EXISTS dbx_joshy_demo
COMMENT 'Catalog for crypto forecaster project - stores all data and ML artifacts';

-- Create Raw Schema
CREATE SCHEMA IF NOT EXISTS dbx_joshy_demo.raw
COMMENT 'Raw data from CoinGecko API - market charts, OHLC data, and historical prices';

-- Create Processed Schema
CREATE SCHEMA IF NOT EXISTS dbx_joshy_demo.processed
COMMENT 'Cleaned and validated data with standardized timestamps and formats';

-- Create Features Schema
CREATE SCHEMA IF NOT EXISTS dbx_joshy_demo.features
COMMENT 'Feature-engineered datasets ready for machine learning models';

-- Create Models Schema
CREATE SCHEMA IF NOT EXISTS dbx_joshy_demo.models
COMMENT 'Trained ML models, predictions, and model metadata';

-- Create Monitoring Schema
CREATE SCHEMA IF NOT EXISTS dbx_joshy_demo.monitoring
COMMENT 'Dashboard metrics, KPIs, and performance tracking data';

-- Verification
SHOW CATALOGS LIKE 'dbx_joshy_demo';
SHOW SCHEMAS IN dbx_joshy_demo;
