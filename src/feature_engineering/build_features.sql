CREATE OR REPLACE TABLE dbx_joshy_demo.features.cg_features
WITH features AS (
        SELECT 
            coin_id,
            timestamp,
            vs_currency,
            price,
            -- Lag features: 5min, 10min, 15min, 30min, 60min
            LAG(price, 1) OVER (PARTITION BY coin_id, vs_currency ORDER BY timestamp) AS price_lag_1,
            LAG(price, 2) OVER (PARTITION BY coin_id, vs_currency ORDER BY timestamp) AS price_lag_2,
            LAG(price, 3) OVER (PARTITION BY coin_id, vs_currency ORDER BY timestamp) AS price_lag_3,
            LAG(price, 6) OVER (PARTITION BY coin_id, vs_currency ORDER BY timestamp) AS price_lag_6,
            LAG(price, 12) OVER (PARTITION BY coin_id, vs_currency ORDER BY timestamp) AS price_lag_12
        FROM dbx_joshy_demo.processed.cg_coin_historical_chart_data
    ),
    features_clean AS (
        SELECT *
        FROM features
        WHERE price_lag_1 IS NOT NULL
          AND price_lag_2 IS NOT NULL
          AND price_lag_3 IS NOT NULL
          AND price_lag_6 IS NOT NULL
          AND price_lag_12 IS NOT NULL
    )

    SELECT 
        f.*
    FROM features_clean f
    ORDER BY coin_id, vs_currency, timestamp