# Feature Engineering Package

## Overview

This package provides univariate time series feature engineering for cryptocurrency price forecasting. It creates lag features and rolling statistics from price data to enable machine learning models to capture temporal patterns.

## Package Structure

```
feature_engineering/
├── __init__.py              # Package initialization
├── config.py                # Feature configuration
├── univariate_features.py   # Core feature engineering logic
├── utils.py                 # Utility functions
├── example.py               # Usage example
└── README.md               # This file
```

## Quick Start

### Basic Usage

```python
# Import the package
import sys
sys.path.insert(0, '/Workspace/Repos/josh.valdeleon@outlook.com/databricks-crypto-forecaster/src')

from feature_engineering import UnivariateFeatureEngineer
import pandas as pd

# Load your price data (must have 'price' column and datetime index)
df = spark.sql("""
    SELECT timestamp, price
    FROM dbx_joshy_demo.processed.cg_coin_historical_chart_data
    WHERE coin_id = 'bitcoin'
    ORDER BY timestamp
""").toPandas()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Create feature engineer with default configuration
engineer = UnivariateFeatureEngineer()

# Generate features
df_features = engineer.fit_transform(df)

print(f"Original columns: {len(df.columns)}")
print(f"With features: {len(df_features.columns)}")
print(f"Features created: {len(engineer.feature_names_)}")
```

### Custom Configuration

```python
from feature_engineering import UnivariateFeatureEngineer, FeatureConfig

# Create custom configuration
custom_config = FeatureConfig(
    price_lags=[1, 2, 3, 6, 12],      # Shorter lags
    rolling_windows=[12, 24, 48],      # Fewer windows
    target_column='price'
)

# Use custom config
engineer = UnivariateFeatureEngineer(config=custom_config)
df_features = engineer.fit_transform(df)
```

## Features Created

### Default Configuration

The default configuration creates **24 features** from price data:

#### Price Lag Features (7 features)
* `price_lag_1` - Price 5 minutes ago
* `price_lag_2` - Price 10 minutes ago
* `price_lag_3` - Price 15 minutes ago
* `price_lag_6` - Price 30 minutes ago
* `price_lag_12` - Price 1 hour ago
* `price_lag_24` - Price 2 hours ago
* `price_lag_48` - Price 4 hours ago

#### Rolling Statistics (16 features)

For each window size (12, 24, 48, 96 intervals = 1hr, 2hr, 4hr, 8hr):
* `price_rolling_mean_{window}` - Rolling average
* `price_rolling_std_{window}` - Rolling standard deviation
* `price_rolling_min_{window}` - Rolling minimum
* `price_rolling_max_{window}` - Rolling maximum

## Complete Example Workflow

```python
import pandas as pd
from feature_engineering import UnivariateFeatureEngineer
from feature_engineering.utils import (
    train_test_split_temporal,
    prepare_features_for_ml,
    remove_nan_rows,
    validate_time_series
)

# 1. Load data
df = spark.sql("""
    SELECT timestamp, price
    FROM dbx_joshy_demo.processed.cg_coin_historical_chart_data
    WHERE coin_id = 'bitcoin'
    ORDER BY timestamp
""").toPandas()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# 2. Validate data
validation = validate_time_series(df, expected_freq='5min')
if not validation['is_valid']:
    print("Data quality issues:", validation['issues'])

# 3. Create features
engineer = UnivariateFeatureEngineer()
df_features = engineer.fit_transform(df)

# 4. Split into train/test (temporal split)
train_df, test_df = train_test_split_temporal(df_features, test_size=0.2)

# 5. Remove NaN rows (from lag/rolling features)
train_clean, stats = remove_nan_rows(train_df)
print(f"Removed {stats['rows_removed']} rows with NaN values")

# 6. Prepare features for ML
X_train, y_train = prepare_features_for_ml(train_clean, target_col='price')
X_test, y_test = prepare_features_for_ml(test_df.dropna(), target_col='price')

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {len(X_train.columns)}")
```

## Integration with ML Models

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)
```

## API Reference

### UnivariateFeatureEngineer

Main class for feature engineering.

**Methods:**
* `__init__(config=None)` - Initialize with optional FeatureConfig
* `fit(df)` - Learn feature names from data
* `transform(df)` - Create features on fitted data
* `fit_transform(df)` - Fit and transform in one step
* `get_feature_info()` - Get DataFrame with feature descriptions

**Attributes:**
* `config` - FeatureConfig object
* `feature_names_` - List of created feature names (after fit)

### FeatureConfig

Configuration for feature parameters.

**Attributes:**
* `price_lags` - List of lag intervals (default: [1, 2, 3, 6, 12, 24, 48])
* `rolling_windows` - List of window sizes (default: [12, 24, 48, 96])
* `target_column` - Name of price column (default: 'price')

### Utility Functions

* `train_test_split_temporal(df, test_size=0.2)` - Time-aware train/test split
* `prepare_features_for_ml(df, target_col='price')` - Separate features from target
* `remove_nan_rows(df)` - Remove NaN rows and return statistics
* `validate_time_series(df, expected_freq='5min')` - Validate data quality

## Best Practices

1. **Always use temporal splits** - Use `train_test_split_temporal()` to avoid data leakage
2. **Handle NaN values** - Lag and rolling features create NaN values at the beginning
3. **Feature scaling** - Scale features before training ML models
4. **Validate data** - Use `validate_time_series()` before feature engineering

## Running the Example

```bash
# From the repo root
python src/feature_engineering/example.py
```

## Requirements

* pandas >= 1.3.0
* numpy >= 1.21.0

## Version History

* **1.0.0** (2026-08-20)
  - Initial release
  - Univariate feature engineering
  - Configuration support
  - Utility functions
