# Tests

## Purpose
Unit and integration tests for the crypto forecaster project.

## Test Structure

```
tests/
├── data_pipeline/           # Data pipeline tests
│   ├── test_ingestion.py
│   ├── test_ohlc_parsing.py
│   └── test_data_quality.py
│
├── feature_engineering/     # Feature engineering tests
│   ├── test_time_series_features.py
│   ├── test_technical_indicators.py
│   └── test_feature_values.py
│
├── model_training/          # Model training tests
│   ├── test_train_prophet.py
│   ├── test_model_serving.py
│   └── test_predictions.py
│
└── integration/             # End-to-end tests
    └── test_full_pipeline.py
```

## Running Tests

### Using pytest (locally)
```bash
pytest tests/
```

### Via DAB
```bash
databricks bundle run test_suite
```

### In CI/CD
Tests run automatically on:
- Pull requests
- Before deployment
- Scheduled daily runs

## Test Categories

### Unit Tests
Test individual functions and transformations:
- SQL query logic
- Feature calculations
- Model predictions

### Integration Tests
Test component interactions:
- Data flow from raw → processed
- Feature pipeline
- Model training pipeline

### Data Quality Tests
Validate data assumptions:
- Schema compliance
- Value ranges
- Completeness checks
- Duplicate detection

## Best Practices

1. **Test Coverage**: Aim for >80% coverage
2. **Fast Tests**: Unit tests < 1 second
3. **Isolated**: Each test independent
4. **Clear Names**: `test_<what>_<condition>_<expected>`
5. **Fixtures**: Use pytest fixtures for common setup

## Example Test

```python
import pytest
from pyspark.sql import SparkSession

@pytest.fixture
def spark():
    return SparkSession.builder.getOrCreate()

def test_ohlc_parsing_extracts_prices(spark):
    # Given: Raw JSON data
    raw_data = spark.createDataFrame([
        (1, '{"prices": [[1234567890000, 50000]]}')
    ], ["id", "data"])
    
    # When: Parsing OHLC
    result = parse_ohlc(raw_data)
    
    # Then: Price extracted correctly
    assert result.count() == 1
    assert result.first()["price"] == 50000
```
