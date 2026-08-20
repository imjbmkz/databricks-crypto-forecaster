# Model Training

## Purpose
Scripts for training, evaluating, and deploying ML forecasting models.

## Upcoming Scripts

### `prepare_training_data.sql`
Create train/validation/test splits

**Input**: `features.*` tables

**Output**: 
- `models.training_data`
- `models.validation_data`
- `models.test_data`

### `train_prophet_model.py`
Time series forecasting with Prophet

**Features**:
- Handles seasonality and trends
- Supports holidays and special events
- Provides uncertainty intervals

### `train_lstm_model.py`
Deep learning LSTM model for sequential data

**Features**:
- Captures long-term dependencies
- Multiple input features
- Multi-step forecasting

### `train_xgboost_model.py`
Gradient boosting with engineered features

**Features**:
- Fast training
- Feature importance
- Handles non-linear relationships

### `evaluate_models.py`
Compare model performance

**Metrics**:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Direction accuracy

### `register_model.py`
Register best model to MLflow Model Registry

```python
import mlflow

mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="crypto_forecaster"
)
```

### `deploy_model.py`
Deploy to model serving endpoint

## MLflow Integration

All model training uses MLflow for:
- Experiment tracking
- Model versioning
- Model registry
- Model serving

## Testing
Add tests in `/tests/model_training/` for:
- Model training pipeline
- Prediction format
- Model performance thresholds
