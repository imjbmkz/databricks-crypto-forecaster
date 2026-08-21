from typing import Dict, Tuple

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


# ============================================================
# Filter data for a specific coin
# ============================================================

def filter_coin_data(
    df: DataFrame,
    coin_id: str,
    vs_currency: str = "usd",
) -> DataFrame:
    """
    Filter DataFrame for a specific coin and currency.
    """

    return (
        df
        .filter(
            (F.col("coin_id") == coin_id)
            & (F.col("vs_currency") == vs_currency)
        )
    )


# ============================================================
# Train Linear Regression model
# ============================================================

def train_linear_regression(
    train_df: DataFrame,
    features_col: str = "features",
    label_col: str = "price",
) -> LinearRegressionModel:
    """
    Train a Linear Regression model.
    """

    estimator = LinearRegression(
        featuresCol=features_col,
        labelCol=label_col,
        predictionCol="prediction",
        maxIter=100,
        regParam=0.0,
        elasticNetParam=0.0,
    )

    return estimator.fit(train_df)


# ============================================================
# Generate predictions
# ============================================================

def generate_predictions(
    model: LinearRegressionModel,
    df: DataFrame,
    dataset_type: str,
) -> DataFrame:
    """
    Generate predictions and identify whether rows
    belong to the train or test dataset.
    """

    return (
        model
        .transform(df)
        .withColumn(
            "dataset_type",
            F.lit(dataset_type),
        )
        .withColumn(
            "residual",
            F.col("price") - F.col("prediction"),
        )
    )


# ============================================================
# Calculate residual baseline statistics
#
# IMPORTANT:
# Statistics should be calculated from TRAINING residuals only.
# ============================================================

def calculate_residual_stats(
    train_predictions: DataFrame,
) -> DataFrame:
    """
    Calculate residual control-chart baseline statistics
    for each coin/currency.

    Control limits:
        Center Line = mean residual
        +/- 1 sigma
        +/- 2 sigma
        +/- 3 sigma (UCL/LCL)
    """

    residual_stats = (
        train_predictions
        .groupBy(
            "coin_id",
            "vs_currency",
        )
        .agg(
            F.avg("residual").alias(
                "residual_mean"
            ),
            F.stddev_samp("residual").alias(
                "residual_std"
            ),
        )
        .withColumn(
            "upper_1_sigma",
            F.col("residual_mean")
            + F.col("residual_std"),
        )
        .withColumn(
            "lower_1_sigma",
            F.col("residual_mean")
            - F.col("residual_std"),
        )
        .withColumn(
            "upper_2_sigma",
            F.col("residual_mean")
            + (F.lit(2) * F.col("residual_std")),
        )
        .withColumn(
            "lower_2_sigma",
            F.col("residual_mean")
            - (F.lit(2) * F.col("residual_std")),
        )
        .withColumn(
            "ucl",
            F.col("residual_mean")
            + (F.lit(3) * F.col("residual_std")),
        )
        .withColumn(
            "lcl",
            F.col("residual_mean")
            - (F.lit(3) * F.col("residual_std")),
        )
    )

    return residual_stats


# ============================================================
# Attach residual statistics to forecasts
# ============================================================

def add_residual_stats(
    forecasts: DataFrame,
    residual_stats: DataFrame,
) -> DataFrame:
    """
    Attach training residual statistics and control limits
    to train/test forecasts.
    """

    return forecasts.join(
        residual_stats,
        on=[
            "coin_id",
            "vs_currency",
        ],
        how="left",
    )


# ============================================================
# Evaluate predictions
# ============================================================

def evaluate_predictions(
    predictions: DataFrame,
    label_col: str = "price",
    prediction_col: str = "prediction",
) -> Dict[str, float]:
    """
    Calculate regression evaluation metrics.
    """

    metrics = {}

    for metric_name in [
        "rmse",
        "mae",
        "r2",
    ]:

        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName=metric_name,
        )

        metrics[metric_name] = (
            evaluator.evaluate(predictions)
        )

    return metrics


# ============================================================
# Save trained model
# ============================================================

def save_model(
    model: LinearRegressionModel,
    model_path: str,
) -> None:
    """
    Save trained Spark ML model.
    """

    (
        model.write()
        .overwrite()
        .save(model_path)
    )


# ============================================================
# Save DataFrame to Unity Catalog
# ============================================================

def save_to_catalog(
    df: DataFrame,
    table_name: str,
) -> None:
    """
    Save DataFrame as Delta table.
    """

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true",
        )
        .saveAsTable(table_name)
    )


# ============================================================
# Train and evaluate one coin
# ============================================================

def train_coin_model(
    train_df: DataFrame,
    test_df: DataFrame,
    coin_id: str,
    vs_currency: str = "usd",
) -> Tuple[
    LinearRegressionModel,
    DataFrame,
    Dict[str, float],
]:
    """
    Train one model for a coin and generate forecasts
    for both train and test datasets.
    """

    # --------------------------------------------------------
    # Filter data
    # --------------------------------------------------------

    train_coin_df = filter_coin_data(
        df=train_df,
        coin_id=coin_id,
        vs_currency=vs_currency,
    )

    test_coin_df = filter_coin_data(
        df=test_df,
        coin_id=coin_id,
        vs_currency=vs_currency,
    )


    # --------------------------------------------------------
    # Validate data
    # --------------------------------------------------------

    if train_coin_df.limit(1).count() == 0:
        raise ValueError(
            f"No training data found for "
            f"{coin_id}/{vs_currency}"
        )

    if test_coin_df.limit(1).count() == 0:
        raise ValueError(
            f"No test data found for "
            f"{coin_id}/{vs_currency}"
        )


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model = train_linear_regression(
        train_df=train_coin_df,
    )


    # --------------------------------------------------------
    # Generate training forecasts
    # --------------------------------------------------------

    train_predictions = generate_predictions(
        model=model,
        df=train_coin_df,
        dataset_type="train",
    )


    # --------------------------------------------------------
    # Generate test forecasts
    # --------------------------------------------------------

    test_predictions = generate_predictions(
        model=model,
        df=test_coin_df,
        dataset_type="test",
    )


    # --------------------------------------------------------
    # Calculate residual statistics using TRAIN only
    # --------------------------------------------------------

    residual_stats = calculate_residual_stats(
        train_predictions=train_predictions,
    )


    # --------------------------------------------------------
    # Combine train and test forecasts
    # --------------------------------------------------------

    forecasts = (
        train_predictions
        .unionByName(test_predictions)
    )


    # --------------------------------------------------------
    # Attach training residual statistics
    # --------------------------------------------------------

    forecasts = add_residual_stats(
        forecasts=forecasts,
        residual_stats=residual_stats,
    )


    # --------------------------------------------------------
    # Evaluate using TEST data
    # --------------------------------------------------------

    metrics = evaluate_predictions(
        predictions=test_predictions,
    )


    # --------------------------------------------------------
    # Select useful forecast columns
    # --------------------------------------------------------

    forecasts = forecasts.select(
        "coin_id",
        "timestamp",
        "vs_currency",
        "dataset_type",
        "price",
        "prediction",
        "residual",
        "residual_mean",
        "residual_std",
        "lower_1_sigma",
        "upper_1_sigma",
        "lower_2_sigma",
        "upper_2_sigma",
        "lcl",
        "ucl",
    )

    return model, forecasts, metrics