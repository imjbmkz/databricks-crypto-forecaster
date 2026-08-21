import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--catalog",
    required=True,
)

args = parser.parse_args()
catalog = args.catalog

from typing import Sequence

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ============================================================
# Configuration
# ============================================================

SOURCE_TABLE = f"{catalog}.processed.cg_coin_historical_chart_data"

TRAIN_TABLE = f"{catalog}.features.cg_coin_forecast_features_train"
TEST_TABLE = f"{catalog}.features.cg_coin_forecast_features_test"

LAG_PERIODS = (1, 2, 3, 6, 12)

GROUP_COLS = ("coin_id", "vs_currency")

TIMESTAMP_COL = "timestamp"

VALUE_COL = "price"

TRAIN_RATIO = 0.80

# ============================================================
# 1. Load source data
# ============================================================

def load_source_data(
    spark: SparkSession,
    source_table: str,
) -> DataFrame:

    return (
        spark.table(source_table)
        .select(
            "coin_id",
            "timestamp",
            "vs_currency",
            "price",
        )
    )


# ============================================================
# 2. Add lag features
# ============================================================

def add_lag_features(
    df: DataFrame,
    lag_periods: Sequence[int],
) -> DataFrame:

    window_spec = (
        Window
        .partitionBy(*GROUP_COLS)
        .orderBy(TIMESTAMP_COL)
    )

    result = df

    for lag in lag_periods:

        result = result.withColumn(
            f"price_lag_{lag}",
            F.lag(VALUE_COL, lag).over(window_spec),
        )

    return result


# ============================================================
# 3. Remove rows with null lag features
# ============================================================

def remove_null_lag_rows(
    df: DataFrame,
    feature_cols: Sequence[str],
) -> DataFrame:

    return df.dropna(
        subset=list(feature_cols)
    )


# ============================================================
# 4. Chronological train/test split
#
# First 80%  = train
# Latest 20% = test
#
# Split is performed separately for every:
# coin_id + vs_currency
# ============================================================

def chronological_train_test_split(
    df: DataFrame,
    train_ratio: float = 0.80,
) -> tuple[DataFrame, DataFrame]:

    order_window = (
        Window
        .partitionBy(*GROUP_COLS)
        .orderBy(TIMESTAMP_COL)
    )

    group_window = (
        Window
        .partitionBy(*GROUP_COLS)
    )

    df_split = (
        df
        .withColumn(
            "_row_number",
            F.row_number().over(order_window),
        )
        .withColumn(
            "_total_rows",
            F.count("*").over(group_window),
        )
        .withColumn(
            "_train_rows",
            F.floor(
                F.col("_total_rows") * F.lit(train_ratio)
            ).cast("long"),
        )
    )

    train_df = (
        df_split
        .filter(
            F.col("_row_number") <= F.col("_train_rows")
        )
        .drop(
            "_row_number",
            "_total_rows",
            "_train_rows",
        )
    )

    test_df = (
        df_split
        .filter(
            F.col("_row_number") > F.col("_train_rows")
        )
        .drop(
            "_row_number",
            "_total_rows",
            "_train_rows",
        )
    )

    return train_df, test_df


# ============================================================
# 5. Calculate scaling statistics
#
# IMPORTANT:
# This function must be called using TRAINING DATA only.
# ============================================================

def calculate_scaling_stats(
    df: DataFrame,
    feature_cols: Sequence[str],
) -> DataFrame:

    aggregations = []

    for col_name in feature_cols:

        aggregations.extend(
            [
                F.avg(col_name).alias(
                    f"{col_name}_mean"
                ),
                F.stddev_samp(col_name).alias(
                    f"{col_name}_std"
                ),
            ]
        )

    return (
        df
        .groupBy(*GROUP_COLS)
        .agg(*aggregations)
    )


# ============================================================
# 6. Apply standardization
#
# z = (x - training_mean) / training_std
#
# The SAME training statistics are used for:
# - train data
# - test data
# ============================================================

def apply_standardization(
    df: DataFrame,
    scaling_stats: DataFrame,
    feature_cols: Sequence[str],
) -> tuple[DataFrame, list[str]]:

    result = df.join(
        scaling_stats,
        on=list(GROUP_COLS),
        how="left",
    )

    scaled_feature_cols = []

    for col_name in feature_cols:

        scaled_col = f"{col_name}_scaled"

        result = result.withColumn(
            scaled_col,
            F.when(
                F.col(f"{col_name}_std").isNull()
                | (F.col(f"{col_name}_std") == 0),
                F.lit(0.0),
            ).otherwise(
                (
                    F.col(col_name)
                    - F.col(f"{col_name}_mean")
                )
                / F.col(f"{col_name}_std")
            ),
        )

        scaled_feature_cols.append(
            scaled_col
        )

    return result, scaled_feature_cols


# ============================================================
# 7. Assemble features into Spark ML vector
# ============================================================

def assemble_features(
    df: DataFrame,
    scaled_feature_cols: Sequence[str],
) -> DataFrame:

    assembler = VectorAssembler(
        inputCols=list(scaled_feature_cols),
        outputCol="features",
    )

    return assembler.transform(df)


# ============================================================
# 8. Select final model columns
# ============================================================

def select_final_columns(
    df: DataFrame,
    feature_cols: Sequence[str],
    scaled_feature_cols: Sequence[str],
) -> DataFrame:

    return df.select(
        "coin_id",
        "timestamp",
        "vs_currency",
        "price",
        *feature_cols,
        *scaled_feature_cols,
        "features",
    )


# ============================================================
# 9. Build train and test feature DataFrames
# ============================================================

def build_feature_dataframes(
    spark: SparkSession,
) -> tuple[DataFrame, DataFrame]:

    feature_cols = [
        f"price_lag_{lag}"
        for lag in LAG_PERIODS
    ]

    # --------------------------------------------------------
    # Load source
    # --------------------------------------------------------

    df = load_source_data(
        spark=spark,
        source_table=SOURCE_TABLE,
    )

    # --------------------------------------------------------
    # Create lag features
    #
    # Lag features are created BEFORE the split so that the
    # first test observation can correctly use historical
    # values from the training period.
    # --------------------------------------------------------

    df = add_lag_features(
        df=df,
        lag_periods=LAG_PERIODS,
    )

    # --------------------------------------------------------
    # Remove rows without complete lag history
    # --------------------------------------------------------

    df = remove_null_lag_rows(
        df=df,
        feature_cols=feature_cols,
    )

    # --------------------------------------------------------
    # Chronological 80/20 split
    # --------------------------------------------------------

    train_df, test_df = chronological_train_test_split(
        df=df,
        train_ratio=TRAIN_RATIO,
    )

    # --------------------------------------------------------
    # Calculate scaling statistics using TRAIN only
    # --------------------------------------------------------

    scaling_stats = calculate_scaling_stats(
        df=train_df,
        feature_cols=feature_cols,
    )

    # --------------------------------------------------------
    # Scale training data
    # --------------------------------------------------------

    train_df, scaled_feature_cols = apply_standardization(
        df=train_df,
        scaling_stats=scaling_stats,
        feature_cols=feature_cols,
    )

    # --------------------------------------------------------
    # Scale test data using SAME training statistics
    # --------------------------------------------------------

    test_df, _ = apply_standardization(
        df=test_df,
        scaling_stats=scaling_stats,
        feature_cols=feature_cols,
    )

    # --------------------------------------------------------
    # VectorAssembler
    # --------------------------------------------------------

    train_df = assemble_features(
        df=train_df,
        scaled_feature_cols=scaled_feature_cols,
    )

    test_df = assemble_features(
        df=test_df,
        scaled_feature_cols=scaled_feature_cols,
    )

    # --------------------------------------------------------
    # Select final columns
    # --------------------------------------------------------

    train_df = select_final_columns(
        df=train_df,
        feature_cols=feature_cols,
        scaled_feature_cols=scaled_feature_cols,
    )

    test_df = select_final_columns(
        df=test_df,
        feature_cols=feature_cols,
        scaled_feature_cols=scaled_feature_cols,
    )

    return train_df, test_df


# ============================================================
# 10. Save DataFrame to Unity Catalog
# ============================================================

def save_to_catalog(
    df: DataFrame,
    table_name: str,
) -> None:

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