# Databricks notebook source
# Build the Bitcoin feature table used by the Spark ML experiments.
# This Python implementation complements (and does not replace) build_features.sql.

# COMMAND ----------

# DBTITLE 1,Configuration and imports
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import Window, functions as F

CATALOG = "dbx_joshy_demo"
SOURCE_TABLE = f"{CATALOG}.processed.cg_coin_historical_chart_data"
OUTPUT_TABLE = f"{CATALOG}.features.cg_bitcoin_ml_features"
SCALER_METADATA_TABLE = f"{CATALOG}.features.cg_bitcoin_feature_scaler"

COIN_ID = "bitcoin"
VS_CURRENCY = "usd"
CUTOFF_TIMESTAMP = "2026-08-19 13:00:00"
TRAIN_FRACTION = 0.70

FEATURE_COLUMNS = [
    "price_lag_1",
    "price_lag_2",
    "price_lag_3",
    "price_lag_6",
    "price_lag_12",
]

# COMMAND ----------

# DBTITLE 1,Normalize and filter the source data
source = (
    spark.table(SOURCE_TABLE)
    .select(
        "coin_id",
        F.lower("vs_currency").alias("vs_currency"),
        F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm:ss").alias("timestamp"),
        F.col("price").cast("double").alias("price"),
    )
    .where(
        (F.col("coin_id") == COIN_ID)
        & (F.lower(F.col("vs_currency")) == VS_CURRENCY)
        & F.col("timestamp").isNotNull()
        & (F.col("timestamp") <= F.to_timestamp(F.lit(CUTOFF_TIMESTAMP)))
    )
    .dropDuplicates(["coin_id", "vs_currency", "timestamp"])
)

# COMMAND ----------

# DBTITLE 1,Create lag features
lag_window = Window.partitionBy("coin_id", "vs_currency").orderBy("timestamp")
featured = source
for lag in [1, 2, 3, 6, 12]:
    featured = featured.withColumn(
        f"price_lag_{lag}",
        F.lag("price", lag).over(lag_window),
    )

featured = featured.dropna(subset=["price"] + FEATURE_COLUMNS)
row_count = featured.count()
if row_count < 100:
    raise ValueError(f"Only {row_count} usable rows found; at least 100 are required.")

# COMMAND ----------

# DBTITLE 1,Create a leakage-safe chronological scaler fitting set
# StandardScaler is a fitted transformer. Fitting it on validation/test rows would leak
# their distribution into training, so it is fitted only on the oldest 70% of observations.
row_window = Window.orderBy("timestamp")
split_at = int(row_count * TRAIN_FRACTION)
indexed = featured.withColumn("feature_row_number", F.row_number().over(row_window))
indexed = indexed.withColumn(
    "dataset_split",
    F.when(F.col("feature_row_number") <= split_at, F.lit("scaler_train"))
    .otherwise(F.lit("scaler_holdout")),
)

assembler = VectorAssembler(
    inputCols=FEATURE_COLUMNS,
    outputCol="unscaled_features",
    handleInvalid="error",
)
assembled = assembler.transform(indexed)

scaler = StandardScaler(
    inputCol="unscaled_features",
    outputCol="features",
    withMean=True,
    withStd=True,
)
scaler_model = scaler.fit(
    assembled.where(F.col("dataset_split") == "scaler_train")
)
scaled = scaler_model.transform(assembled)

# COMMAND ----------

# DBTITLE 1,Publish the feature table
(
    scaled.select(
        "coin_id",
        "timestamp",
        "vs_currency",
        "price",
        *FEATURE_COLUMNS,
        "unscaled_features",
        "features",
        "dataset_split",
    )
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(OUTPUT_TABLE)
)

# Persist the fitted parameters so the same transformation can be reproduced and audited.
scaler_metadata = [
    (
        feature_name,
        float(scaler_model.mean[index]),
        float(scaler_model.std[index]),
        CUTOFF_TIMESTAMP,
        TRAIN_FRACTION,
    )
    for index, feature_name in enumerate(FEATURE_COLUMNS)
]
(
    spark.createDataFrame(
        scaler_metadata,
        ["feature_name", "training_mean", "training_std", "data_cutoff", "train_fraction"],
    )
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SCALER_METADATA_TABLE)
)

print(f"Published {row_count:,} rows to {OUTPUT_TABLE}")
display(spark.table(OUTPUT_TABLE).orderBy("timestamp").limit(10))
display(spark.table(SCALER_METADATA_TABLE).orderBy("feature_name"))
