# Databricks notebook source
# MAGIC %md
# MAGIC # Bitcoin exploratory analysis and Spark ML experiments
# MAGIC Predict Bitcoin/USD at `t` from prices at `t-5`, `t-10`, `t-15`, `t-30`, and `t-60` minutes.
# MAGIC Models use identical chronological holdouts and are compared with persistence (`price_lag_1`).

# COMMAND ----------

# DBTITLE 1,Configuration and imports
from datetime import datetime
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import Window, functions as F

CATALOG = "dbx_joshy_demo"
SOURCE_TABLE = f"{CATALOG}.processed.cg_coin_historical_chart_data"
COIN_ID, VS_CURRENCY = "bitcoin", "usd"
CUTOFF_TIMESTAMP = "2026-08-19 13:00:00"
SEED = 42
FEATURE_COLUMNS = ["price_lag_1", "price_lag_2", "price_lag_3", "price_lag_6", "price_lag_12"]

print(f"Run: {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"Scope: {COIN_ID}/{VS_CURRENCY}; timestamp <= {CUTOFF_TIMESTAMP}")

# COMMAND ----------

# DBTITLE 1,Load Bitcoin and create the project lag features
# The cutoff is explicit. No forward fill is used because that would create synthetic observations.
raw = (
    spark.table(SOURCE_TABLE)
    .select("coin_id", "vs_currency", "timestamp", F.col("price").cast("double").alias("price"))
    .where(
        (F.col("coin_id") == COIN_ID)
        & (F.lower(F.col("vs_currency")) == VS_CURRENCY)
        & (F.col("timestamp") <= F.to_timestamp(F.lit(CUTOFF_TIMESTAMP)))
    )
    .dropDuplicates(["coin_id", "vs_currency", "timestamp"])
)

lag_window = Window.partitionBy("coin_id", "vs_currency").orderBy("timestamp")
features = raw
for lag in [1, 2, 3, 6, 12]:
    features = features.withColumn(f"price_lag_{lag}", F.lag("price", lag).over(lag_window))
features = features.dropna(subset=["price"] + FEATURE_COLUMNS).orderBy("timestamp").cache()

row_count = features.count()
if row_count < 100:
    raise ValueError(f"Only {row_count} usable rows found; at least 100 are required.")
display(features.limit(10))

# COMMAND ----------

# DBTITLE 1,Quality and cadence analysis
ordered = Window.orderBy("timestamp")
quality = features.withColumn(
    "minutes_since_previous",
    (F.col("timestamp").cast("long") - F.lag("timestamp").over(ordered).cast("long")) / 60,
)
quality_summary = quality.agg(
    F.count("*").alias("rows"),
    F.min("timestamp").alias("start"),
    F.max("timestamp").alias("end"),
    F.countDistinct("timestamp").alias("distinct_timestamps"),
    F.expr("percentile_approx(minutes_since_previous, 0.5)").alias("median_interval_minutes"),
    F.sum(F.when(F.col("minutes_since_previous") != 5, 1).otherwise(0)).alias("non_5_minute_gaps"),
)
display(quality_summary)

gap_count = quality.where(F.col("minutes_since_previous") != 5).count()
if gap_count:
    print(f"WARNING: {gap_count} non-5-minute gaps. At those points these are row lags, not exact clock-time lags.")

# COMMAND ----------

# DBTITLE 1,Descriptive statistics and recent price history
display(features.select("price", *FEATURE_COLUMNS).summary())

correlations = [(column, features.stat.corr("price", column)) for column in FEATURE_COLUMNS]
display(spark.createDataFrame(correlations, ["feature", "correlation_with_price"]))

# Bound driver collection to seven days; model training stays distributed.
plot_pdf = (
    features.select("timestamp", "price").orderBy(F.col("timestamp").desc()).limit(2016)
    .orderBy("timestamp").toPandas()
)
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(plot_pdf["timestamp"], plot_pdf["price"], color="#f7931a", linewidth=1)
ax.set(title="Bitcoin/USD (most recent 7 days)", xlabel="Timestamp", ylabel="USD")
ax.grid(alpha=.25)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Chronological train-validation-test split
# 70/15/15 avoids the future-to-past leakage caused by randomSplit or random cross-validation.
indexed = features.withColumn("row_number", F.row_number().over(Window.orderBy("timestamp")))
n_rows = indexed.count()
train_end, validation_end = int(n_rows * .70), int(n_rows * .85)
train = indexed.where(F.col("row_number") <= train_end).drop("row_number").cache()
validation = indexed.where(
    (F.col("row_number") > train_end) & (F.col("row_number") <= validation_end)
).drop("row_number").cache()
test = indexed.where(F.col("row_number") > validation_end).drop("row_number").cache()

split_rows = []
for name, frame in [("train", train), ("validation", validation), ("test", test)]:
    row = frame.agg(F.count("*").alias("rows"), F.min("timestamp").alias("start"), F.max("timestamp").alias("end")).first()
    split_rows.append((name, row["rows"], row["start"], row["end"]))
display(spark.createDataFrame(split_rows, ["split", "rows", "start", "end"]))

# COMMAND ----------

# DBTITLE 1,Common metrics and persistence baseline
evaluators = {
    metric: RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName=metric)
    for metric in ["rmse", "mae", "r2"]
}

def score(predictions, model_name, split_name):
    scored = predictions.select("timestamp", "price", "price_lag_1", F.col("prediction").cast("double")).dropna()
    extra = scored.agg(
        (F.avg(F.abs((F.col("price") - F.col("prediction")) / F.col("price"))) * 100).alias("mape_pct"),
        F.avg(F.when(
            F.signum(F.col("price") - F.col("price_lag_1")) ==
            F.signum(F.col("prediction") - F.col("price_lag_1")), 1.0
        ).otherwise(0.0)).alias("direction_accuracy"),
    ).first()
    return {
        "model": model_name, "split": split_name,
        "rmse": evaluators["rmse"].evaluate(scored),
        "mae": evaluators["mae"].evaluate(scored),
        "r2": evaluators["r2"].evaluate(scored),
        "mape_pct": extra["mape_pct"],
        "direction_accuracy": extra["direction_accuracy"],
    }

def persistence(frame):
    return frame.withColumn("prediction", F.col("price_lag_1"))

validation_results = [score(persistence(validation), "Persistence", "validation")]

# COMMAND ----------

# DBTITLE 1,Spark ML candidates
assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="unscaled_features")
scaler = StandardScaler(inputCol="unscaled_features", outputCol="features", withMean=True, withStd=True)
candidates = [
    ("LinearRegression", Pipeline(stages=[assembler, scaler, LinearRegression(
        featuresCol="features", labelCol="price", regParam=0.0
    )])),
]
for trees, depth in [(50, 5), (100, 8)]:
    candidates.append((f"RandomForest(numTrees={trees},maxDepth={depth})", Pipeline(stages=[
        assembler, RandomForestRegressor(
            featuresCol="unscaled_features", labelCol="price", numTrees=trees,
            maxDepth=depth, subsamplingRate=.8, seed=SEED
        )
    ])))
for iterations, depth in [(50, 3), (100, 5)]:
    candidates.append((f"GBT(maxIter={iterations},maxDepth={depth})", Pipeline(stages=[
        assembler, GBTRegressor(
            featuresCol="unscaled_features", labelCol="price", maxIter=iterations,
            maxDepth=depth, stepSize=.05, subsamplingRate=.8, seed=SEED
        )
    ])))

# Explicit validation preserves time order; Spark's standard TrainValidationSplit is random.
for model_name, estimator in candidates:
    fitted = estimator.fit(train)
    result = score(fitted.transform(validation), model_name, "validation")
    validation_results.append(result)
    print(f"{model_name}: validation RMSE = {result['rmse']:.4f}")

validation_results_df = spark.createDataFrame(validation_results).orderBy("rmse")
display(validation_results_df)

# COMMAND ----------

# DBTITLE 1,Refit selected model and evaluate once on test
best_model_name = validation_results_df.first()["model"]
print(f"Selected using validation RMSE: {best_model_name}")
train_validation = train.unionByName(validation).cache()
test_results = [score(persistence(test), "Persistence", "test")]

if best_model_name == "Persistence":
    best_test_predictions = persistence(test).cache()
    best_fitted_model = None
else:
    best_estimator = next(estimator for name, estimator in candidates if name == best_model_name)
    best_fitted_model = best_estimator.fit(train_validation)
    best_test_predictions = best_fitted_model.transform(test).cache()
    test_results.append(score(best_test_predictions, best_model_name, "test"))

test_results_df = spark.createDataFrame(test_results).orderBy("rmse")
display(test_results_df)

# COMMAND ----------

# DBTITLE 1,Test predictions and residuals
residual_pdf = best_test_predictions.select("timestamp", "price", "prediction").orderBy("timestamp").toPandas()
residual_pdf["residual"] = residual_pdf["price"] - residual_pdf["prediction"]
fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
axes[0].plot(residual_pdf["timestamp"], residual_pdf["price"], label="actual", linewidth=1.5)
axes[0].plot(residual_pdf["timestamp"], residual_pdf["prediction"], label="prediction", linewidth=1)
axes[0].set(title=f"Test predictions: {best_model_name}", ylabel="USD"); axes[0].legend()
axes[1].plot(residual_pdf["timestamp"], residual_pdf["residual"], linewidth=1)
axes[1].axhline(0, color="black", linestyle="--", linewidth=.8)
axes[1].set(title="Residuals", xlabel="Timestamp", ylabel="Actual - predicted")
for axis in axes: axis.grid(alpha=.25)
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Selection guidance
# MAGIC - The production candidate is selected by validation RMSE; test data is evaluated once.
# MAGIC - Prefer a learned model only when it improves materially over persistence on the test period.
# MAGIC - Review MAE, MAPE, directional accuracy, and residuals alongside RMSE.
# MAGIC - Tree models cannot extrapolate outside the training target range, so persistence and linear regression remain important during regime shifts.
# MAGIC - Next, log the fitted Spark pipeline and metrics to MLflow, retrain on a rolling schedule, and monitor it continuously against persistence.
# MAGIC - This is a one-step 5-minute forecasting experiment, not a trading-profitability evaluation.
