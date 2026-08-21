import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--catalog",
    required=True,
)

args = parser.parse_args()
catalog = args.catalog

from model_training_utils import (
    filter_coin_data,
    save_model,
    save_to_catalog,
    train_coin_model,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_TABLE = (
    f"{catalog}.features."
    "cg_coin_forecast_features_train"
)

TEST_TABLE = (
    f"{catalog}.features."
    "cg_coin_forecast_features_test"
)


# Unity Catalog output tables

MODEL_PERFORMANCE_TABLE = (
    f"{catalog}.models."
    "cg_coin_model_performance"
)

FORECAST_TABLE = (
    f"{catalog}.models."
    "cg_coin_forecasts"
)


# Model files

MODEL_BASE_PATH = (
    f"/Volumes/{catalog}/models/"
    "crypto_forecaster"
)


VS_CURRENCY = "usd"


# ============================================================
# 1. Load feature datasets
# ============================================================

train_df = spark.table(
    TRAIN_TABLE
)

test_df = spark.table(
    TEST_TABLE
)


# ============================================================
# 2. Get available coins
# ============================================================

coin_ids = [
    row["coin_id"]
    for row in (
        train_df
        .filter(
            f"vs_currency = '{VS_CURRENCY}'"
        )
        .select("coin_id")
        .distinct()
        .orderBy("coin_id")
        .collect()
    )
]

print(
    f"Coins found: {coin_ids}"
)


# ============================================================
# 3. Train one model per coin
# ============================================================

results = []

all_forecasts = None


for coin_id in coin_ids:

    print(
        "\n" + "=" * 60
    )

    print(
        f"Training model: "
        f"{coin_id}/{VS_CURRENCY}"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # Count train/test rows
    # --------------------------------------------------------

    train_coin_df = filter_coin_data(
        df=train_df,
        coin_id=coin_id,
        vs_currency=VS_CURRENCY,
    )

    test_coin_df = filter_coin_data(
        df=test_df,
        coin_id=coin_id,
        vs_currency=VS_CURRENCY,
    )


    train_count = (
        train_coin_df.count()
    )

    test_count = (
        test_coin_df.count()
    )


    print(
        f"Training rows : {train_count}"
    )

    print(
        f"Test rows     : {test_count}"
    )


    # --------------------------------------------------------
    # Train model and generate forecasts
    # --------------------------------------------------------

    model, forecasts, metrics = train_coin_model(
        train_df=train_df,
        test_df=test_df,
        coin_id=coin_id,
        vs_currency=VS_CURRENCY,
    )


    # --------------------------------------------------------
    # Define model path
    # --------------------------------------------------------

    model_path = (
        f"{MODEL_BASE_PATH}/"
        f"{coin_id}_{VS_CURRENCY}_"
        f"linear_regression"
    )


    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    save_model(
        model=model,
        model_path=model_path,
    )


    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        "\nMODEL PERFORMANCE"
    )

    print(
        "-----------------"
    )

    print(
        f"RMSE : {metrics['rmse']:.6f}"
    )

    print(
        f"MAE  : {metrics['mae']:.6f}"
    )

    print(
        f"R²   : {metrics['r2']:.6f}"
    )

    print(
        "\nModel saved to:"
    )

    print(
        model_path
    )


    # --------------------------------------------------------
    # Store model performance
    # --------------------------------------------------------

    results.append(
        {
            "coin_id": coin_id,
            "vs_currency": VS_CURRENCY,
            "model_type": "linear_regression",
            "train_rows": train_count,
            "test_rows": test_count,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "model_path": model_path,
        }
    )


    # --------------------------------------------------------
    # Combine forecasts from all coins
    # --------------------------------------------------------

    if all_forecasts is None:

        all_forecasts = forecasts

    else:

        all_forecasts = (
            all_forecasts
            .unionByName(forecasts)
        )


# ============================================================
# 4. Build model performance summary
# ============================================================

if results:

    model_performance_df = (
        spark.createDataFrame(results)
    )


    # --------------------------------------------------------
    # Save model performance summary
    # --------------------------------------------------------

    save_to_catalog(
        df=model_performance_df,
        table_name=MODEL_PERFORMANCE_TABLE,
    )


    print(
        "\nModel performance saved to:"
    )

    print(
        MODEL_PERFORMANCE_TABLE
    )


    # --------------------------------------------------------
    # Display performance
    # --------------------------------------------------------

    display(
        model_performance_df
        .orderBy("coin_id")
    )


# ============================================================
# 5. Save forecasts for all coins
# ============================================================

if all_forecasts is not None:

    save_to_catalog(
        df=all_forecasts,
        table_name=FORECAST_TABLE,
    )


    print(
        "\nForecasts saved to:"
    )

    print(
        FORECAST_TABLE
    )


    # # --------------------------------------------------------
    # # Display forecasts
    # # --------------------------------------------------------

    # display(
    #     all_forecasts
    #     .orderBy(
    #         "coin_id",
    #         "timestamp",
    #     )
    # )