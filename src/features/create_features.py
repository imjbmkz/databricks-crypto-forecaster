import feature_engineering_utils as feu

# ============================================================
# 11. Run pipeline
# ============================================================

train_df, test_df = feu.build_feature_dataframes(
    spark=spark
)

# ============================================================
# 12. Save train and test datasets to Unity Catalog
# ============================================================

feu.save_to_catalog(
    df=train_df,
    table_name=feu.TRAIN_TABLE,
)

feu.save_to_catalog(
    df=test_df,
    table_name=feu.TEST_TABLE,
)