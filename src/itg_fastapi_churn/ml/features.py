from typing import cast

import pandas as pd

TARGET_COLUMN = "churn"
NUMERIC_FEATURES = (
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
)
CATEGORICAL_FEATURES = ("region", "device_type", "payment_method")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def class_distribution(target: pd.Series) -> dict[int, int]:
    """
    Count rows of every churn class

    :target: pd.Series - churn values

    :return: number of rows per class, ordered by class
    """
    return cast(
        dict[int, int],
        target.value_counts().sort_index().to_dict(),
    )


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into a feature matrix and the churn target

    Rows with missing values are dropped so that no statistics leak
    from the test split into the train split.

    :data: pd.DataFrame - dataset with feature and target columns

    :return: feature matrix in FEATURE_COLUMNS order and churn target
    """
    clean = data.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])

    features = clean[list(FEATURE_COLUMNS)]
    target = clean[TARGET_COLUMN]

    return features, target
