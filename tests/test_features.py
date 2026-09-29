import pandas as pd

from itg_fastapi_churn.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    class_distribution,
    prepare_data,
)
from itg_fastapi_churn.schemas.churn import (
    EXAMPLE_FEATURES,
    FeatureVectorChurn,
)


def test_feature_columns_match_schema() -> None:
    schema_fields = set(FeatureVectorChurn.model_fields)

    assert set(FEATURE_COLUMNS) == schema_fields


def test_feature_groups_do_not_overlap() -> None:
    assert set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES) == set()


def test_class_distribution_counts_rows_per_class() -> None:
    target = pd.Series([0, 0, 1])

    distribution = class_distribution(target)

    assert distribution == {0: 2, 1: 1}


def test_prepare_data_separates_features_and_target() -> None:
    data = pd.DataFrame(
        [{**EXAMPLE_FEATURES, "churn": 0}, {**EXAMPLE_FEATURES, "churn": 1}]
    )

    features, target = prepare_data(data)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert TARGET_COLUMN not in list(features.columns)
    assert list(target) == [0, 1]


def test_prepare_data_drops_rows_with_missing_values() -> None:
    data = pd.DataFrame(
        [
            {**EXAMPLE_FEATURES, "churn": 0},
            {**EXAMPLE_FEATURES, "churn": 1, "region": None},
        ]
    )

    features, target = prepare_data(data)

    assert len(features) == 1
    assert len(target) == 1
