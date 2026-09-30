import pytest
from pydantic import ValidationError

from itg_fastapi_churn.schemas.churn import (
    EXAMPLE_FEATURES,
    DatasetRowChurn,
    FeatureVectorChurn,
)


def test_feature_vector_accepts_valid_features() -> None:
    features = FeatureVectorChurn.model_validate(EXAMPLE_FEATURES)

    assert features.model_dump() == EXAMPLE_FEATURES


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("monthly_fee", -1.0),
        ("support_requests", -1),
        ("failed_payments", 1.5),
        ("autopay_enabled", 2),
    ],
)
def test_feature_vector_rejects_invalid_value(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn.model_validate({**EXAMPLE_FEATURES, field: value})


def test_dataset_row_includes_churn_target() -> None:
    row = DatasetRowChurn.model_validate({**EXAMPLE_FEATURES, "churn": 1})

    assert row.churn == 1


def test_dataset_row_rejects_non_binary_churn() -> None:
    with pytest.raises(ValidationError):
        DatasetRowChurn.model_validate({**EXAMPLE_FEATURES, "churn": 3})


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_feature_vector_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValidationError):
        FeatureVectorChurn.model_validate(
            {**EXAMPLE_FEATURES, "usage_hours": value}
        )


def test_dataset_row_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValidationError):
        DatasetRowChurn.model_validate(
            {**EXAMPLE_FEATURES, "usage_hours": float("inf"), "churn": 0}
        )
