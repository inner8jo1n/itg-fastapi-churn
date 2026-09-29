import pandas as pd

from itg_fastapi_churn.ml.features import prepare_data
from itg_fastapi_churn.ml.model import train_churn_model
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES


def make_training_data() -> tuple[pd.DataFrame, pd.Series]:
    rows = [
        {**EXAMPLE_FEATURES, "churn": churn, "failed_payments": churn * 5}
        for churn in [0, 1] * 10
    ]
    features, target = prepare_data(pd.DataFrame(rows))

    return features, target


def test_train_churn_model_returns_fitted_pipeline() -> None:
    features, target = make_training_data()

    model = train_churn_model(features, target)

    assert len(model.predict(features)) == len(features)


def test_train_churn_model_learns_obvious_pattern() -> None:
    features, target = make_training_data()

    model = train_churn_model(features, target)

    assert list(model.predict(features)) == list(target)
