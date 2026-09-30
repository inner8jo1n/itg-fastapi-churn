import pytest
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.ml.features import prepare_data
from itg_fastapi_churn.ml.model import train_churn_model
from itg_fastapi_churn.ml.predict import predict_churn
from itg_fastapi_churn.schemas.churn import (
    EXAMPLE_FEATURES,
    FeatureVectorChurn,
)

LOYAL_CLIENT = FeatureVectorChurn.model_validate(
    {**EXAMPLE_FEATURES, "failed_payments": 0}
)
RISKY_CLIENT = FeatureVectorChurn.model_validate(
    {**EXAMPLE_FEATURES, "failed_payments": 5}
)


@pytest.fixture
def pipeline(training_dataset: ChurnDataset) -> Pipeline:
    features, target = prepare_data(training_dataset.data)
    return train_churn_model(features, target)


def test_predict_churn_keeps_client_order(pipeline: Pipeline) -> None:
    predictions = predict_churn(pipeline, [RISKY_CLIENT, LOYAL_CLIENT])

    assert [prediction.churn for prediction in predictions] == [1, 0]


def test_predict_churn_returns_probabilities_of_both_classes(
    pipeline: Pipeline,
) -> None:
    [prediction] = predict_churn(pipeline, [RISKY_CLIENT])

    assert set(prediction.probabilities) == {0, 1}
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert prediction.probabilities[1] > prediction.probabilities[0]


def test_predict_churn_without_clients_returns_nothing(
    pipeline: Pipeline,
) -> None:
    assert predict_churn(pipeline, []) == []
