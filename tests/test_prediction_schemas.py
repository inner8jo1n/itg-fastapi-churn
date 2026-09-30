import pytest
from pydantic import ValidationError

from itg_fastapi_churn.schemas.prediction import (
    EXAMPLE_RESPONSE,
    ChurnPrediction,
    PredictionResponseChurn,
)


def test_example_response_is_valid() -> None:
    response = PredictionResponseChurn.model_validate(EXAMPLE_RESPONSE)

    assert response.predictions[0].churn == 1
    assert response.predictions[0].probabilities == {0: 0.35, 1: 0.65}


def test_prediction_rejects_non_binary_churn() -> None:
    with pytest.raises(ValidationError):
        ChurnPrediction(churn=2, probabilities={0: 0.5, 1: 0.5})
