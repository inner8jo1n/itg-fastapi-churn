from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.openapi.models import Example

from itg_fastapi_churn.api.dependencies import (
    MODEL_NOT_TRAINED,
    TrainedModelDep,
)
from itg_fastapi_churn.ml.predict import predict_churn
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES
from itg_fastapi_churn.schemas.prediction import (
    EXAMPLE_RESPONSE,
    PredictionRequestChurn,
    PredictionResponseChurn,
)

RISKY_CLIENT_EXAMPLE = {
    **EXAMPLE_FEATURES,
    "usage_hours": 3.5,
    "support_requests": 4,
    "failed_payments": 2,
    "autopay_enabled": 0,
}
REQUEST_EXAMPLES = {
    "one_client": Example(
        summary="One client",
        value=EXAMPLE_FEATURES,
    ),
    "several_clients": Example(
        summary="Several clients",
        description="Predictions come back in the same order.",
        value=[EXAMPLE_FEATURES, RISKY_CLIENT_EXAMPLE],
    ),
}

router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    responses={
        200: {"content": {"application/json": {"example": EXAMPLE_RESPONSE}}},
        409: {"description": MODEL_NOT_TRAINED},
    },
)
def predict(
    clients: Annotated[
        PredictionRequestChurn, Body(openapi_examples=REQUEST_EXAMPLES)
    ],
    model: TrainedModelDep,
) -> PredictionResponseChurn:
    """
    Predict churn and class probabilities for one client or a list

    :clients: PredictionRequestChurn - one client or a list of clients
    :model: TrainedModel - current trained model

    :return: one prediction per client, in the same order
    """
    batch = clients if isinstance(clients, list) else [clients]
    return PredictionResponseChurn(
        predictions=predict_churn(model.pipeline, batch)
    )
