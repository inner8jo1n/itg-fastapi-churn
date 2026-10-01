from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.openapi.models import Example

from itg_fastapi_churn.api.dependencies import ModelStoreDep
from itg_fastapi_churn.api.error_docs import PREDICT_ERRORS
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
        **PREDICT_ERRORS,
    },
)
def predict(
    clients: Annotated[
        PredictionRequestChurn, Body(openapi_examples=REQUEST_EXAMPLES)
    ],
    store: ModelStoreDep,
) -> PredictionResponseChurn:
    """
    Predict churn and class probabilities for one client or a list

    The model is checked inside the endpoint, after the request body has
    been validated, so a wrong request is reported even without a model.

    :clients: PredictionRequestChurn - one client or a list of clients
    :store: ModelStore - where the trained model is kept

    :return: one prediction per client, in the same order
    """
    model = store.require_current()
    batch = clients if isinstance(clients, list) else [clients]
    return PredictionResponseChurn(
        predictions=predict_churn(model.pipeline, batch)
    )
