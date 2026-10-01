from typing import Annotated

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag
from pydantic.json_schema import JsonDict

from itg_fastapi_churn.schemas.churn import FeatureVectorChurn

MAX_CLIENTS_PER_REQUEST = 1000

EXAMPLE_PREDICTION: JsonDict = {
    "churn": 1,
    "probabilities": {"0": 0.35, "1": 0.65},
}
EXAMPLE_RESPONSE: JsonDict = {"predictions": [EXAMPLE_PREDICTION]}

ClientBatch = Annotated[
    list[FeatureVectorChurn],
    Field(min_length=1, max_length=MAX_CLIENTS_PER_REQUEST),
]


def _request_shape(body: object) -> str:
    """
    Tell whether the request body is one client or a list of clients

    Only the matching variant is validated, so errors never mention the
    shape the client did not send.

    :body: object - raw request body

    :return: "clients" for a list, "client" otherwise
    """
    return "clients" if isinstance(body, list) else "client"


PredictionRequestChurn = Annotated[
    Annotated[FeatureVectorChurn, Tag("client")]
    | Annotated[ClientBatch, Tag("clients")],
    Discriminator(_request_shape),
]


class ChurnPrediction(BaseModel):
    """
    Prediction for one client

    :churn: int - predicted class, 1 if the client will leave
    :probabilities: dict[int, float] - probability of every churn class
    """

    churn: int = Field(ge=0, le=1)
    probabilities: dict[int, float]


class PredictionResponseChurn(BaseModel):
    """
    Predictions for all clients from the request, in the same order

    :predictions: list[ChurnPrediction] - one prediction per client
    """

    model_config = ConfigDict(
        json_schema_extra={"examples": [EXAMPLE_RESPONSE]},
    )

    predictions: list[ChurnPrediction]
