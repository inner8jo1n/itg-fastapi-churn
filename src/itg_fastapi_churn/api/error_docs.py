from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from fastapi import status
from pydantic import JsonValue

from itg_fastapi_churn.api.errors import STATUS_BY_ERROR
from itg_fastapi_churn.core.errors import (
    DatasetNotFoundError,
    EmptyDatasetError,
    HistoryUnavailableError,
    IncompatibleModelError,
    InvalidDatasetError,
    InvalidHyperparametersError,
    ModelNotTrainedError,
    NotEnoughDataError,
    PredictionFailedError,
    ServiceError,
)
from itg_fastapi_churn.ml.features import FEATURE_COLUMNS
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES
from itg_fastapi_churn.schemas.error import ErrorResponse


@dataclass(frozen=True)
class ErrorExample:
    """
    One documented error response

    :name: str - key of the example in the docs
    :summary: str - short title shown in the docs
    :status_code: int - HTTP status of the response
    :body: ErrorResponse - response body
    """

    name: str
    summary: str
    status_code: int
    body: ErrorResponse


def service_error_example(
    error_class: type[ServiceError],
    summary: str,
    name: str | None = None,
    message: str | None = None,
    details: JsonValue = None,
) -> ErrorExample:
    """
    Document a service error with the code and status the API really uses

    :error_class: type[ServiceError] - documented error
    :summary: str - short title shown in the docs
    :name: str | None - key of the example, the error code if not given
    :message: str | None - message, the default one if not given
    :details: JsonValue - example details

    :return: documented error response
    """
    return ErrorExample(
        name=name or error_class.code,
        summary=summary,
        status_code=STATUS_BY_ERROR[error_class],
        body=ErrorResponse(
            code=error_class.code,
            message=message or error_class.default_message,
            details=details,
        ),
    )


def validation_error_example(
    name: str, summary: str, details: list[JsonValue]
) -> ErrorExample:
    """
    Document a request validation error

    :name: str - key of the example in the docs
    :summary: str - short title shown in the docs
    :details: list[JsonValue] - invalid fields, as the API reports them

    :return: documented error response
    """
    return ErrorExample(
        name=name,
        summary=summary,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        body=ErrorResponse(
            code="validation_error",
            message="Request data is invalid",
            details=details,
        ),
    )


def error_responses(*examples: ErrorExample) -> dict[int | str, Any]:
    """
    Turn error examples into the responses argument of a FastAPI route

    Examples with the same status are shown together under that status.

    :examples: ErrorExample - documented errors of the route

    :return: OpenAPI responses grouped by status code
    """
    responses: dict[int | str, Any] = {}
    for example in examples:
        response = responses.setdefault(
            example.status_code,
            {
                "model": ErrorResponse,
                "description": HTTPStatus(example.status_code).phrase,
                "content": {"application/json": {"examples": {}}},
            },
        )
        response["content"]["application/json"]["examples"][example.name] = {
            "summary": example.summary,
            "value": example.body.model_dump(mode="json"),
        }
    return responses


INTERNAL_ERROR = ErrorExample(
    name="internal_error",
    summary="Unexpected failure, details are only in the server log",
    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    body=ErrorResponse(
        code="internal_error",
        message="Internal server error, please try again later",
    ),
)

TRAIN_ERRORS = error_responses(
    service_error_example(DatasetNotFoundError, "Dataset file is missing"),
    service_error_example(EmptyDatasetError, "Dataset has no rows"),
    service_error_example(
        InvalidDatasetError,
        "Dataset has invalid rows",
        message="Dataset has invalid rows: 1",
        details={
            "invalid_rows": 1,
            "examples": [
                {
                    "row": 7,
                    "errors": [
                        {
                            "field": "churn",
                            "message": (
                                "Input should be less than or equal to 1"
                            ),
                        }
                    ],
                }
            ],
        },
    ),
    service_error_example(
        NotEnoughDataError,
        "Only one churn class",
        name="single_class",
        message="Dataset must contain both churn classes",
        details={"classes": [0]},
    ),
    service_error_example(
        NotEnoughDataError,
        "Too few rows",
        name="too_few_rows",
        message="Not enough rows to split the dataset",
        details={"rows": 3},
    ),
    service_error_example(
        InvalidHyperparametersError,
        "Bad hyperparameter",
        message=(
            "The 'C' parameter of LogisticRegression must be a float "
            "in the range (0.0, inf]. Got -1 instead."
        ),
    ),
    validation_error_example(
        "unknown_model_type",
        "Unknown model type",
        [
            {
                "location": ["body", "model_type"],
                "message": "Input should be 'logreg' or 'random_forest'",
                "type": "enum",
                "input": "svm",
            }
        ],
    ),
    INTERNAL_ERROR,
)

PREDICT_ERRORS = error_responses(
    service_error_example(ModelNotTrainedError, "No trained model yet"),
    service_error_example(
        IncompatibleModelError,
        "Model expects other features",
        details={"trained_features": [*FEATURE_COLUMNS, "extra"]},
    ),
    service_error_example(
        PredictionFailedError, "Model failed, details are in the server log"
    ),
    validation_error_example(
        "extra_feature",
        "Unknown feature",
        [
            {
                "location": ["body", "client", "extra"],
                "message": "Extra inputs are not permitted",
                "type": "extra_forbidden",
                "input": 1,
            }
        ],
    ),
    validation_error_example(
        "missing_feature",
        "Feature is missing",
        [
            {
                "location": ["body", "client", "region"],
                "message": "Field required",
                "type": "missing",
                "input": {
                    name: value
                    for name, value in EXAMPLE_FEATURES.items()
                    if name != "region"
                },
            }
        ],
    ),
    validation_error_example(
        "wrong_type",
        "Value has a wrong type",
        [
            {
                "location": ["body", "client", "monthly_fee"],
                "message": "Input should be a valid number",
                "type": "float_type",
                "input": "19.99",
            }
        ],
    ),
    INTERNAL_ERROR,
)

METRICS_ERRORS = error_responses(
    service_error_example(
        HistoryUnavailableError,
        "History file cannot be read, details are in the server log",
        message="Training history cannot be read",
    ),
    validation_error_example(
        "limit_out_of_range",
        "Too many records requested",
        [
            {
                "location": ["query", "limit"],
                "message": "Input should be less than or equal to 100",
                "type": "less_than_equal",
                "input": "1000",
            }
        ],
    ),
    validation_error_example(
        "unknown_model_type",
        "Unknown model type",
        [
            {
                "location": ["query", "model_type"],
                "message": "Input should be 'logreg' or 'random_forest'",
                "type": "enum",
                "input": "svm",
            }
        ],
    ),
    INTERNAL_ERROR,
)
