import math

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from itg_fastapi_churn.ml.model import InvalidHyperparametersError


def register_error_handlers(application: FastAPI) -> None:
    """
    Attach the service error handlers to the application

    :application: FastAPI - application to configure
    """
    application.add_exception_handler(
        RequestValidationError, validation_error_handler
    )
    application.add_exception_handler(
        InvalidHyperparametersError, hyperparameters_error_handler
    )


async def validation_error_handler(
    _request: Request, error: Exception
) -> JSONResponse:
    """
    Answer 422 with the validation errors, like FastAPI does by default

    Errors echo the rejected input, and JSON cannot hold Infinity or NaN,
    so such numbers are sent back as text instead of failing with 500.

    :_request: Request - request that failed validation, not used
    :error: Exception - RequestValidationError raised by FastAPI

    :return: 422 response with the error details
    """
    errors = (
        error.errors() if isinstance(error, RequestValidationError) else []
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": _json_safe(jsonable_encoder(errors))},
    )


async def hyperparameters_error_handler(
    _request: Request, error: Exception
) -> JSONResponse:
    """
    Answer 422 when the classifier rejects the requested hyperparameters

    :_request: Request - training request, not used
    :error: Exception - InvalidHyperparametersError with sklearn's message

    :return: 422 response that names the wrong hyperparameter
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(error)},
    )


def _json_safe(value: object) -> object:
    """
    Replace Infinity and NaN with their text form at any nesting level

    :value: object - value produced by jsonable_encoder

    :return: the same value that JSON can represent
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
