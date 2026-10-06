import logging
import math
from collections.abc import Mapping
from http import HTTPStatus

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import JsonValue
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint

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
from itg_fastapi_churn.schemas.error import ErrorResponse

STATUS_BY_ERROR: dict[type[ServiceError], int] = {
    DatasetNotFoundError: status.HTTP_404_NOT_FOUND,
    EmptyDatasetError: status.HTTP_409_CONFLICT,
    InvalidDatasetError: status.HTTP_409_CONFLICT,
    NotEnoughDataError: status.HTTP_409_CONFLICT,
    ModelNotTrainedError: status.HTTP_409_CONFLICT,
    IncompatibleModelError: status.HTTP_409_CONFLICT,
    InvalidHyperparametersError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    PredictionFailedError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    HistoryUnavailableError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}

MAX_ECHOED_TEXT = 100
MAX_ECHOED_FIELDS = 20
MAX_REPORTED_ERRORS = 20

logger = logging.getLogger(__name__)


def register_error_handlers(application: FastAPI) -> None:
    """
    Make every error answer with the common ErrorResponse format

    Unexpected errors are caught by a middleware, not by an exception
    handler: Starlette re-raises an error after its handler, and the
    server would log the same traceback a second time.

    :application: FastAPI - application to configure
    """
    application.add_exception_handler(ServiceError, service_error_handler)
    application.add_exception_handler(
        StarletteHTTPException, http_error_handler
    )
    application.add_exception_handler(
        RequestValidationError, validation_error_handler
    )
    application.middleware("http")(catch_unexpected_errors)


async def catch_unexpected_errors(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    """
    Answer any error no handler took care of with a neat 500

    :request: Request - incoming request
    :call_next: RequestResponseEndpoint - rest of the application

    :return: the normal response, or 500 if the request failed
    """
    try:
        return await call_next(request)
    except Exception as error:
        return await unexpected_error_handler(request, error)


async def service_error_handler(
    request: Request, error: Exception
) -> JSONResponse:
    """
    Answer an expected service error with its code and message

    :request: Request - failed request, named in the log
    :error: Exception - ServiceError raised by the service

    :return: error response with the status that fits the error
    """
    if not isinstance(error, ServiceError):
        return await unexpected_error_handler(request, error)

    status_code = _status_for(error)
    if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        logger.error(
            "%s failed with %s", _describe(request), error.code, exc_info=error
        )
    else:
        logger.warning(
            "%s failed with %s: %s",
            _describe(request),
            error.code,
            error.message,
        )
    return error_response(
        status_code, error.code, error.message, error.details
    )


async def http_error_handler(
    request: Request, error: Exception
) -> JSONResponse:
    """
    Answer framework HTTP errors, like an unknown path, in common format

    :request: Request - failed request, passed on if the error is not
        an HTTP one
    :error: Exception - HTTPException raised by FastAPI or Starlette

    :return: error response with the original status and headers
    """
    if not isinstance(error, StarletteHTTPException):
        return await unexpected_error_handler(request, error)

    code = _http_code(error.status_code)
    return error_response(
        error.status_code, code, str(error.detail), headers=error.headers
    )


async def validation_error_handler(
    request: Request, error: Exception
) -> JSONResponse:
    """
    Answer 422 listing the invalid fields of the request

    At most MAX_REPORTED_ERRORS fields are listed, so a request with
    thousands of mistakes still gets a small answer; the message tells
    how many problems there are in total.

    :request: Request - request that failed validation, named in the log
    :error: Exception - RequestValidationError raised by FastAPI

    :return: 422 response with one entry per invalid field
    """
    if not isinstance(error, RequestValidationError):
        return await unexpected_error_handler(request, error)

    errors = error.errors()
    logger.warning(
        "%s rejected: %d invalid fields", _describe(request), len(errors)
    )
    details: list[JsonValue] = [
        {
            "location": list(item["loc"]),
            "message": item["msg"],
            "type": item["type"],
            "input": _short_input(item.get("input")),
        }
        for item in jsonable_encoder(errors[:MAX_REPORTED_ERRORS])
    ]
    message = "Request data is invalid"
    if len(errors) > MAX_REPORTED_ERRORS:
        message += (
            f": {len(errors)} problems, "
            f"the first {MAX_REPORTED_ERRORS} are listed"
        )
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "validation_error",
        message,
        details,
    )


async def unexpected_error_handler(
    request: Request, error: Exception
) -> JSONResponse:
    """
    Hide unexpected failures behind a neat 500 and log the traceback

    :request: Request - failed request, named in the log
    :error: Exception - any error the service did not expect

    :return: 500 response without technical details
    """
    logger.error(
        "%s failed with an unexpected error",
        _describe(request),
        exc_info=error,
    )
    return error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "Internal server error, please try again later",
    )


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: JsonValue = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """
    Build a JSON response in the common ErrorResponse format

    Details may echo the rejected input, and JSON cannot hold Infinity or
    NaN, so such numbers are sent back as text.

    :status_code: int - HTTP status of the response
    :code: str - machine readable error code
    :message: str - human readable description
    :details: JsonValue - extra data about the error
    :headers: Mapping[str, str] | None - extra response headers

    :return: error response
    """
    body = ErrorResponse(code=code, message=message, details=details)
    return JSONResponse(
        status_code=status_code,
        content=_json_safe(body.model_dump(mode="json")),
        headers=dict(headers) if headers else None,
    )


def _describe(request: Request) -> str:
    """
    Name the request in a log message without its body or query

    The body and query may hold client data, so only the method and the
    path are logged.

    :request: Request - request being answered

    :return: text like "POST /predict"
    """
    return f"{request.method} {request.url.path}"


def _short_input(value: JsonValue) -> JsonValue:
    """
    Keep echoed input small, so an error never repeats a huge request back

    A list is described, a large object is described, a small object is
    shortened field by field and a long text is cut.

    :value: JsonValue - input that failed validation

    :return: the input itself, or its short form
    """
    if isinstance(value, list):
        return f"list of {len(value)} items"
    if isinstance(value, dict):
        if len(value) > MAX_ECHOED_FIELDS:
            return f"object with {len(value)} fields"
        return {key: _short_input(item) for key, item in value.items()}
    if isinstance(value, str) and len(value) > MAX_ECHOED_TEXT:
        return f"{value[:MAX_ECHOED_TEXT]}... ({len(value)} characters)"
    return value


def _http_code(status_code: int) -> str:
    """
    Turn an HTTP status into an error code, like 404 into "not_found"

    :status_code: int - HTTP status

    :return: snake case status name, or "http_error" for unknown statuses
    """
    try:
        phrase = HTTPStatus(status_code).phrase
    except ValueError:
        return "http_error"
    return phrase.lower().replace(" ", "_")


def _status_for(error: ServiceError) -> int:
    """
    Find the HTTP status for a service error

    :error: ServiceError - error to answer

    :return: HTTP status, 500 for errors without a known status
    """
    for error_class, status_code in STATUS_BY_ERROR.items():
        if isinstance(error, error_class):
            return status_code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _json_safe(value: object) -> object:
    """
    Replace Infinity and NaN with their text form at any nesting level

    :value: object - value produced by the JSON encoder

    :return: the same value that JSON can represent
    """
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
