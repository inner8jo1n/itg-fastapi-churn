from pydantic import BaseModel, JsonValue


class ErrorResponse(BaseModel):
    """
    Body of every error response of the service

    :code: str - machine readable error code, stable between releases
    :message: str - human readable description of the problem
    :details: JsonValue - extra data, for example the invalid fields
    """

    code: str
    message: str
    details: JsonValue = None
