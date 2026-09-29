from pydantic import BaseModel


class StatusResponse(BaseModel):
    """
    Response telling that the service is running

    :message: str - service status text
    """

    message: str
