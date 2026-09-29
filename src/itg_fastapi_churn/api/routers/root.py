from fastapi import APIRouter

from itg_fastapi_churn.schemas.status import StatusResponse

RUNNING_MESSAGE = "ml churn service is running"

router = APIRouter(tags=["status"])


@router.get("/")
def read_root() -> StatusResponse:
    """
    Report that the service is running

    :return: service status response
    """
    return StatusResponse(message=RUNNING_MESSAGE)
