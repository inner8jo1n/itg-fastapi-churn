from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from itg_fastapi_churn.schemas.training import ModelType


class HealthStatus(StrEnum):
    """
    Overall state of the service

    :OK: str - the model is loaded and the dataset can be read
    :DEGRADED: str - the service runs, but the model or the dataset is
        missing, so some endpoints answer with errors
    """

    OK = "ok"
    DEGRADED = "degraded"


class ModelHealth(BaseModel):
    """
    Whether a trained model is ready for predictions

    :available: bool - True if a model is loaded or has been trained
    :model_type: ModelType | None - type of the loaded model
    :trained_at: datetime | None - moment the loaded model was trained
    """

    available: bool
    model_type: ModelType | None = None
    trained_at: datetime | None = None


class DatasetHealth(BaseModel):
    """
    Whether the training dataset can be read

    :available: bool - True if the file exists, has all columns and rows
    :problem: str | None - error code explaining why it is unavailable
    """

    available: bool
    problem: str | None = None


class HealthResponse(BaseModel):
    """
    Result of GET /health

    :status: HealthStatus - ok when both the model and the dataset are
        available, degraded otherwise
    :model: ModelHealth - state of the trained model
    :dataset: DatasetHealth - state of the training dataset
    """

    status: HealthStatus
    model: ModelHealth
    dataset: DatasetHealth
