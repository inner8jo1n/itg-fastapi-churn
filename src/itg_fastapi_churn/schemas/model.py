from datetime import datetime

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    """
    Quality of the churn model on the test split

    :accuracy: float - share of correct predictions
    :f1: float - F1 score of the churn class (1)
    """

    accuracy: float
    f1: float


class ModelStatus(BaseModel):
    """
    Whether a trained model is available and how it performed

    :is_trained: bool - True if a model is loaded or has been trained
    :trained_at: datetime | None - moment of the last training
    :metrics: ModelMetrics | None - quality on the test split
    """

    is_trained: bool
    trained_at: datetime | None = None
    metrics: ModelMetrics | None = None
