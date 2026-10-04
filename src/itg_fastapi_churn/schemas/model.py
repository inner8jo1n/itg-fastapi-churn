from datetime import datetime

from pydantic import BaseModel, Field, JsonValue

from itg_fastapi_churn.schemas.training import ModelType


class ModelMetrics(BaseModel):
    """
    Quality of the churn model on the test split

    :accuracy: float - share of correct predictions
    :f1: float - F1 score of the churn class (1)
    :roc_auc: float | None - area under the ROC curve; None when the test
        split has only one class and the score is undefined
    """

    accuracy: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    roc_auc: float | None = Field(default=None, ge=0, le=1)


class TrainingResponseChurn(ModelMetrics):
    """
    Result of POST /model/train: metrics and warnings from training

    :warnings: list[str] - problems noticed while training, for example
        that the model did not converge; empty when all went well
    """

    warnings: list[str] = Field(default_factory=list)


class ModelStatus(BaseModel):
    """
    Whether a trained model is available and how it performed

    :is_trained: bool - True if a model is loaded or has been trained
    :trained_at: datetime | None - moment of the last training
    :metrics: ModelMetrics | None - quality on the test split
    :model_type: ModelType | None - type of the trained classifier
    :hyperparameters: dict[str, JsonValue] | None - hyperparameters the
        classifier was trained with
    """

    is_trained: bool
    trained_at: datetime | None = None
    metrics: ModelMetrics | None = None
    model_type: ModelType | None = None
    hyperparameters: dict[str, JsonValue] | None = None
