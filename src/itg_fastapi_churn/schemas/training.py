from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic.json_schema import JsonDict


class ModelType(StrEnum):
    """
    Classifiers the service can train

    :LOGREG: str - logistic regression
    :RANDOM_FOREST: str - random forest
    """

    LOGREG = "logreg"
    RANDOM_FOREST = "random_forest"


LOGREG_CONFIG_EXAMPLE: JsonDict = {
    "model_type": "logreg",
    "hyperparameters": {"C": 0.5},
}
RANDOM_FOREST_CONFIG_EXAMPLE: JsonDict = {
    "model_type": "random_forest",
    "hyperparameters": {"n_estimators": 200, "max_depth": 6},
}


class TrainingConfigChurn(BaseModel):
    """
    Which classifier to train and with which hyperparameters

    :model_type: ModelType - classifier to train
    :hyperparameters: dict[str, JsonValue] - scikit-learn parameters
        of the classifier; they override the service defaults
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [LOGREG_CONFIG_EXAMPLE, RANDOM_FOREST_CONFIG_EXAMPLE]
        },
    )

    model_type: ModelType = ModelType.LOGREG
    hyperparameters: dict[str, JsonValue] = Field(default_factory=dict)
