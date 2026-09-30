from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import joblib
from pydantic import JsonValue
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.schemas.model import ModelMetrics
from itg_fastapi_churn.schemas.training import ModelType


class ModelLoadError(Exception):
    """
    Saved model file exists but cannot be used
    """


@dataclass(frozen=True)
class TrainedModel:
    """
    Trained churn model together with facts about its training

    :pipeline: Pipeline - fitted churn pipeline
    :trained_at: datetime - moment the training finished
    :metrics: ModelMetrics - quality on the test split
    :model_type: ModelType - type of the trained classifier
    :hyperparameters: dict[str, JsonValue] - hyperparameters the
        classifier was trained with, defaults included
    """

    pipeline: Pipeline
    trained_at: datetime
    metrics: ModelMetrics
    model_type: ModelType
    hyperparameters: dict[str, JsonValue]


def save_churn_model(model: TrainedModel, path: Path) -> None:
    """
    Save the trained model to a file, creating missing folders

    The model is written to a temporary file first and then moved over
    the target, so a crash during writing never leaves a broken file.

    :model: TrainedModel - model to save
    :path: Path - target file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    joblib.dump(model, temporary_path)
    temporary_path.replace(path)


def load_churn_model(path: Path) -> TrainedModel | None:
    """
    Load a model saved by save_churn_model

    joblib runs code from the file while loading, so only files written
    by this service must be loaded. Unpickling a damaged file can fail
    with many different exception types, so all of them are reported
    as ModelLoadError.

    :path: Path - file with the saved model

    :return: saved model, or None if no model has been saved yet
    """
    if not path.exists():
        return None

    try:
        loaded = joblib.load(path)
    except Exception as error:
        raise ModelLoadError(f"Cannot read model file {path}") from error

    if not isinstance(loaded, TrainedModel):
        raise ModelLoadError(f"Model file {path} has unexpected content")

    missing = [
        field.name
        for field in fields(TrainedModel)
        if not hasattr(loaded, field.name)
    ]
    if missing:
        raise ModelLoadError(
            f"Model file {path} was saved by an older version, "
            f"missing: {', '.join(missing)}"
        )

    return loaded
