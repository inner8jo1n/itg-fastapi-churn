import logging
import time
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Body, Query
from fastapi.openapi.models import Example

from itg_fastapi_churn.api.dependencies import (
    ModelStoreDep,
    SplitDep,
    TrainingHistoryDep,
)
from itg_fastapi_churn.api.error_docs import METRICS_ERRORS, TRAIN_ERRORS
from itg_fastapi_churn.core.errors import HistoryUnavailableError
from itg_fastapi_churn.ml.feature_schema import describe_features
from itg_fastapi_churn.ml.history import TrainingHistory
from itg_fastapi_churn.ml.metrics import evaluate_model
from itg_fastapi_churn.ml.model import (
    resolve_hyperparameters,
    train_churn_model,
)
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.schemas.feature_schema import ModelSchemaResponse
from itg_fastapi_churn.schemas.history import (
    TrainingMetricsResponse,
    TrainingRecord,
)
from itg_fastapi_churn.schemas.model import ModelStatus, TrainingResponseChurn
from itg_fastapi_churn.schemas.training import (
    LOGREG_CONFIG_EXAMPLE,
    RANDOM_FOREST_CONFIG_EXAMPLE,
    ModelType,
    TrainingConfigChurn,
)

TRAINING_EXAMPLES = {
    "logreg": Example(
        summary="Logistic regression", value=LOGREG_CONFIG_EXAMPLE
    ),
    "random_forest": Example(
        summary="Random forest", value=RANDOM_FOREST_CONFIG_EXAMPLE
    ),
}

HISTORY_NOT_SAVED_WARNING = "Training was not added to the history"

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["model"])


@router.post(
    "/train",
    responses=TRAIN_ERRORS,
)
def train_model(
    split: SplitDep,
    store: ModelStoreDep,
    history: TrainingHistoryDep,
    config: Annotated[
        TrainingConfigChurn | None, Body(openapi_examples=TRAINING_EXAMPLES)
    ] = None,
) -> TrainingResponseChurn:
    """
    Train the churn model, evaluate it, save it and add it to the history

    Without a request body logistic regression with defaults is trained.
    Dataset problems are reported before problems in the request body,
    because the dataset is loaded by a dependency.

    :split: DatasetSplit - stratified train/test split
    :store: ModelStore - where the trained model is kept
    :history: TrainingHistory - where finished trainings are recorded
    :config: TrainingConfigChurn | None - model type and hyperparameters

    :return: accuracy, F1 score and ROC AUC on the test split and
        training warnings
    """
    if config is None:
        config = TrainingConfigChurn()

    logger.info("Training %s model", config.model_type)
    started = time.perf_counter()
    outcome = train_churn_model(split.x_train, split.y_train, config)
    metrics = evaluate_model(
        model=outcome.pipeline, features=split.x_test, target=split.y_test
    )
    trained = TrainedModel(
        pipeline=outcome.pipeline,
        trained_at=datetime.now(UTC),
        metrics=metrics,
        model_type=config.model_type,
        hyperparameters=resolve_hyperparameters(config),
    )
    store.save(trained)
    logger.info(
        "Model %s trained in %.2f s: %s",
        config.model_type,
        time.perf_counter() - started,
        metrics,
    )
    warnings = outcome.warnings + _record_training(history, trained)
    return TrainingResponseChurn(**metrics.model_dump(), warnings=warnings)


def _record_training(
    history: TrainingHistory, trained: TrainedModel
) -> list[str]:
    """
    Add the training to the history without failing the request

    The model is already saved, so a history problem is only logged and
    reported to the client as a warning.

    :history: TrainingHistory - where finished trainings are recorded
    :trained: TrainedModel - model that has just been saved

    :return: warning about the lost record, empty if it was added
    """
    record = TrainingRecord(
        trained_at=trained.trained_at,
        model_type=trained.model_type,
        hyperparameters=trained.hyperparameters,
        metrics=trained.metrics,
    )
    try:
        history.append(record)
    except HistoryUnavailableError:
        logger.exception(HISTORY_NOT_SAVED_WARNING)
        return [HISTORY_NOT_SAVED_WARNING]
    return []


@router.get("/status")
def get_model_status(store: ModelStoreDep) -> ModelStatus:
    """
    Show whether the model is trained, when and with what metrics

    :store: ModelStore - where the trained model is kept

    :return: current model status
    """
    return store.status()


@router.get("/metrics", responses=METRICS_ERRORS)
def get_model_metrics(
    history: TrainingHistoryDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    model_type: ModelType | None = None,
) -> TrainingMetricsResponse:
    """
    Show metrics of the latest training, the best one and recent ones

    Compare the recent trainings to see which model settings work best.

    :history: TrainingHistory - where finished trainings are recorded
    :limit: int - how many recent trainings to list, from 1 to 100
    :model_type: ModelType | None - show only this model type, all
        trainings if not given

    :return: latest, best by F1 and recent trainings
    """
    records = history.records(model_type)
    return TrainingMetricsResponse.from_records(records, limit)


@router.get("/schema")
def get_model_schema(store: ModelStoreDep) -> ModelSchemaResponse:
    """
    List the features POST /predict expects, with types and limits

    Categories seen by the model are included once it is trained.

    :store: ModelStore - where the trained model is kept

    :return: feature names, types, limits and known categories
    """
    model = store.current
    return describe_features(model.pipeline if model else None)
