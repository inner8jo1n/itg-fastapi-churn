from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.openapi.models import Example

from itg_fastapi_churn.api.dependencies import ModelStoreDep, SplitDep
from itg_fastapi_churn.api.error_docs import TRAIN_ERRORS
from itg_fastapi_churn.ml.feature_schema import describe_features
from itg_fastapi_churn.ml.metrics import evaluate_model
from itg_fastapi_churn.ml.model import (
    resolve_hyperparameters,
    train_churn_model,
)
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.schemas.feature_schema import ModelSchemaResponse
from itg_fastapi_churn.schemas.model import ModelStatus, TrainingResponseChurn
from itg_fastapi_churn.schemas.training import (
    LOGREG_CONFIG_EXAMPLE,
    RANDOM_FOREST_CONFIG_EXAMPLE,
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

router = APIRouter(prefix="/model", tags=["model"])


@router.post(
    "/train",
    responses=TRAIN_ERRORS,
)
def train_model(
    split: SplitDep,
    store: ModelStoreDep,
    config: Annotated[
        TrainingConfigChurn | None, Body(openapi_examples=TRAINING_EXAMPLES)
    ] = None,
) -> TrainingResponseChurn:
    """
    Train the churn model, evaluate it on the test split and save it

    Without a request body logistic regression with defaults is trained.
    Dataset problems are reported before problems in the request body,
    because the dataset is loaded by a dependency.

    :split: DatasetSplit - stratified train/test split
    :store: ModelStore - where the trained model is kept
    :config: TrainingConfigChurn | None - model type and hyperparameters

    :return: accuracy and F1 score on the test split and training
        warnings
    """
    if config is None:
        config = TrainingConfigChurn()

    outcome = train_churn_model(split.x_train, split.y_train, config)
    metrics = evaluate_model(
        model=outcome.pipeline, features=split.x_test, target=split.y_test
    )
    store.save(
        TrainedModel(
            pipeline=outcome.pipeline,
            trained_at=datetime.now(UTC),
            metrics=metrics,
            model_type=config.model_type,
            hyperparameters=resolve_hyperparameters(config),
        )
    )
    return TrainingResponseChurn(
        **metrics.model_dump(), warnings=outcome.warnings
    )


@router.get("/status")
def get_model_status(store: ModelStoreDep) -> ModelStatus:
    """
    Show whether the model is trained, when and with what metrics

    :store: ModelStore - where the trained model is kept

    :return: current model status
    """
    return store.status()


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
