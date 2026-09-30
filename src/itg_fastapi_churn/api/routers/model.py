from datetime import UTC, datetime

from fastapi import APIRouter

from itg_fastapi_churn.api.dependencies import ModelStoreDep, SplitDep
from itg_fastapi_churn.ml.metrics import evaluate_model
from itg_fastapi_churn.ml.model import train_churn_model
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.schemas.model import ModelMetrics, ModelStatus

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def train_model(split: SplitDep, store: ModelStoreDep) -> ModelMetrics:
    """
    Train the churn model, evaluate it on the test split and save it

    :split: DatasetSplit - stratified train/test split
    :store: ModelStore - where the trained model is kept

    :return: accuracy and F1 score on the test split
    """
    pipeline = train_churn_model(split.x_train, split.y_train)
    metrics = evaluate_model(
        model=pipeline, features=split.x_test, target=split.y_test
    )
    store.save(
        TrainedModel(
            pipeline=pipeline, trained_at=datetime.now(UTC), metrics=metrics
        )
    )
    return metrics


@router.get("/status")
def get_model_status(store: ModelStoreDep) -> ModelStatus:
    """
    Show whether the model is trained, when and with what metrics

    :store: ModelStore - where the trained model is kept

    :return: current model status
    """
    return store.status()
