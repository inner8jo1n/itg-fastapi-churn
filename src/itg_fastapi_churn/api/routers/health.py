from fastapi import APIRouter

from itg_fastapi_churn.api.dependencies import ModelStoreDep, SettingsDep
from itg_fastapi_churn.core.config import Settings
from itg_fastapi_churn.core.errors import ServiceError
from itg_fastapi_churn.dataset.loader import check_dataset
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.schemas.health import (
    DatasetHealth,
    HealthResponse,
    HealthStatus,
    ModelHealth,
)

router = APIRouter(tags=["status"])


@router.get("/health")
def get_health(store: ModelStoreDep, settings: SettingsDep) -> HealthResponse:
    """
    Report whether the model and the training dataset are available

    Always answers 200 while the service runs: a missing model or dataset
    makes the status degraded, because the service can still train a
    model once the dataset is back.

    :store: ModelStore - where the trained model is kept
    :settings: Settings - application settings with the dataset path

    :return: overall status and the state of the model and the dataset
    """
    model = _model_health(store.current)
    dataset = _dataset_health(settings)
    is_ok = model.available and dataset.available
    return HealthResponse(
        status=HealthStatus.OK if is_ok else HealthStatus.DEGRADED,
        model=model,
        dataset=dataset,
    )


def _model_health(model: TrainedModel | None) -> ModelHealth:
    """
    Describe the model kept in memory

    :model: TrainedModel | None - current model, None if not trained

    :return: whether the model is available, its type and training time
    """
    if model is None:
        return ModelHealth(available=False)
    return ModelHealth(
        available=True,
        model_type=model.model_type,
        trained_at=model.trained_at,
    )


def _dataset_health(settings: Settings) -> DatasetHealth:
    """
    Check that the training dataset can be read

    :settings: Settings - application settings with the dataset path

    :return: whether the dataset is available and the problem code if not
    """
    try:
        check_dataset(settings.dataset_path)
    except ServiceError as error:
        return DatasetHealth(available=False, problem=error.code)
    return DatasetHealth(available=True)
