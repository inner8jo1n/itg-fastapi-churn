from typing import Annotated

from fastapi import Depends, Request

from itg_fastapi_churn.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.dataset.loader import load_dataset
from itg_fastapi_churn.errors import (
    EmptyDatasetError,
    NotEnoughDataError,
)
from itg_fastapi_churn.ml.features import prepare_data
from itg_fastapi_churn.ml.history import TrainingHistory
from itg_fastapi_churn.ml.split import DatasetSplit, split_dataset
from itg_fastapi_churn.ml.store import ModelStore

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_dataset(
    settings: SettingsDep,
) -> ChurnDataset:
    """
    Load the training dataset from the path set in the settings

    :settings: Settings - application settings with the dataset path

    :return: loaded churn dataset
    """
    return ChurnDataset(data=load_dataset(settings.dataset_path))


DatasetDep = Annotated[ChurnDataset, Depends(get_dataset)]


def get_split(dataset: DatasetDep, settings: SettingsDep) -> DatasetSplit:
    """
    Prepare the dataset and split it into train and test parts

    Responds with 409 Conflict when the dataset is empty, has only one
    churn class or has too few rows for a stratified split.

    :dataset: ChurnDataset - training dataset
    :settings: Settings - application settings with the split parameters

    :return: stratified train/test split
    """
    if dataset.is_empty:
        raise EmptyDatasetError()

    features, target = prepare_data(dataset.data)
    if target.nunique() != 2:
        raise NotEnoughDataError(
            "Dataset must contain both churn classes",
            details={"classes": [int(label) for label in target.unique()]},
        )

    try:
        return split_dataset(
            features,
            target,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )
    except ValueError as error:
        raise NotEnoughDataError(
            "Not enough rows to split the dataset",
            details={"rows": len(target)},
        ) from error


SplitDep = Annotated[DatasetSplit, Depends(get_split)]


def get_model_store(request: Request) -> ModelStore:
    """
    Give access to the model store shared by the whole application

    :request: Request - current request

    :return: application model store
    """
    return request.app.state.model_store


ModelStoreDep = Annotated[ModelStore, Depends(get_model_store)]


def get_training_history(request: Request) -> TrainingHistory:
    """
    Give access to the training history shared by the whole application

    :request: Request - current request

    :return: application training history
    """
    return request.app.state.training_history


TrainingHistoryDep = Annotated[TrainingHistory, Depends(get_training_history)]
