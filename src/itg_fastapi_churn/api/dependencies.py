from typing import Annotated

from fastapi import Depends, HTTPException, status

from itg_fastapi_churn.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.dataset.loader import load_dataset


def get_dataset(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChurnDataset:
    """
    Load the training dataset from the path set in the settings

    :settings: Settings - application settings with the dataset path

    :return: loaded churn dataset
    """
    try:
        data = load_dataset(settings.dataset_path)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found",
        ) from error

    return ChurnDataset(data=data)


DatasetDep = Annotated[ChurnDataset, Depends(get_dataset)]
