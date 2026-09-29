from typing import Annotated

from fastapi import APIRouter, Query

from itg_fastapi_churn.api.dependencies import DatasetDep
from itg_fastapi_churn.schemas.churn import DatasetRowChurn
from itg_fastapi_churn.schemas.dataset import DatasetInfo

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def preview_dataset(
    dataset: DatasetDep,
    n: Annotated[int, Query(ge=1, le=100)] = 5,
) -> list[DatasetRowChurn]:
    """
    Return the first rows of the training dataset

    :dataset: ChurnDataset - training dataset
    :n: int - number of rows to return, from 1 to 100

    :return: first n dataset rows
    """
    return dataset.preview(n)


@router.get("/info")
def get_dataset_info(dataset: DatasetDep) -> DatasetInfo:
    """
    Summarize the training dataset

    :dataset: ChurnDataset - training dataset

    :return: dataset size, feature names and churn class balance
    """
    return dataset.info()
