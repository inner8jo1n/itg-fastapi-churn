from typing import cast

import pandas as pd

from itg_fastapi_churn.schemas.churn import DatasetRowChurn
from itg_fastapi_churn.schemas.dataset import DatasetInfo

TARGET_COLUMN = "churn"


class ChurnDataset:
    """
    Read-only view over a validated churn dataset
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Wrap an already validated dataset

        :data: pd.DataFrame - dataset rows matching DatasetRowChurn
        """
        self._data = data

    def preview(self, n: int) -> list[DatasetRowChurn]:
        """
        Return the first rows of the dataset

        :n: int - number of rows to return

        :return: first n rows, or all rows if the dataset is smaller
        """
        head = self._data.head(n)
        return [
            DatasetRowChurn.model_validate(row)
            for row in head.to_dict(orient="records")
        ]

    def info(self) -> DatasetInfo:
        """
        Summarize the dataset size, features and churn class balance

        :return: dataset summary
        """
        n_rows, n_columns = self._data.shape
        feature_names = [
            column for column in self._data.columns if column != TARGET_COLUMN
        ]
        churn_distribution = cast(
            dict[int, int],
            self._data[TARGET_COLUMN].value_counts().sort_index().to_dict(),
        )

        return DatasetInfo(
            n_rows=n_rows,
            n_columns=n_columns,
            feature_names=feature_names,
            churn_distribution=churn_distribution,
        )
