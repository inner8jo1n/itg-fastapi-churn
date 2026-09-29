import pandas as pd

from itg_fastapi_churn.ml.features import TARGET_COLUMN, class_distribution
from itg_fastapi_churn.schemas.churn import DatasetRowChurn
from itg_fastapi_churn.schemas.dataset import DatasetInfo


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

    @property
    def data(self) -> pd.DataFrame:
        """
        Copy of the dataset, so callers cannot change the wrapped data

        :return: dataset rows as a new DataFrame
        """
        return self._data.copy()

    @property
    def is_empty(self) -> bool:
        """
        Tell whether the dataset has no rows

        :return: True if there are no rows
        """
        return self._data.empty

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
        churn_distribution = class_distribution(self._data[TARGET_COLUMN])

        return DatasetInfo(
            n_rows=n_rows,
            n_columns=n_columns,
            feature_names=feature_names,
            churn_distribution=churn_distribution,
        )
