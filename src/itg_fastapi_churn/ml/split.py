from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from itg_fastapi_churn.ml.features import class_distribution
from itg_fastapi_churn.schemas.dataset import SplitInfo


@dataclass(frozen=True)
class DatasetSplit:
    """
    Train and test parts of the dataset

    :x_train: pd.DataFrame - features used for training
    :x_test: pd.DataFrame - features held out for evaluation
    :y_train: pd.Series - churn target for training
    :y_test: pd.Series - churn target for evaluation
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    def info(self) -> SplitInfo:
        """
        Summarize the sizes and churn class balance of both parts

        :return: split summary
        """
        return SplitInfo(
            train_rows=len(self.x_train),
            test_rows=len(self.x_test),
            train_churn_distribution=class_distribution(self.y_train),
            test_churn_distribution=class_distribution(self.y_test),
        )


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
) -> DatasetSplit:
    """
    Split features and target into stratified train and test parts

    :features: pd.DataFrame - feature matrix
    :target: pd.Series - churn target
    :test_size: float - share of rows held out for the test part
    :random_state: int - seed that makes the split reproducible

    :return: train and test parts with the same churn class balance
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return DatasetSplit(
        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )
