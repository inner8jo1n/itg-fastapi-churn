from pydantic import BaseModel


class DatasetInfo(BaseModel):
    """
    Summary of the training dataset

    :n_rows: int - number of rows
    :n_columns: int - number of columns including the churn target
    :feature_names: list[str] - feature column names without the target
    :churn_distribution: dict[int, int] - number of rows per churn class
    """

    n_rows: int
    n_columns: int
    feature_names: list[str]
    churn_distribution: dict[int, int]


class SplitInfo(BaseModel):
    """
    Summary of the train/test split

    :train_rows: int - number of rows in the train part
    :test_rows: int - number of rows in the test part
    :train_churn_distribution: dict[int, int] - rows per churn class in train
    :test_churn_distribution: dict[int, int] - rows per churn class in test
    """

    train_rows: int
    test_rows: int
    train_churn_distribution: dict[int, int]
    test_churn_distribution: dict[int, int]
