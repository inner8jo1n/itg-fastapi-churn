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
