from pathlib import Path

import pandas as pd

from itg_fastapi_churn.schemas.churn import DatasetRowChurn

COLUMNS = list(DatasetRowChurn.model_fields)


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Read the churn dataset CSV and validate every row as DatasetRowChurn

    An empty file or a file with only the header gives an empty dataset
    that still has all the columns.

    :path: Path - path to the dataset CSV file

    :return: validated dataset with columns ordered as in DatasetRowChurn
    """
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)

    rows = [
        DatasetRowChurn.model_validate(row)
        for row in raw.to_dict(orient="records")
    ]

    return pd.DataFrame([row.model_dump() for row in rows], columns=COLUMNS)
