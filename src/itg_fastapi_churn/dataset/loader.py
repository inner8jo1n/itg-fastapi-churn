from pathlib import Path

import pandas as pd

from itg_fastapi_churn.schemas.churn import DatasetRowChurn


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Read the churn dataset CSV and validate every row as DatasetRowChurn

    :path: Path - path to the dataset CSV file

    :return: validated dataset with columns ordered as in DatasetRowChurn
    """
    raw = pd.read_csv(path)

    rows = [
        DatasetRowChurn.model_validate(row)
        for row in raw.to_dict(orient="records")
    ]

    return pd.DataFrame([row.model_dump() for row in rows])
