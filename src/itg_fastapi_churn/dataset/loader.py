from pathlib import Path

import pandas as pd
from pydantic import JsonValue, ValidationError

from itg_fastapi_churn.errors import DatasetNotFoundError, InvalidDatasetError
from itg_fastapi_churn.schemas.churn import DatasetRowChurn

COLUMNS = list(DatasetRowChurn.model_fields)
MAX_REPORTED_ROWS = 5


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Read the churn dataset CSV and validate every row as DatasetRowChurn

    An empty file or a file with only the header gives an empty dataset
    that still has all the columns. A missing file raises
    DatasetNotFoundError. A file that cannot be read, is not a valid CSV,
    lacks columns or has invalid rows raises InvalidDatasetError that
    explains what is wrong.

    :path: Path - path to the dataset CSV file

    :return: validated dataset with columns ordered as in DatasetRowChurn
    """
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)
    except FileNotFoundError as error:
        raise DatasetNotFoundError() from error
    except OSError as error:
        raise InvalidDatasetError(
            "Dataset file cannot be read", details={"reason": error.strerror}
        ) from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise InvalidDatasetError(
            "Dataset file is not a valid CSV", details={"reason": str(error)}
        ) from error

    missing = [column for column in COLUMNS if column not in raw.columns]
    if missing:
        raise InvalidDatasetError(
            "Dataset file misses columns", details={"missing_columns": missing}
        )

    rows: list[DatasetRowChurn] = []
    problems: list[JsonValue] = []
    for number, record in enumerate(raw.to_dict(orient="records"), start=1):
        try:
            rows.append(DatasetRowChurn.model_validate(record))
        except ValidationError as error:
            problems.append({"row": number, "errors": _describe(error)})

    if problems:
        raise InvalidDatasetError(
            f"Dataset has invalid rows: {len(problems)}",
            details={
                "invalid_rows": len(problems),
                "examples": problems[:MAX_REPORTED_ROWS],
            },
        )

    return pd.DataFrame([row.model_dump() for row in rows], columns=COLUMNS)


def _describe(error: ValidationError) -> list[JsonValue]:
    """
    Turn a row validation error into short field messages

    :error: ValidationError - error raised for one dataset row

    :return: one entry per invalid field
    """
    return [
        {"field": ".".join(map(str, item["loc"])), "message": item["msg"]}
        for item in error.errors()
    ]
