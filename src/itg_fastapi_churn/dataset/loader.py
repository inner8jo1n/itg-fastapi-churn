import logging
from pathlib import Path

import pandas as pd
from pydantic import JsonValue, ValidationError

from itg_fastapi_churn.core.errors import (
    DatasetNotFoundError,
    EmptyDatasetError,
    InvalidDatasetError,
)
from itg_fastapi_churn.schemas.churn import DatasetRowChurn

COLUMNS = list(DatasetRowChurn.model_fields)
MAX_REPORTED_ROWS = 5

logger = logging.getLogger(__name__)


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
    raw = _read_csv(path)
    if raw is None:
        logger.info("Dataset loaded from %s: the file is empty", path)
        return pd.DataFrame(columns=COLUMNS)
    _require_columns(raw)

    rows = _validate_rows(raw)
    logger.info("Dataset loaded from %s: %d rows", path, len(rows))
    return pd.DataFrame([row.model_dump() for row in rows], columns=COLUMNS)


def check_dataset(path: Path) -> None:
    """
    Check cheaply that the dataset can be used for training

    Only the header and the first row are read and validated, so the
    check is fast even for a large file; other rows are validated on
    loading. Raises
    the same errors as load_dataset and EmptyDatasetError for a file
    without rows.

    :path: Path - path to the dataset CSV file
    """
    raw = _read_csv(path, nrows=1)
    if raw is None:
        raise EmptyDatasetError()
    _require_columns(raw)
    if raw.empty:
        raise EmptyDatasetError()
    _validate_rows(raw)


def _read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    """
    Read the CSV file, turning read problems into service errors

    :path: Path - path to the dataset CSV file
    :nrows: int | None - read only this many rows, all if not given

    :return: raw table, or None if the file is completely empty
    """
    try:
        return pd.read_csv(path, nrows=nrows)
    except pd.errors.EmptyDataError:
        return None
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


def _require_columns(raw: pd.DataFrame) -> None:
    """
    Make sure the table has every dataset column

    :raw: pd.DataFrame - table read from the file
    """
    missing = [column for column in COLUMNS if column not in raw.columns]
    if missing:
        raise InvalidDatasetError(
            "Dataset file misses columns", details={"missing_columns": missing}
        )


def _validate_rows(raw: pd.DataFrame) -> list[DatasetRowChurn]:
    """
    Validate every row of the table as DatasetRowChurn

    Raises InvalidDatasetError that shows the first invalid rows.

    :raw: pd.DataFrame - table read from the file

    :return: validated rows in file order
    """
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
    return rows


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
