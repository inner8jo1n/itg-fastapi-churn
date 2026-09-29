from pathlib import Path

import pytest
from pydantic import ValidationError

from itg_fastapi_churn.dataset.loader import load_dataset

HEADER = (
    "monthly_fee,usage_hours,"
    "support_requests,account_age_months,"
    "failed_payments,region,device_type,"
    "payment_method,autopay_enabled,churn"
)
VALID_ROW = "19.99,42.5,1,14,0,europe,mobile,card,1,0"


def write_csv(directory: Path, *rows: str) -> Path:
    content = "\n".join([HEADER, *rows])

    path = directory / "churn.csv"
    path.write_text(content)

    return path


def test_load_dataset_returns_validated_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path, VALID_ROW, VALID_ROW)

    df = load_dataset(path)

    assert len(df) == 2
    assert list(df.columns) == HEADER.split(",")


def test_load_dataset_rejects_invalid_row(tmp_path: Path) -> None:
    path = write_csv(tmp_path, VALID_ROW[:-1] + "5")

    with pytest.raises(ValidationError):
        load_dataset(path)


def test_load_dataset_raises_for_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_dataset(path)
