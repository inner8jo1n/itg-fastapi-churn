from pathlib import Path

import pytest

from itg_fastapi_churn.dataset.loader import load_dataset
from itg_fastapi_churn.errors import DatasetNotFoundError, InvalidDatasetError

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

    with pytest.raises(InvalidDatasetError) as raised:
        load_dataset(path)

    assert raised.value.details == {
        "invalid_rows": 1,
        "examples": [
            {
                "row": 1,
                "errors": [
                    {
                        "field": "churn",
                        "message": "Input should be less than or equal to 1",
                    }
                ],
            }
        ],
    }


def test_load_dataset_raises_for_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"

    with pytest.raises(DatasetNotFoundError):
        load_dataset(path)


def test_load_dataset_keeps_columns_when_no_rows(tmp_path: Path) -> None:
    path = write_csv(tmp_path)

    df = load_dataset(path)

    assert len(df) == 0
    assert list(df.columns) == HEADER.split(",")


def test_load_dataset_handles_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("")

    df = load_dataset(path)

    assert len(df) == 0
    assert list(df.columns) == HEADER.split(",")


def test_load_dataset_ignores_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "churn.csv"
    path.write_text(f"client_id,{HEADER}\n7,{VALID_ROW}")

    df = load_dataset(path)

    assert list(df.columns) == HEADER.split(",")
    assert len(df) == 1


def test_load_dataset_reports_only_first_invalid_rows(tmp_path: Path) -> None:
    bad_row = VALID_ROW[:-1] + "5"
    path = write_csv(tmp_path, *[bad_row] * 8)

    with pytest.raises(InvalidDatasetError) as raised:
        load_dataset(path)

    details = raised.value.details
    assert isinstance(details, dict)
    assert details["invalid_rows"] == 8
    assert isinstance(details["examples"], list)
    assert len(details["examples"]) == 5


def test_load_dataset_reports_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "churn.csv"
    path.write_text("a;b\n1;2")

    with pytest.raises(InvalidDatasetError, match="misses columns") as raised:
        load_dataset(path)

    details = raised.value.details
    assert isinstance(details, dict)
    assert details["missing_columns"] == HEADER.split(",")


def test_load_dataset_rejects_malformed_csv(tmp_path: Path) -> None:
    path = write_csv(tmp_path, VALID_ROW, VALID_ROW + ",9,9")

    with pytest.raises(InvalidDatasetError, match="not a valid CSV"):
        load_dataset(path)


def test_load_dataset_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(InvalidDatasetError, match="cannot be read"):
        load_dataset(tmp_path)


def test_load_dataset_rejects_binary_file(tmp_path: Path) -> None:
    path = tmp_path / "churn.csv"
    path.write_bytes(bytes(range(128, 256)) * 10)

    with pytest.raises(InvalidDatasetError, match="not a valid CSV"):
        load_dataset(path)
