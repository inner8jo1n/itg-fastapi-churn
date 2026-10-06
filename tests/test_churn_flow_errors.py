import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from itg_fastapi_churn.core.config import Settings, get_settings
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES


@pytest.fixture
def dataset_copy(
    app: FastAPI, sample_dataset_path: Path, tmp_path: Path
) -> Path:
    path = tmp_path / "churn.csv"
    shutil.copy(sample_dataset_path, path)
    app.dependency_overrides[get_settings] = lambda: Settings(
        dataset_path=path
    )
    return path


def delete_dataset(path: Path) -> None:
    path.unlink()


def break_dataset(path: Path) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write("9.99,10,1,5,0,europe,mobile,card,1,7\n")


def empty_dataset(path: Path) -> None:
    path.write_text("")


@pytest.mark.usefixtures("dataset_copy")
def test_predict_fails_before_training_and_works_after(
    client: TestClient,
) -> None:
    before = client.post("/predict", json=EXAMPLE_FEATURES)
    client.post("/model/train")
    after = client.post("/predict", json=EXAMPLE_FEATURES)

    assert before.status_code == 409
    assert before.json()["code"] == "model_not_trained"
    assert after.status_code == 200


@pytest.mark.parametrize(
    ("spoil", "status_code", "code"),
    [
        (delete_dataset, 404, "dataset_not_found"),
        (break_dataset, 409, "dataset_invalid"),
        (empty_dataset, 409, "dataset_empty"),
    ],
)
def test_failed_training_keeps_previous_model(
    client: TestClient,
    dataset_copy: Path,
    spoil: Callable[[Path], None],
    status_code: int,
    code: str,
) -> None:
    client.post("/model/train")
    status = client.get("/model/status").json()
    prediction = client.post("/predict", json=EXAMPLE_FEATURES).json()
    spoil(dataset_copy)

    failed = client.post("/model/train")

    assert failed.status_code == status_code
    assert failed.json()["code"] == code
    assert client.get("/model/status").json() == status
    assert client.post("/predict", json=EXAMPLE_FEATURES).json() == prediction


@pytest.mark.usefixtures("dataset_copy")
def test_rejected_hyperparameters_keep_previous_model(
    client: TestClient,
) -> None:
    client.post("/model/train")
    status = client.get("/model/status").json()

    failed = client.post("/model/train", json={"hyperparameters": {"C": -1}})

    assert failed.status_code == 422
    assert client.get("/model/status").json() == status


@pytest.mark.usefixtures("dataset_copy")
def test_invalid_prediction_request_does_not_break_model(
    client: TestClient,
) -> None:
    client.post("/model/train")

    rejected = client.post(
        "/predict", json={**EXAMPLE_FEATURES, "monthly_fee": "a lot"}
    )
    accepted = client.post("/predict", json=EXAMPLE_FEATURES)

    assert rejected.status_code == 422
    assert rejected.json()["code"] == "validation_error"
    assert accepted.status_code == 200
