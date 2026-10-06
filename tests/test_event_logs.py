import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itg_fastapi_churn.core.errors import ModelNotTrainedError
from itg_fastapi_churn.core.logging_config import SERVICE_LOGGER
from itg_fastapi_churn.dataset.loader import load_dataset
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.ml.store import ModelStore
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES

SECRET_FEE = 123.45
CLIENT = {**EXAMPLE_FEATURES, "monthly_fee": SECRET_FEE}
LEAVING_CLIENT = {
    **EXAMPLE_FEATURES,
    "usage_hours": 1.0,
    "support_requests": 6,
    "account_age_months": 1,
    "failed_payments": 4,
    "autopay_enabled": 0,
}


@pytest.fixture(autouse=True)
def _capture_service_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=SERVICE_LOGGER)


def messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def test_dataset_load_is_logged(
    sample_dataset_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    load_dataset(sample_dataset_path)

    assert messages(caplog, logging.INFO) == [
        f"Dataset loaded from {sample_dataset_path}: 80 rows"
    ]


def test_empty_dataset_load_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "churn.csv"
    path.write_text("")

    load_dataset(path)

    assert messages(caplog, logging.INFO) == [
        f"Dataset loaded from {path}: the file is empty"
    ]


@pytest.mark.usefixtures("sample_app")
def test_training_start_and_result_are_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    client.post("/model/train", json={"model_type": "random_forest"})

    loaded, started, trained = messages(caplog, logging.INFO)
    assert loaded.startswith("Dataset loaded from")
    assert started == "Training random_forest model"
    assert trained.startswith("Model random_forest trained in ")
    assert "accuracy=" in trained
    assert "roc_auc=" in trained


@pytest.mark.usefixtures("sample_app")
def test_prediction_is_logged_without_client_data(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    client.post("/model/train")
    caplog.clear()

    response = client.post("/predict", json=[CLIENT, LEAVING_CLIENT])

    leaving = sum(item["churn"] for item in response.json()["predictions"])
    assert leaving == 1
    [predicted] = messages(caplog, logging.INFO)
    assert predicted.startswith("Predicted churn for 2 clients in ")
    assert predicted.endswith(f" ms: {leaving} likely to leave")
    assert str(SECRET_FEE) not in caplog.text


def test_service_error_is_logged_as_warning(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    client.post("/predict", json=EXAMPLE_FEATURES)

    assert messages(caplog, logging.WARNING) == [
        "POST /predict failed with model_not_trained: "
        + ModelNotTrainedError.default_message
    ]


def test_invalid_request_is_logged_without_its_data(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    client.post("/predict", json={**CLIENT, "region": 5, "device_type": 6})

    assert messages(caplog, logging.WARNING) == [
        "POST /predict rejected: 2 invalid fields"
    ]
    assert str(SECRET_FEE) not in caplog.text


def test_model_load_on_start_is_logged(
    tmp_path: Path,
    trained_model: TrainedModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "model.joblib"
    ModelStore(path).save(trained_model)

    ModelStore(path).load()

    assert messages(caplog, logging.INFO) == [
        f"Model logreg trained at 2026-01-01T00:00:00+00:00 loaded from {path}"
    ]


def test_missing_saved_model_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "model.joblib"

    ModelStore(path).load()

    assert messages(caplog, logging.INFO) == [
        f"No saved model at {path}, train one first"
    ]


def test_damaged_model_is_not_reported_as_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not a model")

    ModelStore(path).load()

    assert messages(caplog, logging.INFO) == []
    assert messages(caplog, logging.WARNING) == ["Saved model is ignored"]
