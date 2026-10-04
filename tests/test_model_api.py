import logging
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from itg_fastapi_churn.api.dependencies import get_dataset
from itg_fastapi_churn.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.ml.history import TrainingHistory
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES


def test_train_model_returns_metrics(
    app: FastAPI, client: TestClient, training_dataset: ChurnDataset
) -> None:
    app.dependency_overrides[get_dataset] = lambda: training_dataset
    app.dependency_overrides[get_settings] = lambda: Settings(
        test_size=0.25, random_state=0
    )

    response = client.post("/model/train")

    assert response.status_code == 200
    assert response.json()["accuracy"] == pytest.approx(1.0)
    assert response.json()["f1"] == pytest.approx(1.0)


def test_train_model_rejects_empty_dataset(
    app: FastAPI, client: TestClient
) -> None:
    app.dependency_overrides[get_dataset] = lambda: ChurnDataset(
        pd.DataFrame()
    )

    response = client.post("/model/train")

    assert response.status_code == 409
    assert response.json()["code"] == "dataset_empty"


def test_train_model_returns_404_when_file_missing(
    app: FastAPI, client: TestClient, tmp_path: Path
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        dataset_path=tmp_path / "missing.csv"
    )

    response = client.post("/model/train")

    assert response.status_code == 404
    assert response.json()["code"] == "dataset_not_found"


def test_train_model_rejects_single_class_dataset(
    app: FastAPI, client: TestClient
) -> None:
    rows = [
        {**EXAMPLE_FEATURES, "churn": churn, "failed_payments": churn * 5}
        for churn in [0] * 20
    ]
    app.dependency_overrides[get_dataset] = lambda: ChurnDataset(
        pd.DataFrame(rows)
    )

    response = client.post("/model/train")

    assert response.status_code == 409
    assert response.json()["code"] == "not_enough_data"
    assert "both churn classes" in response.json()["message"]


def test_train_model_rejects_too_small_dataset(
    app: FastAPI, client: TestClient, dataset: ChurnDataset
) -> None:
    app.dependency_overrides[get_dataset] = lambda: dataset

    response = client.post("/model/train")

    assert response.status_code == 409
    assert response.json()["code"] == "not_enough_data"
    assert "Not enough rows" in response.json()["message"]


def test_status_reports_untrained_model(client: TestClient) -> None:
    response = client.get("/model/status")

    assert response.status_code == 200
    assert response.json() == {
        "is_trained": False,
        "trained_at": None,
        "metrics": None,
        "model_type": None,
        "hyperparameters": None,
    }


def test_train_model_saves_model_and_updates_status(
    app: FastAPI,
    client: TestClient,
    training_dataset: ChurnDataset,
    tmp_path: Path,
) -> None:
    app.dependency_overrides[get_dataset] = lambda: training_dataset
    app.dependency_overrides[get_settings] = lambda: Settings(
        test_size=0.25, random_state=0
    )

    trained = client.post("/model/train").json()
    status = client.get("/model/status").json()

    assert (tmp_path / "model.joblib").exists()
    assert status["is_trained"]
    assert status["trained_at"] is not None
    assert status["metrics"] == {
        "accuracy": trained["accuracy"],
        "f1": trained["f1"],
        "roc_auc": trained["roc_auc"],
    }


def test_failed_training_keeps_model_untrained(
    app: FastAPI, client: TestClient
) -> None:
    app.dependency_overrides[get_dataset] = lambda: ChurnDataset(
        pd.DataFrame()
    )

    client.post("/model/train")
    status = client.get("/model/status").json()

    assert not status["is_trained"]


@pytest.mark.usefixtures("train_ready_app")
def test_train_without_body_uses_logreg(client: TestClient) -> None:
    client.post("/model/train")
    status = client.get("/model/status").json()

    assert status["model_type"] == "logreg"
    assert status["hyperparameters"]["class_weight"] == "balanced"


@pytest.mark.usefixtures("train_ready_app")
def test_train_random_forest_with_hyperparameters(client: TestClient) -> None:
    config = {
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 10, "max_depth": 3},
    }

    response = client.post("/model/train", json=config)
    status = client.get("/model/status").json()

    assert response.status_code == 200
    assert status["model_type"] == "random_forest"
    assert status["hyperparameters"]["n_estimators"] == 10
    assert status["hyperparameters"]["max_depth"] == 3


@pytest.mark.parametrize(
    ("hyperparameters", "wrong_name"),
    [({"n_trees": 5}, "n_trees"), ({"C": -1}, "'C'")],
)
@pytest.mark.usefixtures("train_ready_app")
def test_train_rejects_bad_hyperparameters(
    client: TestClient,
    hyperparameters: dict[str, int],
    wrong_name: str,
) -> None:
    response = client.post(
        "/model/train", json={"hyperparameters": hyperparameters}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_hyperparameters"
    assert wrong_name in response.json()["message"]
    assert not client.get("/model/status").json()["is_trained"]


@pytest.mark.usefixtures("train_ready_app")
def test_train_rejects_unknown_model_type(client: TestClient) -> None:
    response = client.post("/model/train", json={"model_type": "svm"})

    assert response.status_code == 422


def test_train_docs_show_config_examples(client: TestClient) -> None:
    operation = client.get("/openapi.json").json()["paths"]["/model/train"]
    body = operation["post"]["requestBody"]["content"]["application/json"]

    assert set(body["examples"]) == {"logreg", "random_forest"}


@pytest.mark.usefixtures("train_ready_app")
def test_train_returns_training_warnings(client: TestClient) -> None:
    response = client.post(
        "/model/train", json={"hyperparameters": {"max_iter": 1}}
    )

    assert response.status_code == 200
    [warning] = response.json()["warnings"]
    assert "failed to converge" in warning


@pytest.mark.usefixtures("train_ready_app")
def test_train_without_problems_returns_no_warnings(
    client: TestClient,
) -> None:
    response = client.post("/model/train")

    assert response.json()["warnings"] == []


@pytest.mark.usefixtures("train_ready_app")
def test_train_adds_record_to_history(
    app: FastAPI, client: TestClient
) -> None:
    config = {
        "model_type": "random_forest",
        "hyperparameters": {"n_estimators": 10},
    }

    trained = client.post("/model/train", json=config).json()
    status = client.get("/model/status").json()

    [record] = app.state.training_history.records()
    assert record.trained_at.isoformat() == status["trained_at"].replace(
        "Z", "+00:00"
    )
    assert record.model_type == "random_forest"
    assert record.hyperparameters == status["hyperparameters"]
    assert record.metrics.model_dump() == {
        "accuracy": trained["accuracy"],
        "f1": trained["f1"],
        "roc_auc": trained["roc_auc"],
    }


@pytest.mark.usefixtures("train_ready_app")
def test_failed_training_is_not_added_to_history(
    app: FastAPI, client: TestClient
) -> None:
    client.post("/model/train", json={"hyperparameters": {"C": -1}})

    assert app.state.training_history.records() == []


@pytest.mark.usefixtures("train_ready_app")
def test_train_warns_when_history_cannot_be_written(
    app: FastAPI,
    client: TestClient,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app.state.training_history = TrainingHistory(tmp_path)

    with caplog.at_level(logging.ERROR):
        response = client.post("/model/train")

    assert response.status_code == 200
    assert response.json()["warnings"] == [
        "Training was not added to the history"
    ]
    assert client.get("/model/status").json()["is_trained"]
    assert "Training was not added to the history" in caplog.text


@pytest.mark.parametrize(
    "value", [b"1e309", b"NaN", b"-Infinity", b"1" + b"0" * 400]
)
@pytest.mark.usefixtures("train_ready_app")
def test_train_rejects_unsafe_number_in_hyperparameter(
    client: TestClient, value: bytes
) -> None:
    response = client.post(
        "/model/train",
        content=b'{"hyperparameters": {"C": ' + value + b"}}",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_hyperparameters"
