from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from itg_fastapi_churn.core.config import Settings, get_settings
from itg_fastapi_churn.main import create_app
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.ml.store import ModelStore


@pytest.mark.usefixtures("sample_app")
def test_health_is_ok_with_model_and_dataset(client: TestClient) -> None:
    client.post("/model/train")
    status = client.get("/model/status").json()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": {
            "available": True,
            "model_type": "logreg",
            "trained_at": status["trained_at"],
        },
        "dataset": {"available": True, "problem": None},
    }


@pytest.mark.usefixtures("sample_app")
def test_health_is_degraded_without_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model"] == {
        "available": False,
        "model_type": None,
        "trained_at": None,
    }
    assert response.json()["dataset"]["available"]


@pytest.mark.parametrize(
    ("content", "problem"),
    [(None, "dataset_not_found"), ("", "dataset_empty")],
)
def test_health_is_degraded_without_dataset(
    app: FastAPI,
    client: TestClient,
    trained_model: TrainedModel,
    tmp_path: Path,
    content: str | None,
    problem: str,
) -> None:
    path = tmp_path / "churn.csv"
    if content is not None:
        path.write_text(content)
    app.dependency_overrides[get_settings] = lambda: Settings(
        dataset_path=path
    )
    app.state.model_store.save(trained_model)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model"]["available"]
    assert response.json()["dataset"] == {
        "available": False,
        "problem": problem,
    }


def test_health_sees_model_loaded_on_start(
    trained_model: TrainedModel, sample_dataset_path: Path, tmp_path: Path
) -> None:
    ModelStore(tmp_path / "model.joblib").save(trained_model)
    application = create_app()
    application.state.model_store = ModelStore(tmp_path / "model.joblib")
    application.dependency_overrides[get_settings] = lambda: Settings(
        dataset_path=sample_dataset_path
    )

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.json()["status"] == "ok"
