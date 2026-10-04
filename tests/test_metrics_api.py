from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from itg_fastapi_churn.ml.history import TrainingHistory

FOREST = {"model_type": "random_forest", "hyperparameters": {"max_depth": 2}}


def test_metrics_without_trainings(client: TestClient) -> None:
    response = client.get("/model/metrics")

    assert response.status_code == 200
    assert response.json() == {
        "total": 0,
        "latest": None,
        "best": None,
        "recent": [],
    }


@pytest.mark.usefixtures("train_ready_app")
def test_metrics_show_latest_training(client: TestClient) -> None:
    client.post("/model/train")
    trained = client.post("/model/train", json=FOREST).json()
    status = client.get("/model/status").json()

    body = client.get("/model/metrics").json()

    assert body["total"] == 2
    assert body["latest"] == {
        "trained_at": status["trained_at"],
        "model_type": "random_forest",
        "hyperparameters": status["hyperparameters"],
        "metrics": {
            "accuracy": trained["accuracy"],
            "f1": trained["f1"],
            "roc_auc": trained["roc_auc"],
        },
    }
    assert [item["model_type"] for item in body["recent"]] == [
        "random_forest",
        "logreg",
    ]


@pytest.mark.usefixtures("train_ready_app")
def test_metrics_filter_by_model_type(client: TestClient) -> None:
    client.post("/model/train")
    client.post("/model/train", json=FOREST)

    body = client.get("/model/metrics", params={"model_type": "logreg"}).json()

    assert body["total"] == 1
    assert body["latest"]["model_type"] == "logreg"
    assert body["best"]["model_type"] == "logreg"


@pytest.mark.usefixtures("train_ready_app")
def test_metrics_limit_recent_trainings(client: TestClient) -> None:
    for _ in range(3):
        client.post("/model/train")

    body = client.get("/model/metrics", params={"limit": 2}).json()

    assert body["total"] == 3
    assert len(body["recent"]) == 2
    assert body["recent"][0] == body["latest"]


@pytest.mark.parametrize("limit", [0, 101, "many"])
def test_metrics_reject_bad_limit(
    client: TestClient, limit: int | str
) -> None:
    response = client.get("/model/metrics", params={"limit": limit})

    assert response.status_code == 422
    assert response.json()["details"][0]["location"] == ["query", "limit"]


def test_metrics_hide_server_path_when_history_is_unreadable(
    app: FastAPI, tmp_path: Path
) -> None:
    app.state.training_history = TrainingHistory(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/model/metrics")

    assert response.status_code == 500
    assert response.json() == {
        "code": "history_unavailable",
        "message": "Training history cannot be read",
        "details": None,
    }
