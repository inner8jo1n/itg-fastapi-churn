import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import JsonValue

from itg_fastapi_churn.main import create_app
from itg_fastapi_churn.ml.features import FEATURE_COLUMNS
from itg_fastapi_churn.ml.history import TrainingHistory
from itg_fastapi_churn.ml.store import ModelStore

CONFIGS = [
    pytest.param({"model_type": "logreg"}, id="logreg"),
    pytest.param(
        {"model_type": "random_forest", "hyperparameters": {"max_depth": 4}},
        id="random_forest",
    ),
]


@pytest.fixture
def clients(sample_dataset_path: Path) -> list[JsonValue]:
    features = pd.read_csv(sample_dataset_path)[list(FEATURE_COLUMNS)]
    return json.loads(features.head(5).to_json(orient="records"))


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.usefixtures("sample_app")
def test_read_train_status_predict(
    client: TestClient,
    clients: list[JsonValue],
    config: dict[str, JsonValue],
) -> None:
    info = client.get("/dataset/info").json()
    trained = client.post("/model/train", json=config)
    status = client.get("/model/status").json()
    one = client.post("/predict", json=clients[0])
    several = client.post("/predict", json=clients)

    assert info["n_rows"] == 80
    assert info["churn_distribution"] == {"0": 64, "1": 16}
    assert trained.status_code == 200
    assert status["is_trained"]
    assert status["model_type"] == config["model_type"]
    assert status["metrics"] == {
        name: trained.json()[name] for name in ("accuracy", "f1", "roc_auc")
    }
    assert one.status_code == 200
    assert several.status_code == 200
    [single] = one.json()["predictions"]
    first, *_ = several.json()["predictions"]
    assert len(several.json()["predictions"]) == len(clients)
    assert first["churn"] == single["churn"]
    assert first["probabilities"] == pytest.approx(single["probabilities"])


def test_predictions_match_the_trained_pipeline(
    sample_app: FastAPI, client: TestClient, clients: list[JsonValue]
) -> None:
    client.post("/model/train")

    predictions = client.post("/predict", json=clients).json()["predictions"]

    pipeline = sample_app.state.model_store.current.pipeline
    expected = pipeline.predict_proba(pd.DataFrame(clients))[:, 1]
    churn_probabilities = [item["probabilities"]["1"] for item in predictions]
    assert churn_probabilities == pytest.approx(list(expected))


@pytest.mark.usefixtures("sample_app")
def test_saved_model_works_after_restart(
    client: TestClient,
    clients: list[JsonValue],
    tmp_path: Path,
) -> None:
    client.post("/model/train")
    status = client.get("/model/status").json()
    predictions = client.post("/predict", json=clients).json()

    restarted = create_app()
    restarted.state.model_store = ModelStore(tmp_path / "model.joblib")
    restarted.state.training_history = TrainingHistory(
        tmp_path / "history.jsonl"
    )
    with TestClient(restarted) as restarted_client:
        restarted_status = restarted_client.get("/model/status").json()
        restarted_predictions = restarted_client.post(
            "/predict", json=clients
        ).json()

    assert restarted_status == status
    assert restarted_predictions == predictions
