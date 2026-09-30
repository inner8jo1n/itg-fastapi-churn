from pathlib import Path

from fastapi.testclient import TestClient

from itg_fastapi_churn.main import create_app
from itg_fastapi_churn.ml.persistence import TrainedModel, save_churn_model
from itg_fastapi_churn.ml.store import ModelStore


def test_app_loads_saved_model_on_startup(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.joblib"
    save_churn_model(trained_model, path)
    app = create_app()
    app.state.model_store = ModelStore(path)

    with TestClient(app) as client:
        response = client.get("/model/status")

    assert response.json()["is_trained"]
    assert response.json()["metrics"] == trained_model.metrics.model_dump()


def test_app_starts_without_saved_model(tmp_path: Path) -> None:
    app = create_app()
    app.state.model_store = ModelStore(tmp_path / "missing.joblib")

    with TestClient(app) as client:
        response = client.get("/model/status")

    assert response.status_code == 200
    assert response.json() == {
        "is_trained": False,
        "trained_at": None,
        "metrics": None,
    }


def test_app_starts_with_damaged_model_file(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not a model")
    app = create_app()
    app.state.model_store = ModelStore(path)

    with TestClient(app) as client:
        response = client.get("/model/status")

    assert response.status_code == 200
    assert not response.json()["is_trained"]
