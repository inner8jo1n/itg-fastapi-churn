import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import JsonValue
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.config import Settings, get_settings
from itg_fastapi_churn.errors import ServiceError
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES


@pytest.fixture(autouse=True)
def _capture_error_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR)


class BrokenClassifier(DummyClassifier):
    def predict(self, X: object) -> object:
        raise RuntimeError("model internals")


def test_unknown_path_uses_common_format(client: TestClient) -> None:
    response = client.get("/no/such/path")

    assert response.status_code == 404
    assert response.json() == {
        "code": "not_found",
        "message": "Not Found",
        "details": None,
    }


def test_wrong_method_keeps_allow_header(client: TestClient) -> None:
    response = client.get("/predict")

    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"
    assert response.headers["allow"] == "POST"


def test_validation_error_lists_invalid_fields(client: TestClient) -> None:
    response = client.get("/dataset/preview", params={"n": 0})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"][0]["location"] == ["query", "n"]
    assert body["details"][0]["input"] == "0"


def test_unexpected_error_hides_traceback(
    app: FastAPI, caplog: pytest.LogCaptureFixture
) -> None:
    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "Internal server error, please try again later",
        "details": None,
    }
    assert "secret internal detail" not in response.text
    assert "secret internal detail" in caplog.text


def test_service_error_without_known_status_is_500(app: FastAPI) -> None:
    @app.get("/service-error")
    def service_error() -> None:
        raise ServiceError(details={"hint": "base class"})

    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/service-error")

    assert response.status_code == 500
    assert response.json()["code"] == "service_error"
    assert response.json()["details"] == {"hint": "base class"}


def test_service_error_uses_default_message() -> None:
    error = ServiceError()

    assert error.message == "Churn service error"
    assert str(error) == "Churn service error"


def test_invalid_dataset_is_reported(
    app: FastAPI, client: TestClient, tmp_path: Path
) -> None:
    path = tmp_path / "churn.csv"
    path.write_text("a;b\n1;2")
    app.dependency_overrides[get_settings] = lambda: Settings(
        dataset_path=path
    )

    response = client.get("/dataset/info")

    assert response.status_code == 409
    assert response.json()["code"] == "dataset_invalid"
    assert "missing_columns" in response.json()["details"]


def test_model_failure_is_reported_as_prediction_failed(
    app: FastAPI,
    trained_model: TrainedModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    features = pd.DataFrame([EXAMPLE_FEATURES, EXAMPLE_FEATURES])
    broken = Pipeline([("classifier", BrokenClassifier())]).fit(
        features, pd.Series([0, 1])
    )
    app.state.model_store.save(replace(trained_model, pipeline=broken))
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/predict", json=EXAMPLE_FEATURES)

    assert response.status_code == 500
    assert response.json()["code"] == "prediction_failed"
    assert "model internals" not in response.text
    assert "model internals" in caplog.text


def test_rejected_list_is_not_echoed_back(trained_client: TestClient) -> None:
    response = trained_client.post("/predict", json=[EXAMPLE_FEATURES] * 1001)

    assert response.status_code == 422
    assert response.json()["details"][0]["input"] == "list of 1001 items"
    assert len(response.content) < 1000


def test_long_rejected_text_is_cut(trained_client: TestClient) -> None:
    payload = {**EXAMPLE_FEATURES, "monthly_fee": "x" * 200_000}

    response = trained_client.post("/predict", json=payload)

    [error] = response.json()["details"]
    assert error["input"].endswith("... (200000 characters)")
    assert len(response.content) < 1000


@pytest.mark.parametrize(
    "payload",
    [
        {**EXAMPLE_FEATURES, **{f"extra_{i}": i for i in range(5000)}},
        [{**EXAMPLE_FEATURES, "usage_hours": -1}] * 1000,
    ],
)
def test_many_problems_give_small_answer(
    trained_client: TestClient, payload: JsonValue
) -> None:
    response = trained_client.post("/predict", json=payload)

    body = response.json()
    assert response.status_code == 422
    assert len(body["details"]) == 20
    assert "the first 20 are listed" in body["message"]
    assert len(response.content) < 10_000


def test_echoed_object_is_shortened(trained_client: TestClient) -> None:
    payload = {
        **{k: v for k, v in EXAMPLE_FEATURES.items() if k != "region"},
        "note": "x" * 200_000,
    }

    response = trained_client.post("/predict", json=payload)

    assert response.status_code == 422
    assert len(response.content) < 2_000


def test_unknown_http_status_gets_generic_code(app: FastAPI) -> None:
    @app.get("/teapot")
    def teapot() -> None:
        raise HTTPException(status_code=599, detail="custom")

    response = TestClient(app).get("/teapot")

    assert response.status_code == 599
    assert response.json()["code"] == "http_error"
