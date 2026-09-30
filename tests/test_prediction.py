import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from itg_fastapi_churn.api.dependencies import MODEL_NOT_TRAINED, get_dataset
from itg_fastapi_churn.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES
from itg_fastapi_churn.schemas.prediction import EXAMPLE_RESPONSE

LOYAL_CLIENT = {**EXAMPLE_FEATURES, "failed_payments": 0}
RISKY_CLIENT = {**EXAMPLE_FEATURES, "failed_payments": 5}


@pytest.fixture
def trained_client(
    app: FastAPI, client: TestClient, training_dataset: ChurnDataset
) -> TestClient:
    app.dependency_overrides[get_dataset] = lambda: training_dataset
    app.dependency_overrides[get_settings] = lambda: Settings(
        test_size=0.25, random_state=0
    )
    client.post("/model/train")
    return client


def test_predict_one_client(trained_client: TestClient) -> None:
    response = trained_client.post("/predict", json=RISKY_CLIENT)

    assert response.status_code == 200
    [prediction] = response.json()["predictions"]
    assert prediction["churn"] == 1
    assert set(prediction["probabilities"]) == {"0", "1"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)


def test_predict_several_clients_in_request_order(
    trained_client: TestClient,
) -> None:
    response = trained_client.post(
        "/predict", json=[RISKY_CLIENT, LOYAL_CLIENT, RISKY_CLIENT]
    )

    assert response.status_code == 200
    churn = [item["churn"] for item in response.json()["predictions"]]
    assert churn == [1, 0, 1]


def test_predict_rejects_empty_list(trained_client: TestClient) -> None:
    response = trained_client.post("/predict", json=[])

    assert response.status_code == 422


def test_predict_rejects_missing_feature(trained_client: TestClient) -> None:
    payload = {k: v for k, v in EXAMPLE_FEATURES.items() if k != "region"}

    response = trained_client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_without_trained_model(client: TestClient) -> None:
    response = client.post("/predict", json=EXAMPLE_FEATURES)

    assert response.status_code == 409
    assert response.json()["detail"] == MODEL_NOT_TRAINED


def test_predict_docs_show_request_examples(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    operation = paths["/predict"]["post"]
    body = operation["requestBody"]["content"]["application/json"]

    success = operation["responses"]["200"]["content"]["application/json"]

    assert set(body["examples"]) == {"one_client", "several_clients"}
    assert success["example"] == EXAMPLE_RESPONSE
    assert "409" in operation["responses"]


@pytest.mark.parametrize("value", ["Infinity", "NaN"])
def test_predict_rejects_non_finite_numbers(
    trained_client: TestClient, value: str
) -> None:
    fields = ",".join(
        f'"{name}": {value if name == "monthly_fee" else json.dumps(field)}'
        for name, field in EXAMPLE_FEATURES.items()
    )

    response = trained_client.post(
        "/predict",
        content="{" + fields + "}",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    [error] = [
        error
        for error in response.json()["detail"]
        if error["type"] == "finite_number"
    ]
    assert error["loc"][-1] == "monthly_fee"
    assert error["input"] == str(float(value))
