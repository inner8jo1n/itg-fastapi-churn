from fastapi.testclient import TestClient

from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES


def test_predict_echoes_received_features(client: TestClient) -> None:
    response = client.post("/predict", json=EXAMPLE_FEATURES)

    assert response.status_code == 200
    assert response.json() == EXAMPLE_FEATURES


def test_predict_rejects_missing_feature(client: TestClient) -> None:
    payload = {k: v for k, v in EXAMPLE_FEATURES.items() if k != "region"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
