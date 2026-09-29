from fastapi.testclient import TestClient


def test_root_reports_service_is_running(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}
