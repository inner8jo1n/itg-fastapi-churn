from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from itg_fastapi_churn.api.dependencies import get_dataset
from itg_fastapi_churn.core.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset


def test_dataset_info_returns_summary(
    app: FastAPI, client: TestClient, dataset: ChurnDataset
) -> None:
    app.dependency_overrides[get_dataset] = lambda: dataset

    response = client.get("/dataset/info")

    assert response.status_code == 200
    assert response.json()["n_rows"] == 3
    assert response.json()["churn_distribution"] == {"0": 2, "1": 1}


def test_dataset_preview_returns_requested_rows(
    app: FastAPI, client: TestClient, dataset: ChurnDataset
) -> None:
    app.dependency_overrides[get_dataset] = lambda: dataset

    response = client.get("/dataset/preview", params={"n": 2})

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_dataset_preview_rejects_invalid_n(
    app: FastAPI, client: TestClient, dataset: ChurnDataset
) -> None:
    app.dependency_overrides[get_dataset] = lambda: dataset

    response = client.get("/dataset/preview", params={"n": 0})

    assert response.status_code == 422


def test_dataset_returns_404_when_file_missing(
    app: FastAPI, client: TestClient, tmp_path: Path
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        dataset_path=tmp_path / "missing.csv"
    )

    response = client.get("/dataset/info")

    assert response.status_code == 404


def test_split_info_returns_sizes_and_balance(
    app: FastAPI, client: TestClient, training_dataset: ChurnDataset
) -> None:
    app.dependency_overrides[get_dataset] = lambda: training_dataset
    app.dependency_overrides[get_settings] = lambda: Settings(
        test_size=0.25, random_state=0
    )

    response = client.get("/dataset/split-info")

    assert response.status_code == 200
    assert response.json()["train_rows"] == 15
    assert response.json()["test_rows"] == 5
    assert response.json()["train_churn_distribution"] == {"0": 12, "1": 3}
    assert response.json()["test_churn_distribution"] == {"0": 4, "1": 1}
