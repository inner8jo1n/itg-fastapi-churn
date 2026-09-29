from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from itg_fastapi_churn.api.dependencies import get_dataset
from itg_fastapi_churn.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
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
    assert response.json()["detail"] == "Dataset is empty"


def test_train_model_returns_404_when_file_missing(
    app: FastAPI, client: TestClient, tmp_path: Path
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        dataset_path=tmp_path / "missing.csv"
    )

    response = client.post("/model/train")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset file not found"


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
    assert (
        response.json()["detail"] == "Dataset must contain both churn classes"
    )


def test_train_model_rejects_too_small_dataset(
    app: FastAPI, client: TestClient, dataset: ChurnDataset
) -> None:
    app.dependency_overrides[get_dataset] = lambda: dataset

    response = client.post("/model/train")

    assert response.status_code == 409
    assert response.json()["detail"] == "Not enough rows to split the dataset"
