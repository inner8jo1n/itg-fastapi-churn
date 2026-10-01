from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.api.dependencies import get_dataset
from itg_fastapi_churn.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.main import create_app
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.ml.store import ModelStore
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES
from itg_fastapi_churn.schemas.model import ModelMetrics
from itg_fastapi_churn.schemas.training import ModelType


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    application = create_app()
    application.state.model_store = ModelStore(tmp_path / "model.joblib")
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def dataset() -> ChurnDataset:
    return ChurnDataset(
        pd.DataFrame(
            [
                {**EXAMPLE_FEATURES, "churn": 0},
                {**EXAMPLE_FEATURES, "churn": 1},
                {**EXAMPLE_FEATURES, "churn": 0},
            ]
        )
    )


@pytest.fixture
def training_dataset() -> ChurnDataset:
    churn_values = [0] * 16 + [1] * 4
    rows = [
        {**EXAMPLE_FEATURES, "churn": churn, "failed_payments": churn * 5}
        for churn in churn_values
    ]
    return ChurnDataset(pd.DataFrame(rows))


@pytest.fixture
def trained_model() -> TrainedModel:
    pipeline = Pipeline(
        steps=[
            (
                "classifier",
                DummyClassifier(strategy="constant", constant=1),
            )
        ]
    ).fit(pd.DataFrame({"x": [0, 1]}), pd.Series([0, 1]))

    return TrainedModel(
        pipeline=pipeline,
        trained_at=datetime(2026, 1, 1, tzinfo=UTC),
        metrics=ModelMetrics(accuracy=0.5, f1=0.4),
        model_type=ModelType.LOGREG,
        hyperparameters={"C": 0.5},
    )


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
