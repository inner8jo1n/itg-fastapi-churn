import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.main import create_app
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES


@pytest.fixture
def app() -> FastAPI:
    return create_app()


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
