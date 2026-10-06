from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx2 import Response
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.api.dependencies import get_dataset
from itg_fastapi_churn.api.error_docs import (
    METRICS_ERRORS,
    PREDICT_ERRORS,
    TRAIN_ERRORS,
)
from itg_fastapi_churn.core.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.ml.features import FEATURE_COLUMNS
from itg_fastapi_churn.ml.history import TrainingHistory
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES
from itg_fastapi_churn.schemas.error import ErrorResponse

Scenario = Callable[[FastAPI, TestClient, Path, TrainedModel], Response]


class BrokenClassifier(DummyClassifier):
    def predict(self, X: object) -> object:
        raise RuntimeError("model internals")


def dataset_of(churn_values: list[int]) -> ChurnDataset:
    rows = [{**EXAMPLE_FEATURES, "churn": churn} for churn in churn_values]
    return ChurnDataset(pd.DataFrame(rows))


def train_on(
    app: FastAPI, client: TestClient, dataset: ChurnDataset, body: dict
) -> Response:
    app.dependency_overrides[get_dataset] = lambda: dataset
    return client.post("/model/train", json=body or None)


def train_from_file(app: FastAPI, client: TestClient, path: Path) -> Response:
    app.dependency_overrides[get_settings] = lambda: Settings(
        dataset_path=path
    )
    return client.post("/model/train")


def invalid_csv(tmp_path: Path) -> Path:
    path = tmp_path / "churn.csv"
    rows = [{**EXAMPLE_FEATURES, "churn": 0}] * 6
    rows.append({**EXAMPLE_FEATURES, "churn": 5})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def with_pipeline(
    app: FastAPI, model: TrainedModel, features: pd.DataFrame, step: object
) -> None:
    pipeline = Pipeline([("classifier", step)]).fit(
        features, pd.Series([0, 1])
    )
    app.state.model_store.save(replace(model, pipeline=pipeline))


def predict_with_model(
    app: FastAPI, client: TestClient, payload: dict
) -> Response:
    dataset = dataset_of([0] * 16 + [1] * 4)
    app.dependency_overrides[get_dataset] = lambda: dataset
    client.post("/model/train")
    return client.post("/predict", json=payload)


ALL_FEATURES = pd.DataFrame([EXAMPLE_FEATURES, EXAMPLE_FEATURES])[
    list(FEATURE_COLUMNS)
]
WITHOUT_REGION = {k: v for k, v in EXAMPLE_FEATURES.items() if k != "region"}

TRAIN_SCENARIOS: dict[str, Scenario] = {
    "dataset_not_found": lambda app, client, tmp, model: train_from_file(
        app, client, tmp / "missing.csv"
    ),
    "dataset_empty": lambda app, client, tmp, model: train_on(
        app, client, ChurnDataset(pd.DataFrame()), {}
    ),
    "dataset_invalid": lambda app, client, tmp, model: train_from_file(
        app, client, invalid_csv(tmp)
    ),
    "single_class": lambda app, client, tmp, model: train_on(
        app, client, dataset_of([0] * 20), {}
    ),
    "too_few_rows": lambda app, client, tmp, model: train_on(
        app, client, dataset_of([0, 1, 0]), {}
    ),
    "invalid_hyperparameters": lambda app, client, tmp, model: train_on(
        app, client, dataset_of([0, 1] * 10), {"hyperparameters": {"C": -1}}
    ),
    "unknown_model_type": lambda app, client, tmp, model: train_on(
        app, client, dataset_of([0, 1] * 10), {"model_type": "svm"}
    ),
}


def incompatible(
    app: FastAPI, client: TestClient, tmp: Path, model: TrainedModel
) -> Response:
    with_pipeline(app, model, ALL_FEATURES.assign(extra=0), DummyClassifier())
    return client.post("/predict", json=EXAMPLE_FEATURES)


def broken(
    app: FastAPI, client: TestClient, tmp: Path, model: TrainedModel
) -> Response:
    with_pipeline(app, model, ALL_FEATURES, BrokenClassifier())
    return client.post("/predict", json=EXAMPLE_FEATURES)


PREDICT_SCENARIOS: dict[str, Scenario] = {
    "model_not_trained": lambda app, client, tmp, model: client.post(
        "/predict", json=EXAMPLE_FEATURES
    ),
    "incompatible_model": incompatible,
    "prediction_failed": broken,
    "extra_feature": lambda app, client, tmp, model: predict_with_model(
        app, client, {**EXAMPLE_FEATURES, "extra": 1}
    ),
    "missing_feature": lambda app, client, tmp, model: predict_with_model(
        app, client, WITHOUT_REGION
    ),
    "wrong_type": lambda app, client, tmp, model: predict_with_model(
        app, client, {**EXAMPLE_FEATURES, "monthly_fee": "19.99"}
    ),
}


def unreadable_history(
    app: FastAPI, client: TestClient, tmp: Path, model: TrainedModel
) -> Response:
    app.state.training_history = TrainingHistory(tmp)
    return client.get("/model/metrics")


METRICS_SCENARIOS: dict[str, Scenario] = {
    "history_unavailable": unreadable_history,
    "limit_out_of_range": lambda app, client, tmp, model: client.get(
        "/model/metrics", params={"limit": 1000}
    ),
    "unknown_model_type": lambda app, client, tmp, model: client.get(
        "/model/metrics", params={"model_type": "svm"}
    ),
}

ROUTE_ERRORS = [
    (TRAIN_ERRORS, TRAIN_SCENARIOS),
    (PREDICT_ERRORS, PREDICT_SCENARIOS),
    (METRICS_ERRORS, METRICS_SCENARIOS),
]


def documented_examples(
    responses: dict,
) -> dict[str, tuple[int, dict]]:
    examples = {}
    for status, response in responses.items():
        content = response["content"]["application/json"]
        for name, item in content["examples"].items():
            examples[name] = (int(status), item["value"])
    return examples


@pytest.mark.parametrize(
    ("path", "method", "statuses"),
    [
        ("/model/train", "post", {"404", "409", "422", "500"}),
        ("/predict", "post", {"409", "422", "500"}),
        ("/model/metrics", "get", {"422", "500"}),
    ],
)
def test_docs_list_error_statuses(
    client: TestClient, path: str, method: str, statuses: set[str]
) -> None:
    responses = client.get("/openapi.json").json()["paths"][path][method]

    assert statuses <= set(responses["responses"])
    for status in statuses:
        content = responses["responses"][status]["content"]
        assert content["application/json"]["schema"]["$ref"].endswith(
            "/ErrorResponse"
        )


@pytest.mark.parametrize(
    "responses", [TRAIN_ERRORS, PREDICT_ERRORS, METRICS_ERRORS]
)
def test_every_example_is_a_valid_error_response(responses: dict) -> None:
    for _, body in documented_examples(responses).values():
        ErrorResponse.model_validate(body)


@pytest.mark.parametrize(("responses", "scenarios"), ROUTE_ERRORS)
def test_every_example_has_a_scenario(
    responses: dict, scenarios: dict[str, Scenario]
) -> None:
    documented = set(documented_examples(responses)) - {"internal_error"}

    assert documented == set(scenarios)


@pytest.mark.parametrize(
    ("responses", "name", "scenario"),
    [
        (responses, name, run)
        for responses, scenarios in ROUTE_ERRORS
        for name, run in scenarios.items()
    ],
)
def test_example_matches_real_response(
    app: FastAPI,
    tmp_path: Path,
    trained_model: TrainedModel,
    responses: dict,
    name: str,
    scenario: Scenario,
) -> None:
    client = TestClient(app, raise_server_exceptions=False)
    status, body = documented_examples(responses)[name]

    response = scenario(app, client, tmp_path, trained_model)

    assert response.status_code == status
    assert response.json() == body
