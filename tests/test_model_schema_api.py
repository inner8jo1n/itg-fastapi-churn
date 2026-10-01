from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from itg_fastapi_churn.api.dependencies import get_dataset
from itg_fastapi_churn.config import Settings, get_settings
from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.ml.features import FEATURE_COLUMNS, prepare_data
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.schemas.churn import (
    EXAMPLE_FEATURES,
    FeatureVectorChurn,
)


def features_by_name(client: TestClient) -> dict[str, dict]:
    response = client.get("/model/schema")
    assert response.status_code == 200
    return {spec["name"]: spec for spec in response.json()["features"]}


def test_schema_lists_every_feature_with_types(client: TestClient) -> None:
    response = client.get("/model/schema").json()
    specs = features_by_name(client)

    assert [spec["name"] for spec in response["features"]] == list(
        FEATURE_COLUMNS
    )
    assert response["target"] == "churn"
    assert specs["monthly_fee"]["type"] == "number"
    assert specs["support_requests"]["type"] == "integer"
    assert specs["region"] == {
        "name": "region",
        "type": "string",
        "kind": "categorical",
        "minimum": None,
        "maximum": None,
        "known_values": None,
    }
    assert specs["autopay_enabled"]["kind"] == "numeric"
    assert specs["autopay_enabled"]["minimum"] == 0
    assert specs["autopay_enabled"]["maximum"] == 1


def test_schema_shows_categories_of_trained_model(
    app: FastAPI, client: TestClient, training_dataset: ChurnDataset
) -> None:
    app.dependency_overrides[get_dataset] = lambda: training_dataset
    app.dependency_overrides[get_settings] = lambda: Settings(
        test_size=0.25, random_state=0
    )
    client.post("/model/train")

    specs = features_by_name(client)

    assert specs["region"]["known_values"] == ["europe"]
    assert specs["payment_method"]["known_values"] == ["card"]
    assert specs["monthly_fee"]["known_values"] is None


@pytest.mark.parametrize(
    ("name", "limit", "step"),
    [
        ("monthly_fee", "minimum", -1),
        ("autopay_enabled", "minimum", -1),
        ("autopay_enabled", "maximum", 1),
    ],
)
def test_schema_limits_match_validation(
    client: TestClient, name: str, limit: str, step: int
) -> None:
    bound = features_by_name(client)[name][limit]

    FeatureVectorChurn.model_validate({**EXAMPLE_FEATURES, name: bound})
    with pytest.raises(ValidationError):
        FeatureVectorChurn.model_validate(
            {**EXAMPLE_FEATURES, name: bound + step}
        )


def test_schema_survives_model_with_other_structure(
    app: FastAPI,
    client: TestClient,
    training_dataset: ChurnDataset,
    trained_model: TrainedModel,
) -> None:
    features, target = prepare_data(training_dataset.data)
    encoder_step = ColumnTransformer(
        [("categorical", OneHotEncoder(), ["device_type"])]
    )
    pipeline = Pipeline(
        [("preprocess", encoder_step), ("classifier", DummyClassifier())]
    ).fit(features, target)
    app.state.model_store.save(replace(trained_model, pipeline=pipeline))

    specs = features_by_name(client)

    assert specs["device_type"]["known_values"] == ["mobile"]
    assert specs["region"]["known_values"] is None


def test_schema_survives_model_without_preprocessing(
    app: FastAPI, client: TestClient, trained_model: TrainedModel
) -> None:
    app.state.model_store.save(trained_model)

    specs = features_by_name(client)

    assert all(spec["known_values"] is None for spec in specs.values())


def test_schema_covers_every_validation_rule() -> None:
    handled = {"type", "title", "minimum", "maximum"}
    properties = FeatureVectorChurn.model_json_schema()["properties"]

    for name, rules in properties.items():
        assert set(rules) <= handled, f"{name}: {set(rules) - handled}"
