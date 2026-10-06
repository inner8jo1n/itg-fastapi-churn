from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from itg_fastapi_churn.dataset.loader import load_dataset
from itg_fastapi_churn.ml.features import prepare_data
from itg_fastapi_churn.ml.metrics import evaluate_model
from itg_fastapi_churn.ml.model import train_churn_model
from itg_fastapi_churn.ml.split import split_dataset
from itg_fastapi_churn.schemas.model import ModelMetrics
from itg_fastapi_churn.schemas.training import ModelType, TrainingConfigChurn

CONFIGS = [
    pytest.param(TrainingConfigChurn(), id="logreg"),
    pytest.param(
        TrainingConfigChurn(model_type=ModelType.RANDOM_FOREST),
        id="random_forest",
    ),
]


def train_on_sample(
    path: Path, config: TrainingConfigChurn
) -> tuple[ModelMetrics, np.ndarray]:
    features, target = prepare_data(load_dataset(path))
    split = split_dataset(features, target, test_size=0.2, random_state=42)
    pipeline = train_churn_model(split.x_train, split.y_train, config).pipeline
    metrics = evaluate_model(pipeline, split.x_test, split.y_test)
    return metrics, pipeline.predict_proba(split.x_test)


@pytest.mark.parametrize("config", CONFIGS)
def test_training_twice_gives_same_model(
    sample_dataset_path: Path, config: TrainingConfigChurn
) -> None:
    first_metrics, first_probabilities = train_on_sample(
        sample_dataset_path, config
    )
    second_metrics, second_probabilities = train_on_sample(
        sample_dataset_path, config
    )

    assert first_metrics == second_metrics
    assert np.array_equal(first_probabilities, second_probabilities)


def test_forest_seed_changes_the_model(sample_dataset_path: Path) -> None:
    def forest(seed: int) -> TrainingConfigChurn:
        return TrainingConfigChurn(
            model_type=ModelType.RANDOM_FOREST,
            hyperparameters={"random_state": seed},
        )

    _, first = train_on_sample(sample_dataset_path, forest(1))
    _, second = train_on_sample(sample_dataset_path, forest(2))

    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"model_type": "logreg"}, id="logreg"),
        pytest.param({"model_type": "random_forest"}, id="random_forest"),
    ],
)
@pytest.mark.usefixtures("sample_app")
def test_api_training_twice_gives_same_metrics(
    client: TestClient, config: dict[str, JsonValue]
) -> None:
    first = client.post("/model/train", json=config).json()
    second = client.post("/model/train", json=config).json()

    assert first == second
