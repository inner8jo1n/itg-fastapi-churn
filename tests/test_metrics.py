import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.ml.features import prepare_data
from itg_fastapi_churn.ml.metrics import evaluate_model
from itg_fastapi_churn.ml.model import train_churn_model

FEATURES = pd.DataFrame({"x": [0, 0, 0, 0]})
TARGET = pd.Series([0, 0, 0, 1])


def make_constant_model(answer: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "classifier",
                DummyClassifier(strategy="constant", constant=answer),
            )
        ]
    ).fit(FEATURES, TARGET)


def test_evaluate_model_exposes_accuracy_trap() -> None:
    model = make_constant_model(0)

    metrics = evaluate_model(model, FEATURES, TARGET)

    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.f1 == pytest.approx(0.0)


def test_evaluate_model_computes_f1() -> None:
    model = make_constant_model(1)

    metrics = evaluate_model(model, FEATURES, TARGET)

    assert metrics.accuracy == pytest.approx(0.25)
    assert metrics.f1 == pytest.approx(0.4)


def test_evaluate_model_computes_roc_auc() -> None:
    model = make_constant_model(1)

    metrics = evaluate_model(model, FEATURES, TARGET)

    assert metrics.roc_auc == pytest.approx(0.5)


def test_roc_auc_is_none_for_single_class_target() -> None:
    model = make_constant_model(0)

    metrics = evaluate_model(model, FEATURES, pd.Series([0, 0, 0, 0]))

    assert metrics.roc_auc is None


def test_roc_auc_ranks_by_probability(
    training_dataset: ChurnDataset,
) -> None:
    features, target = prepare_data(training_dataset.data)
    model = train_churn_model(features, target).pipeline

    metrics = evaluate_model(model, features, target)

    assert metrics.roc_auc == pytest.approx(1.0)
