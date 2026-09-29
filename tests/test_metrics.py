import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.ml.metrics import evaluate_model

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
