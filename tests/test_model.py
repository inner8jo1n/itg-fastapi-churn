import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from itg_fastapi_churn.ml.features import prepare_data
from itg_fastapi_churn.ml.model import (
    InvalidHyperparametersError,
    build_classifier,
    resolve_hyperparameters,
    train_churn_model,
)
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES
from itg_fastapi_churn.schemas.training import ModelType, TrainingConfigChurn


def make_training_data() -> tuple[pd.DataFrame, pd.Series]:
    rows = [
        {**EXAMPLE_FEATURES, "churn": churn, "failed_payments": churn * 5}
        for churn in [0, 1] * 10
    ]
    features, target = prepare_data(pd.DataFrame(rows))

    return features, target


def test_train_churn_model_returns_fitted_pipeline() -> None:
    features, target = make_training_data()

    model = train_churn_model(features, target).pipeline

    assert len(model.predict(features)) == len(features)


def test_train_churn_model_learns_obvious_pattern() -> None:
    features, target = make_training_data()

    model = train_churn_model(features, target).pipeline

    assert list(model.predict(features)) == list(target)


@pytest.mark.parametrize(
    ("model_type", "classifier_class"),
    [
        (ModelType.LOGREG, LogisticRegression),
        (ModelType.RANDOM_FOREST, RandomForestClassifier),
    ],
)
def test_build_classifier_picks_model_by_type(
    model_type: ModelType, classifier_class: type
) -> None:
    classifier = build_classifier(TrainingConfigChurn(model_type=model_type))

    assert isinstance(classifier, classifier_class)


def test_hyperparameters_override_defaults() -> None:
    config = TrainingConfigChurn(
        model_type=ModelType.LOGREG, hyperparameters={"C": 0.5}
    )

    classifier = build_classifier(config)

    assert classifier.get_params()["C"] == 0.5
    assert classifier.get_params()["class_weight"] == "balanced"


def test_resolve_hyperparameters_lets_config_win() -> None:
    config = TrainingConfigChurn(
        model_type=ModelType.RANDOM_FOREST,
        hyperparameters={"n_estimators": 5, "max_depth": 3},
    )

    assert resolve_hyperparameters(config) == {
        "n_estimators": 5,
        "max_depth": 3,
        "class_weight": "balanced",
        "random_state": 42,
    }


def test_random_forest_learns_obvious_pattern() -> None:
    features, target = make_training_data()
    config = TrainingConfigChurn(model_type=ModelType.RANDOM_FOREST)

    model = train_churn_model(features, target, config).pipeline

    assert list(model.predict(features)) == list(target)


def test_unknown_hyperparameter_is_rejected() -> None:
    config = TrainingConfigChurn(hyperparameters={"n_trees": 5})

    with pytest.raises(InvalidHyperparametersError, match="n_trees"):
        build_classifier(config)


def test_invalid_hyperparameter_value_is_rejected_on_training() -> None:
    features, target = make_training_data()
    config = TrainingConfigChurn(hyperparameters={"C": -1})

    with pytest.raises(InvalidHyperparametersError, match="'C'"):
        train_churn_model(features, target, config)


@pytest.mark.parametrize(
    ("model_type", "name"),
    [
        (ModelType.RANDOM_FOREST, "n_estimators"),
        (ModelType.RANDOM_FOREST, "max_depth"),
        (ModelType.LOGREG, "C"),
    ],
)
def test_bool_is_rejected_for_non_bool_parameter(
    model_type: ModelType, name: str
) -> None:
    config = TrainingConfigChurn(
        model_type=model_type, hyperparameters={name: True}
    )

    with pytest.raises(InvalidHyperparametersError, match="true/false"):
        build_classifier(config)


def test_bool_is_accepted_for_bool_parameter() -> None:
    config = TrainingConfigChurn(hyperparameters={"fit_intercept": False})

    classifier = build_classifier(config)

    assert classifier.get_params()["fit_intercept"] is False


@pytest.mark.parametrize(
    ("model_type", "hyperparameters"),
    [
        (ModelType.RANDOM_FOREST, {"n_estimators": 1001}),
        (ModelType.LOGREG, {"max_iter": 10_001}),
    ],
)
def test_too_large_value_is_rejected(
    model_type: ModelType, hyperparameters: dict[str, int]
) -> None:
    config = TrainingConfigChurn(
        model_type=model_type, hyperparameters=dict(hyperparameters)
    )

    with pytest.raises(InvalidHyperparametersError, match="at most"):
        build_classifier(config)


def test_class_weight_keys_become_class_labels() -> None:
    features, target = make_training_data()
    config = TrainingConfigChurn(
        hyperparameters={"class_weight": {"0": 1, "1": 5}}
    )

    model = train_churn_model(features, target, config).pipeline

    assert model.named_steps["classifier"].class_weight == {0: 1, 1: 5}


def test_training_without_problems_has_no_warnings() -> None:
    features, target = make_training_data()

    outcome = train_churn_model(features, target)

    assert outcome.warnings == []


def test_training_reports_convergence_warning_once() -> None:
    features, target = make_training_data()
    config = TrainingConfigChurn(hyperparameters={"max_iter": 1})

    outcome = train_churn_model(features, target, config)

    assert len(outcome.warnings) == 1
    assert "failed to converge" in outcome.warnings[0]
