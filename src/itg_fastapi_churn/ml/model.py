import logging
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from pydantic import JsonValue
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from itg_fastapi_churn.errors import InvalidHyperparametersError
from itg_fastapi_churn.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from itg_fastapi_churn.schemas.training import ModelType, TrainingConfigChurn

CLASSIFIERS: dict[ModelType, type[BaseEstimator]] = {
    ModelType.LOGREG: LogisticRegression,
    ModelType.RANDOM_FOREST: RandomForestClassifier,
}
DEFAULT_HYPERPARAMETERS: dict[ModelType, dict[str, JsonValue]] = {
    ModelType.LOGREG: {"max_iter": 1000, "class_weight": "balanced"},
    ModelType.RANDOM_FOREST: {
        "n_estimators": 100,
        "class_weight": "balanced",
        "random_state": 42,
    },
}

UPPER_LIMITS: dict[str, int] = {
    "n_estimators": 1000,
    "max_iter": 10_000,
    "n_jobs": 16,
}
LARGEST_INTEGER = 2**63 - 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingOutcome:
    """
    Result of training: the fitted pipeline and what sklearn warned about

    :pipeline: Pipeline - fitted churn pipeline
    :warnings: list[str] - warnings raised while fitting, without repeats
    """

    pipeline: Pipeline
    warnings: list[str]


def resolve_hyperparameters(
    config: TrainingConfigChurn,
) -> dict[str, JsonValue]:
    """
    Combine the service defaults with the hyperparameters from the config

    Values from the config win, so any default can be overridden.

    :config: TrainingConfigChurn - requested model and hyperparameters

    :return: hyperparameters the classifier is trained with
    """
    return {
        **DEFAULT_HYPERPARAMETERS[config.model_type],
        **config.hyperparameters,
    }


def build_classifier(config: TrainingConfigChurn) -> BaseEstimator:
    """
    Create the classifier chosen in the config

    Raises InvalidHyperparametersError if a hyperparameter name is unknown
    to the classifier, a number is given as true/false, a value is
    above the service limit or has a number that is infinite, NaN or
    too large for a 64-bit integer.

    :config: TrainingConfigChurn - requested model and hyperparameters

    :return: unfitted classifier
    """
    classifier = CLASSIFIERS[config.model_type]()
    defaults = classifier.get_params()
    hyperparameters = {
        name: _prepare_value(name, value, defaults)
        for name, value in resolve_hyperparameters(config).items()
    }
    try:
        classifier.set_params(**hyperparameters)
    except ValueError as error:
        raise InvalidHyperparametersError(str(error)) from error
    return classifier


def _prepare_value(
    name: str, value: JsonValue, defaults: dict[str, object]
) -> object:
    """
    Check one hyperparameter and convert it from JSON to what sklearn
    expects

    JSON has no integer keys, so class_weight keys like "1" become 1.

    :name: str - hyperparameter name
    :value: JsonValue - value from the request
    :defaults: dict[str, object] - default parameters of the classifier

    :return: value ready for the classifier
    """
    expects_bool = isinstance(defaults.get(name), bool)
    if isinstance(value, bool) and name in defaults and not expects_bool:
        raise InvalidHyperparametersError(
            f"The '{name}' parameter expects a value, not true/false"
        )

    if not _has_safe_numbers(value):
        raise InvalidHyperparametersError(
            f"The '{name}' parameter has an infinite, NaN or too large number"
        )

    limit = UPPER_LIMITS.get(name)
    is_number = isinstance(value, int | float)
    if limit is not None and is_number and value > limit:
        raise InvalidHyperparametersError(
            f"The '{name}' parameter must be at most {limit}"
        )

    if name == "class_weight" and isinstance(value, dict):
        return {_class_key(key): weight for key, weight in value.items()}

    return value


def _has_safe_numbers(value: JsonValue) -> bool:
    """
    Tell whether every number inside a JSON value is safe for sklearn

    JSON parsing turns 1e309 into infinity, which history and status
    would show as null, and sklearn fails with OverflowError on integers
    that do not fit into 64 bits.

    :value: JsonValue - value from the request

    :return: True if all floats are finite and all integers fit 64 bits
    """
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int):
        return abs(value) <= LARGEST_INTEGER
    if isinstance(value, dict):
        return all(_has_safe_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_has_safe_numbers(item) for item in value)
    return True


def _class_key(key: str) -> int | str:
    """
    Turn a JSON class key like "1" into the class label 1

    :key: str - key from a JSON object

    :return: integer class label, or the key unchanged if it is not a number
    """
    return int(key) if key.lstrip("-").isdigit() else key


def build_pipeline(classifier: BaseEstimator) -> Pipeline:
    """
    Build an untrained pipeline: scaling and one-hot encoding
    followed by the given classifier

    Preprocessing lives inside the pipeline, so it is fitted on the train
    part only and is saved and loaded together with the classifier.
    Columns outside the feature lists are dropped.

    :classifier: BaseEstimator - unfitted classifier

    :return: unfitted churn classification pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[("preprocess", preprocessor), ("classifier", classifier)]
    )


def train_churn_model(
    features: pd.DataFrame,
    target: pd.Series,
    config: TrainingConfigChurn | None = None,
) -> TrainingOutcome:
    """
    Train the churn classification pipeline

    Raises InvalidHyperparametersError if a hyperparameter is unknown or
    has a value the classifier rejects. Warnings such as "failed to
    converge" do not stop training; they are logged and returned so the
    caller can show them.

    :features: pd.DataFrame - training feature matrix
    :target: pd.Series - training churn target
    :config: TrainingConfigChurn | None - model and hyperparameters;
        logistic regression with defaults when not given

    :return: fitted pipeline and training warnings
    """
    if config is None:
        config = TrainingConfigChurn()

    pipeline = build_pipeline(build_classifier(config))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pipeline.fit(features, target)
        except ValueError as error:
            raise InvalidHyperparametersError(str(error)) from error

    messages = _unique_messages(str(warning.message) for warning in caught)
    for message in messages:
        logger.warning("Training warning: %s", message)
    return TrainingOutcome(pipeline=pipeline, warnings=messages)


def _unique_messages(messages: Iterable[str]) -> list[str]:
    """
    Put every message on one line and drop repeats, keeping order

    :messages: Iterable[str] - raw warning messages, maybe multi-line

    :return: unique one-line messages
    """
    one_line = (" ".join(message.split()) for message in messages)
    return list(dict.fromkeys(one_line))
