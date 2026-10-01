from collections.abc import Iterable

import pandas as pd
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.errors import (
    IncompatibleModelError,
    PredictionFailedError,
)
from itg_fastapi_churn.ml.features import FEATURE_COLUMNS
from itg_fastapi_churn.schemas.churn import FeatureVectorChurn
from itg_fastapi_churn.schemas.prediction import ChurnPrediction


def build_feature_frame(
    pipeline: Pipeline, clients: list[FeatureVectorChurn]
) -> pd.DataFrame:
    """
    Turn clients into a table with exactly the columns the model was
    trained on, in the same order

    The column list comes from the fitted model itself, so it cannot drift
    away from what the model saw during training.

    :pipeline: Pipeline - fitted churn pipeline
    :clients: list[FeatureVectorChurn] - clients to predict churn for

    :return: feature table ready for the model
    """
    trained_columns = list(getattr(pipeline, "feature_names_in_", []))
    if set(trained_columns) != set(FEATURE_COLUMNS):
        raise IncompatibleModelError(
            details={"trained_features": trained_columns}
        )

    return pd.DataFrame(
        [client.model_dump() for client in clients], columns=trained_columns
    )


def predict_churn(
    pipeline: Pipeline, clients: list[FeatureVectorChurn]
) -> list[ChurnPrediction]:
    """
    Predict churn class and class probabilities for every client

    Raises IncompatibleModelError if the model expects other features and
    PredictionFailedError if the model itself fails; any model failure is
    caught because a broken model can raise almost anything.

    :pipeline: Pipeline - fitted churn pipeline
    :clients: list[FeatureVectorChurn] - clients to predict churn for

    :return: one prediction per client, in the same order
    """
    if not clients:
        return []

    features = build_feature_frame(pipeline, clients)
    try:
        classes = [int(label) for label in pipeline.classes_]
        labels = pipeline.predict(features)
        probabilities = pipeline.predict_proba(features)
        return [
            ChurnPrediction(
                churn=int(label),
                probabilities=_by_class(classes, class_probabilities),
            )
            for label, class_probabilities in zip(
                labels, probabilities, strict=True
            )
        ]
    except Exception as error:
        raise PredictionFailedError() from error


def _by_class(
    classes: list[int], probabilities: Iterable[float]
) -> dict[int, float]:
    """
    Pair every class with its predicted probability

    :classes: list[int] - classes in the order the model knows them
    :probabilities: Iterable[float] - probabilities in the same order

    :return: probability of every class
    """
    return {
        label: float(probability)
        for label, probability in zip(classes, probabilities, strict=True)
    }
