from collections.abc import Iterable

import pandas as pd
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.ml.features import FEATURE_COLUMNS
from itg_fastapi_churn.schemas.churn import FeatureVectorChurn
from itg_fastapi_churn.schemas.prediction import ChurnPrediction


def predict_churn(
    pipeline: Pipeline, clients: list[FeatureVectorChurn]
) -> list[ChurnPrediction]:
    """
    Predict churn class and class probabilities for every client

    :pipeline: Pipeline - fitted churn pipeline
    :clients: list[FeatureVectorChurn] - clients to predict churn for

    :return: one prediction per client, in the same order
    """
    if not clients:
        return []

    features = pd.DataFrame(
        [client.model_dump() for client in clients],
        columns=list(FEATURE_COLUMNS),
    )
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
