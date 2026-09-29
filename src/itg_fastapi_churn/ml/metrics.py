import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.schemas.model import ModelMetrics


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> ModelMetrics:
    """
    Measure accuracy and F1 of the model on held-out data

    :model: Pipeline - fitted churn pipeline
    :features: pd.DataFrame - feature matrix the model has not seen
    :target: pd.Series - true churn values for the features

    :return: accuracy and F1 score
    """
    predictions = model.predict(features)
    return ModelMetrics(
        accuracy=accuracy_score(target, predictions),
        f1=f1_score(target, predictions, zero_division=0),
    )
