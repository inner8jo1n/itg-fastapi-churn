import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from itg_fastapi_churn.schemas.model import ModelMetrics

CHURN_CLASS = 1


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> ModelMetrics:
    """
    Measure accuracy, F1 and ROC AUC of the model on held-out data

    :model: Pipeline - fitted churn pipeline
    :features: pd.DataFrame - feature matrix the model has not seen
    :target: pd.Series - true churn values for the features

    :return: accuracy, F1 score and ROC AUC
    """
    predictions = model.predict(features)
    return ModelMetrics(
        accuracy=accuracy_score(target, predictions),
        f1=f1_score(target, predictions, zero_division=0),
        roc_auc=_roc_auc(model, features, target),
    )


def _roc_auc(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> float | None:
    """
    Compute ROC AUC from the predicted churn probability

    Unlike accuracy and F1 it does not depend on the 0.5 threshold: it
    shows how well the model ranks leaving clients above staying ones.

    :model: Pipeline - fitted churn pipeline
    :features: pd.DataFrame - feature matrix the model has not seen
    :target: pd.Series - true churn values for the features

    :return: ROC AUC, or None if it is undefined for this data
    """
    if target.nunique() < 2 or CHURN_CLASS not in model.classes_:
        return None

    churn_column = list(model.classes_).index(CHURN_CLASS)
    probabilities = model.predict_proba(features)[:, churn_column]
    return float(roc_auc_score(target, probabilities))
