import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from itg_fastapi_churn.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def build_pipeline() -> Pipeline:
    """
    Build an untrained pipeline: scaling and one-hot encoding
    followed by logistic regression

    Classes are weighted by their frequency, so the model does not ignore
    the rare churn class.

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
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def train_churn_model(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """
    Train the churn classification pipeline

    :features: pd.DataFrame - training feature matrix
    :target: pd.Series - training churn target

    :return: fitted pipeline
    """
    pipeline = build_pipeline()
    pipeline.fit(features, target)
    return pipeline
