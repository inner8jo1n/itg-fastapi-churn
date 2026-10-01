from datetime import UTC, datetime
from pathlib import Path

from sklearn.compose import ColumnTransformer

from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.ml.features import prepare_data
from itg_fastapi_churn.ml.model import train_churn_model
from itg_fastapi_churn.ml.persistence import (
    TrainedModel,
    load_churn_model,
    save_churn_model,
)
from itg_fastapi_churn.schemas.model import ModelMetrics
from itg_fastapi_churn.schemas.training import ModelType


def test_preprocessing_and_model_are_saved_as_one_object(
    training_dataset: ChurnDataset, tmp_path: Path
) -> None:
    features, target = prepare_data(training_dataset.data)
    pipeline = train_churn_model(features, target).pipeline
    path = tmp_path / "model.joblib"
    save_churn_model(
        TrainedModel(
            pipeline=pipeline,
            trained_at=datetime(2026, 1, 1, tzinfo=UTC),
            metrics=ModelMetrics(accuracy=1.0, f1=1.0),
            model_type=ModelType.LOGREG,
            hyperparameters={},
        ),
        path,
    )

    loaded = load_churn_model(path)

    assert loaded is not None
    preprocess = loaded.pipeline.named_steps["preprocess"]
    assert isinstance(preprocess, ColumnTransformer)
    fitted_scaler = preprocess.named_transformers_["numeric"]
    original_scaler = pipeline.named_steps["preprocess"].named_transformers_[
        "numeric"
    ]
    assert list(fitted_scaler.mean_) == list(original_scaler.mean_)
    assert list(loaded.pipeline.predict_proba(features)[:, 1]) == list(
        pipeline.predict_proba(features)[:, 1]
    )
