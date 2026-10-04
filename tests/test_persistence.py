from dataclasses import replace
from pathlib import Path

import joblib
import pandas as pd
import pytest

from itg_fastapi_churn.ml.persistence import (
    ModelLoadError,
    TrainedModel,
    load_churn_model,
    save_churn_model,
)
from itg_fastapi_churn.schemas.model import ModelMetrics

FEATURES = pd.DataFrame({"x": [0, 1]})


def test_saved_model_loads_back(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "models" / "model.joblib"

    save_churn_model(trained_model, path)
    loaded = load_churn_model(path)

    assert loaded is not None
    assert loaded.trained_at == trained_model.trained_at
    assert loaded.metrics == trained_model.metrics
    assert list(loaded.pipeline.predict(FEATURES)) == [1, 1]


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    loaded = load_churn_model(tmp_path / "missing.joblib")

    assert loaded is None


def test_save_leaves_no_temporary_file(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    save_churn_model(trained_model, tmp_path / "model.joblib")

    assert [file.name for file in tmp_path.iterdir()] == ["model.joblib"]


@pytest.mark.parametrize("content", [b"", b"not a model"])
def test_load_rejects_damaged_file(content: bytes, tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    path.write_bytes(content)

    with pytest.raises(ModelLoadError):
        load_churn_model(path)


def test_load_rejects_unexpected_object(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump({"not": "a model"}, path)

    with pytest.raises(ModelLoadError):
        load_churn_model(path)


def test_saved_model_keeps_type_and_hyperparameters(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.joblib"

    save_churn_model(trained_model, path)
    loaded = load_churn_model(path)

    assert loaded is not None
    assert loaded.model_type == trained_model.model_type
    assert loaded.hyperparameters == trained_model.hyperparameters


def test_load_rejects_model_saved_by_older_version(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.joblib"
    old_model = object.__new__(TrainedModel)
    for name in ("pipeline", "trained_at", "metrics"):
        object.__setattr__(old_model, name, getattr(trained_model, name))
    joblib.dump(old_model, path)

    with pytest.raises(ModelLoadError, match="model_type, hyperparameters"):
        load_churn_model(path)


def test_model_saved_before_roc_auc_gets_default(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.joblib"
    old_metrics = ModelMetrics(accuracy=0.5, f1=0.4)
    del old_metrics.__dict__["roc_auc"]
    joblib.dump(replace(trained_model, metrics=old_metrics), path)

    loaded = load_churn_model(path)

    assert loaded is not None
    assert loaded.metrics.roc_auc is None
    assert loaded.metrics.f1 == 0.4


def test_load_rejects_broken_metrics(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump(replace(trained_model, metrics="not metrics"), path)

    with pytest.raises(ModelLoadError, match="invalid metrics"):
        load_churn_model(path)
