from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from itg_fastapi_churn.errors import ModelNotTrainedError
from itg_fastapi_churn.ml.persistence import TrainedModel, load_churn_model
from itg_fastapi_churn.ml.store import ModelStore


def test_status_when_no_model(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "model.joblib")

    status = store.status()

    assert not status.is_trained
    assert status.trained_at is None
    assert status.metrics is None


def test_save_keeps_model_in_memory_and_on_disk(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.joblib"
    store = ModelStore(path)

    store.save(trained_model)

    assert store.current is trained_model
    assert path.exists()
    assert store.status().is_trained


def test_load_restores_model_after_restart(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.joblib"
    store = ModelStore(path)
    store.save(trained_model)

    restarted = ModelStore(path)
    restarted.load()

    assert restarted.current is not None
    assert restarted.current.trained_at == trained_model.trained_at
    assert restarted.current.metrics == trained_model.metrics


def test_load_without_file_keeps_no_model(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "missing.joblib")
    store.load()

    assert store.current is None


def test_load_skips_damaged_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not a model")
    store = ModelStore(path)

    store.load()

    assert store.current is None
    assert "Saved model is ignored" in caplog.text


def test_parallel_saves_keep_memory_and_disk_in_sync(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    path = tmp_path / "model.joblib"
    store = ModelStore(path)
    models = [
        replace(trained_model, trained_at=datetime(2026, 1, day, tzinfo=UTC))
        for day in range(1, 21)
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(store.save, models))

    saved = load_churn_model(path)
    assert saved is not None
    assert store.current is not None
    assert saved.trained_at == store.current.trained_at
    assert [file.name for file in tmp_path.iterdir()] == ["model.joblib"]


def test_status_shows_model_type_and_hyperparameters(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    store = ModelStore(tmp_path / "model.joblib")
    store.save(trained_model)

    status = store.status()

    assert status.model_type == trained_model.model_type
    assert status.hyperparameters == trained_model.hyperparameters


def test_require_current_returns_model(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    store = ModelStore(tmp_path / "model.joblib")
    store.save(trained_model)

    assert store.require_current() is trained_model


def test_require_current_without_model_raises(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "model.joblib")

    with pytest.raises(ModelNotTrainedError):
        store.require_current()
