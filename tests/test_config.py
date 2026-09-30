from pathlib import Path

import pytest
from pydantic import ValidationError

from itg_fastapi_churn.config import Settings


def test_settings_use_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHURN_DATASET_PATH", raising=False)
    monkeypatch.delenv("CHURN_TEST_SIZE", raising=False)
    monkeypatch.delenv("CHURN_RANDOM_STATE", raising=False)
    monkeypatch.delenv("CHURN_MODEL_PATH", raising=False)

    settings = Settings()

    assert settings.dataset_path == Path("data/churn_dataset.csv")
    assert settings.test_size == 0.2
    assert settings.random_state == 42
    assert settings.model_path == Path("models/churn_model.joblib")


def test_settings_read_dataset_path_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHURN_DATASET_PATH", "data/example_path")

    settings = Settings()

    assert settings.dataset_path == Path("data/example_path")


def test_settings_reject_invalid_test_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHURN_TEST_SIZE", "1.5")

    with pytest.raises(ValidationError):
        Settings()
