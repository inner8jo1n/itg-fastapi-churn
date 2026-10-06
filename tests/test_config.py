from pathlib import Path

import pytest
from pydantic import ValidationError

from itg_fastapi_churn.core.config import Settings


def test_settings_use_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHURN_DATASET_PATH", raising=False)
    monkeypatch.delenv("CHURN_TEST_SIZE", raising=False)
    monkeypatch.delenv("CHURN_RANDOM_STATE", raising=False)
    monkeypatch.delenv("CHURN_MODEL_PATH", raising=False)
    monkeypatch.delenv("CHURN_HISTORY_PATH", raising=False)
    monkeypatch.delenv("CHURN_LOG_LEVEL", raising=False)

    settings = Settings()

    assert settings.dataset_path == Path("data/churn_dataset.csv")
    assert settings.test_size == 0.2
    assert settings.random_state == 42
    assert settings.model_path == Path("models/churn_model.joblib")
    assert settings.history_path == Path("models/training_history.jsonl")
    assert settings.log_level == "INFO"


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


def test_settings_accept_log_level_in_any_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHURN_LOG_LEVEL", "debug")

    assert Settings().log_level == "DEBUG"


def test_settings_reject_unknown_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHURN_LOG_LEVEL", "verbose")

    with pytest.raises(ValidationError):
        Settings()
