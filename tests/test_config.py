from pathlib import Path

import pytest

from itg_fastapi_churn.config import Settings


def test_settings_use_default_dataset_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHURN_DATASET_PATH", raising=False)

    settings = Settings()

    assert settings.dataset_path == Path("data/churn_dataset.csv")


def test_settings_read_dataset_path_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHURN_DATASET_PATH", "data/example_path")

    settings = Settings()

    assert settings.dataset_path == Path("data/example_path")
