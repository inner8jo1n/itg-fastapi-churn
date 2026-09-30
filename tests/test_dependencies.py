from pathlib import Path

import pytest
from fastapi import HTTPException

from itg_fastapi_churn.api.dependencies import (
    MODEL_NOT_TRAINED,
    get_trained_model,
)
from itg_fastapi_churn.ml.persistence import TrainedModel
from itg_fastapi_churn.ml.store import ModelStore


def test_get_trained_model_returns_current_model(
    trained_model: TrainedModel, tmp_path: Path
) -> None:
    store = ModelStore(tmp_path / "model.joblib")
    store.save(trained_model)

    assert get_trained_model(store) is trained_model


def test_get_trained_model_rejects_untrained_store(tmp_path: Path) -> None:
    store = ModelStore(tmp_path / "model.joblib")

    with pytest.raises(HTTPException) as raised:
        get_trained_model(store)

    assert raised.value.status_code == 409
    assert raised.value.detail == MODEL_NOT_TRAINED
