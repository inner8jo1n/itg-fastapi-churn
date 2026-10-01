import logging
import threading
from pathlib import Path

from itg_fastapi_churn.errors import ModelNotTrainedError
from itg_fastapi_churn.ml.persistence import (
    ModelLoadError,
    TrainedModel,
    load_churn_model,
    save_churn_model,
)
from itg_fastapi_churn.schemas.model import ModelStatus

logger = logging.getLogger(__name__)


class ModelStore:
    """
    Keeps the current churn model in memory and in a file on disk

    Saving is guarded by a lock: requests run in parallel threads, and
    without it two trainings could leave one model in memory and another
    one on disk.
    """

    def __init__(self, path: Path) -> None:
        """
        Create an empty store; call load to read a saved model

        :path: Path - file where the model is saved
        """
        self._path = path
        self._model: TrainedModel | None = None
        self._lock = threading.Lock()

    @property
    def current(self) -> TrainedModel | None:
        """
        Model kept in memory

        :return: current model, or None if there is none yet
        """
        return self._model

    def require_current(self) -> TrainedModel:
        """
        Model kept in memory, for operations that cannot work without it

        Raises ModelNotTrainedError while no model has been trained.

        :return: current model
        """
        model = self._model
        if model is None:
            raise ModelNotTrainedError()
        return model

    def load(self) -> None:
        """
        Read the saved model from disk, if the file exists

        A damaged file is logged and skipped, so the service still starts
        and the model can be trained again.
        """
        try:
            self._model = load_churn_model(self._path)
        except ModelLoadError:
            logger.warning("Saved model is ignored", exc_info=True)
            self._model = None

    def save(self, model: TrainedModel) -> None:
        """
        Save the model to disk first, then keep it in memory

        :model: TrainedModel - freshly trained model
        """
        with self._lock:
            save_churn_model(model, self._path)
            self._model = model

    def status(self) -> ModelStatus:
        """
        Describe the current model

        :return: whether a model exists, when it was trained, its metrics,
            type and hyperparameters
        """
        model = self._model
        if model is None:
            return ModelStatus(is_trained=False)

        return ModelStatus(
            is_trained=True,
            trained_at=model.trained_at,
            metrics=model.metrics,
            model_type=model.model_type,
            hyperparameters=model.hyperparameters,
        )
