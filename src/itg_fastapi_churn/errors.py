from typing import ClassVar

from pydantic import JsonValue


class ServiceError(Exception):
    """
    Base of all expected churn service errors

    Every subclass has a stable machine readable code and a default
    message, so clients can react to the code and show the message.

    :code: str - machine readable error code
    :default_message: str - message used when none is given
    """

    code: ClassVar[str] = "service_error"
    default_message: ClassVar[str] = "Churn service error"

    def __init__(
        self, message: str | None = None, details: JsonValue = None
    ) -> None:
        """
        Create the error with a message and optional details

        :message: str | None - human readable description
        :details: JsonValue - extra data that helps to fix the problem
        """
        self.message = message or self.default_message
        self.details = details
        super().__init__(self.message)


class DatasetNotFoundError(ServiceError):
    """
    Training dataset file does not exist
    """

    code = "dataset_not_found"
    default_message = "Dataset file not found"


class EmptyDatasetError(ServiceError):
    """
    Training dataset has no rows
    """

    code = "dataset_empty"
    default_message = "Dataset is empty"


class InvalidDatasetError(ServiceError):
    """
    Dataset file exists but its content cannot be used
    """

    code = "dataset_invalid"
    default_message = "Dataset file has invalid content"


class NotEnoughDataError(ServiceError):
    """
    Dataset has rows, but not enough to split it and train a model
    """

    code = "not_enough_data"
    default_message = "Not enough data to train the model"


class ModelNotTrainedError(ServiceError):
    """
    Prediction was requested before any model was trained
    """

    code = "model_not_trained"
    default_message = "Model is not trained yet, call POST /model/train first"


class IncompatibleModelError(ServiceError):
    """
    Loaded model was trained on other features than the API accepts
    """

    code = "incompatible_model"
    default_message = (
        "Model was trained on other features, "
        "call POST /model/train to retrain it"
    )


class PredictionFailedError(ServiceError):
    """
    Model failed while predicting for valid client data
    """

    code = "prediction_failed"
    default_message = "Model failed to make a prediction"


class InvalidHyperparametersError(ServiceError):
    """
    Hyperparameters do not fit the chosen classifier
    """

    code = "invalid_hyperparameters"
    default_message = "Hyperparameters are not valid for this model"
