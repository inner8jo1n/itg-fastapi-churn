import pytest
from pydantic import ValidationError
from pydantic.json_schema import JsonDict

from itg_fastapi_churn.schemas.training import (
    LOGREG_CONFIG_EXAMPLE,
    RANDOM_FOREST_CONFIG_EXAMPLE,
    ModelType,
    TrainingConfigChurn,
)


def test_config_defaults_to_logreg_without_hyperparameters() -> None:
    config = TrainingConfigChurn()

    assert config.model_type == ModelType.LOGREG
    assert config.hyperparameters == {}


@pytest.mark.parametrize(
    "example", [LOGREG_CONFIG_EXAMPLE, RANDOM_FOREST_CONFIG_EXAMPLE]
)
def test_config_examples_are_valid(example: JsonDict) -> None:
    config = TrainingConfigChurn.model_validate(example)

    assert config.model_type == example["model_type"]
    assert config.hyperparameters == example["hyperparameters"]


def test_config_rejects_unknown_model_type() -> None:
    with pytest.raises(ValidationError):
        TrainingConfigChurn.model_validate({"model_type": "randomforest"})


def test_model_type_is_a_string() -> None:
    assert ModelType.RANDOM_FOREST == "random_forest"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TrainingConfigChurn.model_validate({"hyperparams": {"C": 0.1}})
