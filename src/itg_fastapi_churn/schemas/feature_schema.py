from enum import StrEnum

from pydantic import BaseModel


class ValueType(StrEnum):
    """
    JSON type a feature value must have

    :NUMBER: str - any number
    :INTEGER: str - whole number
    :STRING: str - text
    """

    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"


class FeatureKind(StrEnum):
    """
    How the model treats a feature

    :NUMERIC: str - scaled number
    :CATEGORICAL: str - one-hot encoded category
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class FeatureSpec(BaseModel):
    """
    Description of one churn feature for clients building requests

    :name: str - field name in the request
    :type: ValueType - JSON type of the value
    :kind: FeatureKind - numeric or categorical
    :minimum: int | float | None - smallest allowed value, if limited
    :maximum: int | float | None - largest allowed value, if limited
    :known_values: list[str] | None - categories the trained model has
        seen; other values are accepted but ignored by the model
    """

    name: str
    type: ValueType
    kind: FeatureKind
    minimum: int | float | None = None
    maximum: int | float | None = None
    known_values: list[str] | None = None


class ModelSchemaResponse(BaseModel):
    """
    Features the churn model expects in POST /predict

    :features: list[FeatureSpec] - every feature of a client
    :target: str - name of the predicted value
    """

    features: list[FeatureSpec]
    target: str
