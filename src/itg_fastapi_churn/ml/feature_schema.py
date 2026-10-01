from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from itg_fastapi_churn.ml.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from itg_fastapi_churn.schemas.churn import FeatureVectorChurn
from itg_fastapi_churn.schemas.feature_schema import (
    FeatureKind,
    FeatureSpec,
    ModelSchemaResponse,
)


def describe_features(pipeline: Pipeline | None) -> ModelSchemaResponse:
    """
    Describe the churn features: types and limits come from the request
    schema, known categories come from the trained model

    :pipeline: Pipeline | None - trained pipeline, None if not trained yet

    :return: description of every feature and the target name
    """
    properties = FeatureVectorChurn.model_json_schema()["properties"]
    categories = _known_categories(pipeline)

    return ModelSchemaResponse(
        features=[
            FeatureSpec(
                name=name,
                type=properties[name]["type"],
                kind=(
                    FeatureKind.CATEGORICAL
                    if name in CATEGORICAL_FEATURES
                    else FeatureKind.NUMERIC
                ),
                minimum=properties[name].get("minimum"),
                maximum=properties[name].get("maximum"),
                known_values=categories.get(name),
            )
            for name in FEATURE_COLUMNS
        ],
        target=TARGET_COLUMN,
    )


def _known_categories(pipeline: Pipeline | None) -> dict[str, list[str]]:
    """
    Read the categories the one-hot encoder learned during training

    Names are taken from the encoder itself, so a model built by another
    version of the service never breaks the schema: categories it does
    not describe are simply left unknown.

    :pipeline: Pipeline | None - trained pipeline, None if not trained yet

    :return: categories of every categorical feature the model knows
    """
    encoder = _categorical_encoder(pipeline)
    names = getattr(encoder, "feature_names_in_", None)
    if encoder is None or names is None:
        return {}

    return {
        str(name): [str(value) for value in values]
        for name, values in zip(names, encoder.categories_, strict=True)
    }


def _categorical_encoder(pipeline: Pipeline | None) -> OneHotEncoder | None:
    """
    Find the fitted one-hot encoder inside the pipeline

    :pipeline: Pipeline | None - trained pipeline, None if not trained yet

    :return: fitted encoder, or None if the pipeline has none
    """
    if pipeline is None:
        return None

    preprocess = pipeline.named_steps.get("preprocess")
    transformers = getattr(preprocess, "named_transformers_", {})
    encoder = transformers.get("categorical")
    return encoder if isinstance(encoder, OneHotEncoder) else None
