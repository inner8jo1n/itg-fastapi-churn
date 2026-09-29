from fastapi import APIRouter

from itg_fastapi_churn.schemas.churn import FeatureVectorChurn

router = APIRouter(tags=["prediction"])


@router.post("/predict")
def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    """
    Echo the received features until the model is available

    :features: FeatureVectorChurn - client features to predict churn for

    :return: the same features that were received
    """
    return features
