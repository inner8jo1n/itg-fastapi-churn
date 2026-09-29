from fastapi import APIRouter

from itg_fastapi_churn.api.dependencies import SplitDep
from itg_fastapi_churn.ml.metrics import evaluate_model
from itg_fastapi_churn.ml.model import train_churn_model
from itg_fastapi_churn.schemas.model import ModelMetrics

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def train_model(split: SplitDep) -> ModelMetrics:
    model = train_churn_model(split.x_train, split.y_train)
    return evaluate_model(
        model=model, features=split.x_test, target=split.y_test
    )
