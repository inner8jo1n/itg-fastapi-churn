from pydantic import BaseModel


class ModelMetrics(BaseModel):
    """
    Quality of the churn model on the test split

    :accuracy: float - share of correct predictions
    :f1: float - F1 score of the churn class (1)
    """

    accuracy: float
    f1: float
