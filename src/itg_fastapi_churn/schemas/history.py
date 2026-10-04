from typing import Self

from pydantic import AwareDatetime, BaseModel, JsonValue

from itg_fastapi_churn.schemas.model import ModelMetrics
from itg_fastapi_churn.schemas.training import ModelType


class TrainingRecord(BaseModel):
    """
    One finished training of the churn model

    :trained_at: AwareDatetime - moment the training finished, with
        a time zone
    :model_type: ModelType - trained classifier
    :hyperparameters: dict[str, JsonValue] - hyperparameters used,
        service defaults included
    :metrics: ModelMetrics - quality on the test split
    """

    trained_at: AwareDatetime
    model_type: ModelType
    hyperparameters: dict[str, JsonValue]
    metrics: ModelMetrics


class TrainingMetricsResponse(BaseModel):
    """
    Result of GET /model/metrics: latest, best and recent trainings

    :total: int - number of trainings that match the filter
    :latest: TrainingRecord | None - most recent matching training
    :best: TrainingRecord | None - matching training with the highest
        F1 score; the newest one wins a tie
    :recent: list[TrainingRecord] - latest matching trainings, newest
        first
    """

    total: int
    latest: TrainingRecord | None
    best: TrainingRecord | None
    recent: list[TrainingRecord]

    @classmethod
    def from_records(cls, records: list[TrainingRecord], limit: int) -> Self:
        """
        Summarize trainings ordered from newest to oldest

        :records: list[TrainingRecord] - trainings, newest first
        :limit: int - how many recent trainings to list

        :return: latest, best and recent trainings
        """
        best = max(records, key=lambda record: record.metrics.f1, default=None)
        return cls(
            total=len(records),
            latest=records[0] if records else None,
            best=best,
            recent=records[:limit],
        )
