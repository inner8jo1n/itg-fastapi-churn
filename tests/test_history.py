from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from itg_fastapi_churn.core.errors import HistoryUnavailableError
from itg_fastapi_churn.ml.history import TrainingHistory
from itg_fastapi_churn.schemas.history import (
    TrainingMetricsResponse,
    TrainingRecord,
)
from itg_fastapi_churn.schemas.model import ModelMetrics
from itg_fastapi_churn.schemas.training import ModelType


def make_record(
    day: int, model_type: ModelType = ModelType.LOGREG, f1: float = 0.3
) -> TrainingRecord:
    return TrainingRecord(
        trained_at=datetime(2026, 1, day, tzinfo=UTC),
        model_type=model_type,
        hyperparameters={"C": 0.5},
        metrics=ModelMetrics(accuracy=0.6, f1=f1, roc_auc=0.7),
    )


def test_history_is_empty_without_file(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path / "history.jsonl")

    assert history.records() == []


def test_records_come_newest_first(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path / "history.jsonl")
    for day in (1, 2, 3):
        history.append(make_record(day))

    days = [record.trained_at.day for record in history.records()]

    assert days == [3, 2, 1]


def test_history_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "models" / "history.jsonl"
    TrainingHistory(path).append(make_record(1))

    restarted = TrainingHistory(path)

    assert restarted.records() == [make_record(1)]


def test_records_can_be_filtered_by_model_type(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path / "history.jsonl")
    history.append(make_record(1, ModelType.LOGREG))
    history.append(make_record(2, ModelType.RANDOM_FOREST))
    history.append(make_record(3, ModelType.LOGREG))

    forest = history.records(ModelType.RANDOM_FOREST)

    assert [record.trained_at.day for record in forest] == [2]


def test_damaged_lines_are_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.jsonl"
    history = TrainingHistory(path)
    history.append(make_record(1))
    with path.open("a", encoding="utf-8") as file:
        file.write("not json\n\n")
        file.write('{"trained_at": "2026-01-02T00:00:00Z", "model_ty')
    history.append(make_record(3))

    days = [record.trained_at.day for record in history.records()]

    assert days == [3, 1]
    assert "Damaged history line 2" in caplog.text


@pytest.mark.parametrize(
    "line",
    [
        '{"trained_at": "2026-01-02T00:00:00", "model_type": "logreg", '
        '"hyperparameters": {}, "metrics": {"accuracy": 1, "f1": 1}}',
        '{"trained_at": "2026-01-02T00:00:00Z", "model_type": "logreg", '
        '"hyperparameters": {}, "metrics": {"accuracy": "NaN", "f1": 5}}',
    ],
)
def test_record_without_time_zone_or_with_bad_metrics_is_skipped(
    tmp_path: Path, line: str
) -> None:
    path = tmp_path / "history.jsonl"
    path.write_text(line + "\n", encoding="utf-8")

    assert TrainingHistory(path).records() == []


def test_records_are_ordered_by_training_time(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path / "history.jsonl")
    for day in [1, 3, 2]:
        history.append(make_record(day))

    days = [record.trained_at.day for record in history.records()]

    assert days == [3, 2, 1]


def test_records_with_equal_time_keep_newest_append_first(
    tmp_path: Path,
) -> None:
    history = TrainingHistory(tmp_path / "history.jsonl")
    history.append(make_record(1, f1=0.1))
    history.append(make_record(1, f1=0.2))

    scores = [record.metrics.f1 for record in history.records()]

    assert scores == [0.2, 0.1]


def test_blank_lines_are_not_reported_as_damaged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "history.jsonl"
    history = TrainingHistory(path)
    history.append(make_record(1))
    with path.open("a", encoding="utf-8") as file:
        file.write("\n  \n")

    assert len(history.records()) == 1
    assert "Damaged" not in caplog.text


def test_append_to_existing_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    path.touch()
    history = TrainingHistory(path)

    history.append(make_record(1))

    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert len(history.records()) == 1


def test_broken_encoding_skips_only_its_line(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    history = TrainingHistory(path)
    history.append(make_record(1))
    with path.open("ab") as file:
        file.write(b'{"model_type": "\xff"}\n')
    history.append(make_record(3))

    days = [record.trained_at.day for record in history.records()]

    assert days == [3, 1]


def test_unreadable_history_raises(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path)

    with pytest.raises(HistoryUnavailableError):
        history.records()


def test_unwritable_history_raises(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path)

    with pytest.raises(HistoryUnavailableError):
        history.append(make_record(1))


def test_parallel_appends_keep_every_record(tmp_path: Path) -> None:
    history = TrainingHistory(tmp_path / "history.jsonl")
    records = [make_record(day % 28 + 1, f1=day / 100) for day in range(50)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(history.append, records))

    stored = history.records()
    assert len(stored) == 50
    assert {record.metrics.f1 for record in stored} == {
        record.metrics.f1 for record in records
    }


def test_summary_of_empty_history() -> None:
    summary = TrainingMetricsResponse.from_records([], limit=5)

    assert summary.total == 0
    assert summary.latest is None
    assert summary.best is None
    assert summary.recent == []


def test_summary_limits_only_recent_records() -> None:
    records = [make_record(3, f1=0.2), make_record(2), make_record(1, f1=0.9)]

    summary = TrainingMetricsResponse.from_records(records, limit=1)

    assert summary.total == 3
    assert summary.latest == records[0]
    assert summary.best == records[2]
    assert summary.recent == [records[0]]


def test_summary_prefers_newest_of_equal_best() -> None:
    records = [make_record(2), make_record(1)]

    summary = TrainingMetricsResponse.from_records(records, limit=5)

    assert summary.best == records[0]
