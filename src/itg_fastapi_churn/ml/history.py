import logging
import os
import threading
from pathlib import Path

from pydantic import ValidationError

from itg_fastapi_churn.core.errors import HistoryUnavailableError
from itg_fastapi_churn.schemas.history import TrainingRecord
from itg_fastapi_churn.schemas.training import ModelType

logger = logging.getLogger(__name__)


class TrainingHistory:
    """
    History of trainings kept in a JSON Lines file, one record per line

    Adding a record appends one line and never rewrites the file, so the
    history survives restarts and a crash can damage only the last line.
    Damaged lines are skipped when reading.
    """

    def __init__(self, path: Path) -> None:
        """
        Create the history stored in the given file

        :path: Path - JSON Lines file with the records
        """
        self._path = path
        self._lock = threading.Lock()

    def append(self, record: TrainingRecord) -> None:
        """
        Add a record to the end of the history

        If the last line was cut by a crash, the record starts on a new
        line, so the damage stays in that one line. Raises
        HistoryUnavailableError if the file cannot be written.

        :record: TrainingRecord - finished training
        """
        line = record.model_dump_json() + "\n"
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                if not self._ends_with_newline():
                    line = "\n" + line
                with self._path.open("a", encoding="utf-8") as file:
                    file.write(line)
            except OSError as error:
                raise HistoryUnavailableError(
                    "Training history cannot be written"
                ) from error

    def _ends_with_newline(self) -> bool:
        """
        Tell whether the file is empty or its last line is complete

        :return: True if a record can be appended right away
        """
        try:
            with self._path.open("rb") as file:
                if file.seek(0, os.SEEK_END) == 0:
                    return True
                file.seek(-1, os.SEEK_END)
                return file.read(1) == b"\n"
        except FileNotFoundError:
            return True

    def records(
        self, model_type: ModelType | None = None
    ) -> list[TrainingRecord]:
        """
        Read the history, newest record first

        Records are ordered by training time, not by file order: two
        trainings running at once may append in either order. Raises
        HistoryUnavailableError if the file exists but cannot be read.

        :model_type: ModelType | None - keep only this model type, all
            records if not given

        :return: matching records, newest first
        """
        records = [
            record
            for record in self._read_all()
            if model_type is None or record.model_type == model_type
        ]
        return sorted(
            reversed(records),
            key=lambda record: record.trained_at,
            reverse=True,
        )

    def _read_all(self) -> list[TrainingRecord]:
        """
        Read every valid record in file order, skipping damaged lines

        Lines are parsed as bytes, so broken UTF-8 damages only its line.
        The lock keeps a half-written line from being read as damaged.

        :return: records from oldest to newest
        """
        try:
            with self._lock:
                lines = self._path.read_bytes().splitlines()
        except FileNotFoundError:
            return []
        except OSError as error:
            raise HistoryUnavailableError(
                "Training history cannot be read"
            ) from error

        records = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                records.append(TrainingRecord.model_validate_json(line))
            except ValidationError:
                logger.warning(
                    "Damaged history line %s in %s is skipped",
                    number,
                    self._path,
                )
        return records
