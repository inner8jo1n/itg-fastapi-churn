import logging
from collections.abc import Iterator

import pytest

from itg_fastapi_churn.core.config import get_settings
from itg_fastapi_churn.core.logging_config import (
    HANDLER_NAME,
    SERVICE_LOGGER,
    configure_logging,
)
from itg_fastapi_churn.main import create_app


@pytest.fixture
def service_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger(SERVICE_LOGGER)
    handlers, level = list(logger.handlers), logger.level
    yield logger
    logger.handlers = handlers
    logger.setLevel(level)


def service_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if h.get_name() == HANDLER_NAME]


def test_service_messages_go_to_stdout_in_one_format(
    service_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO")

    logging.getLogger(f"{SERVICE_LOGGER}.ml").info("Model trained")

    line = capsys.readouterr().out.strip()
    assert line.endswith(" INFO itg_fastapi_churn.ml: Model trained")


def test_messages_below_level_are_hidden(
    service_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("WARNING")

    service_logger.info("hidden")
    service_logger.warning("shown")

    assert capsys.readouterr().out.count("\n") == 1


def test_configuring_twice_keeps_one_handler(
    service_logger: logging.Logger,
) -> None:
    configure_logging("INFO")
    configure_logging("DEBUG")

    assert len(service_handlers(service_logger)) == 1
    assert service_logger.level == logging.DEBUG


def test_other_loggers_are_not_touched(service_logger: logging.Logger) -> None:
    uvicorn_handlers = list(logging.getLogger("uvicorn").handlers)

    configure_logging("INFO")

    assert logging.getLogger("uvicorn").handlers == uvicorn_handlers


def test_app_uses_level_from_settings(
    service_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("CHURN_LOG_LEVEL", "error")

    try:
        create_app()
    finally:
        get_settings.cache_clear()

    assert service_logger.level == logging.ERROR


def test_foreign_handlers_are_kept(service_logger: logging.Logger) -> None:
    foreign = logging.NullHandler()
    service_logger.addHandler(foreign)

    configure_logging("INFO")

    assert foreign in service_logger.handlers
