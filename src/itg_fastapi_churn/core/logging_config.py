import logging
import sys

SERVICE_LOGGER = "itg_fastapi_churn"
HANDLER_NAME = "churn_service"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str) -> None:
    """
    Send service log messages to stdout in one format

    Only the service logger is configured, so uvicorn keeps its own
    logs. Calling it again replaces the handler instead of adding a
    second one, so messages are never printed twice.

    :level: str - lowest level to print, like "INFO"
    """
    logger = logging.getLogger(SERVICE_LOGGER)
    for handler in list(logger.handlers):
        if handler.get_name() == HANDLER_NAME:
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(HANDLER_NAME)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
