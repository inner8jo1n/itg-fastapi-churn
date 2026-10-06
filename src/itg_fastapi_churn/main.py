from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from itg_fastapi_churn import __version__
from itg_fastapi_churn.api.errors import register_error_handlers
from itg_fastapi_churn.api.routers import (
    dataset,
    health,
    model,
    prediction,
    root,
)
from itg_fastapi_churn.core.config import get_settings
from itg_fastapi_churn.core.logging_config import configure_logging
from itg_fastapi_churn.ml.history import TrainingHistory
from itg_fastapi_churn.ml.store import ModelStore


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """
    Load the saved model when the server starts

    :application: FastAPI - application being started
    """
    application.state.model_store.load()
    yield


def create_app() -> FastAPI:
    """
    Build the FastAPI application with logging and all routers set up

    :return: configured FastAPI application
    """
    application = FastAPI(
        title="ML Churn Service", version=__version__, lifespan=lifespan
    )
    settings = get_settings()
    configure_logging(settings.log_level)
    application.state.model_store = ModelStore(settings.model_path)
    application.state.training_history = TrainingHistory(settings.history_path)
    register_error_handlers(application)
    application.include_router(root.router)
    application.include_router(health.router)
    application.include_router(prediction.router)
    application.include_router(dataset.router)
    application.include_router(model.router)
    return application


app = create_app()
