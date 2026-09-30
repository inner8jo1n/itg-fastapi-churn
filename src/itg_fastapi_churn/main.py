from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from itg_fastapi_churn import __version__
from itg_fastapi_churn.api.errors import register_error_handlers
from itg_fastapi_churn.api.routers import dataset, model, prediction, root
from itg_fastapi_churn.config import get_settings
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
    Build the FastAPI application with all routers attached

    :return: configured FastAPI application
    """
    application = FastAPI(
        title="ML Churn Service", version=__version__, lifespan=lifespan
    )
    application.state.model_store = ModelStore(get_settings().model_path)
    register_error_handlers(application)
    application.include_router(root.router)
    application.include_router(prediction.router)
    application.include_router(dataset.router)
    application.include_router(model.router)
    return application


app = create_app()
