from fastapi import FastAPI

from itg_fastapi_churn import __version__
from itg_fastapi_churn.api.routers import dataset, model, prediction, root


def create_app() -> FastAPI:
    """
    Build the FastAPI application with all routers attached

    :return: configured FastAPI application
    """
    application = FastAPI(title="ML Churn Service", version=__version__)
    application.include_router(root.router)
    application.include_router(prediction.router)
    application.include_router(dataset.router)
    application.include_router(model.router)
    return application


app = create_app()
