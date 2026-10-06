import importlib
import runpy

import pytest
import uvicorn
from fastapi import FastAPI

from itg_fastapi_churn import cli


def test_app_path_points_to_the_application() -> None:
    module_name, attribute = cli.APP_PATH.split(":")

    application = getattr(importlib.import_module(module_name), attribute)

    assert isinstance(application, FastAPI)


def test_main_starts_uvicorn_with_the_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    cli.main()

    assert calls == [((cli.APP_PATH,), {"host": "127.0.0.1", "port": 8000})]


def test_package_runs_as_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_: calls.append(app))

    runpy.run_module("itg_fastapi_churn", run_name="__main__")

    assert calls == [cli.APP_PATH]
