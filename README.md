# itg-fastapi-churn

ML service predicting customer churn, built with FastAPI and scikit-learn.

## Running

```bash
make run           # uvicorn itg_fastapi_churn.main:app --reload
curl http://127.0.0.1:8000/
# {"message":"ml churn service is running"}
```

Interactive API docs: http://127.0.0.1:8000/docs

## Development

```bash
uv sync            # create .venv and install dev dependencies
make check         # lint + format check + type check + tests
```

| Command | Description |
| --- | --- |
| `make run` | start the dev server with auto-reload |
| `make test` | run pytest |
| `make lint` | run ruff check |
| `make format` | run ruff format |
| `make fix` | auto-fix lint issues (imports etc.) and format |
| `make typecheck` | run ty check |
| `make check` | run `scripts/check.sh` (all of the above) |
