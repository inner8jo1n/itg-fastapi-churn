FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies change rarely, so they get their own cached layer
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.13-slim

RUN useradd --create-home --uid 1000 churn

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY data ./data
RUN mkdir models && chown churn:churn models

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER churn

# The trained model and the training history survive container restarts
VOLUME ["/app/models"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "itg_fastapi_churn.main:app", "--host", "0.0.0.0", "--port", "8000"]
