.PHONY: run test lint format fix typecheck check docker-build docker-run

IMAGE = itg-fastapi-churn

run:
	uv run uvicorn itg_fastapi_churn.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

fix:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run ty check

check:
	@./scripts/check.sh

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 -v churn-models:/app/models --name churn $(IMAGE)
