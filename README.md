# itg-fastapi-churn

## Development

```bash
uv sync            # create .venv and install dev dependencies
make check         # lint + format check + type check + tests
```

| Command | Description |
| --- | --- |
| `make test` | run pytest |
| `make lint` | run ruff check |
| `make format` | run ruff format |
| `make typecheck` | run ty check |
| `make check` | run `scripts/check.sh` (all of the above) |
