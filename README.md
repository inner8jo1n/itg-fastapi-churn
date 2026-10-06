# itg-fastapi-churn

A FastAPI service that trains a model on customer data and predicts
**churn**: whether a customer is likely to cancel the subscription.

The goal is to find customers at risk early, while there is still time to
keep them (a discount, a call from support, help with a failed payment).
The service:

- reads the training dataset `data/churn_dataset.csv` and validates it;
- trains a scikit-learn model (logistic regression or random forest) and
  reports its quality on a held-out test split;
- saves the model to disk, so it survives restarts;
- predicts churn and its probability for one customer or a list;
- keeps a history of trainings to compare model settings;
- reports its own health and logs key events, ready to run in Docker.

## Contents

- [Endpoints](#endpoints)
- [Dataset format](#dataset-format)
- [Running locally](#running-locally)
- [Running in Docker](#running-in-docker)
- [Request examples](#request-examples)
- [Errors](#errors)
- [Model](#model)
- [Logging](#logging)
- [Project structure](#project-structure)
- [Development](#development)

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | the service is running |
| GET | `/health` | is the model available and can the dataset be read |
| GET | `/dataset/preview?n=5` | first `n` dataset rows (1–100) |
| GET | `/dataset/info` | dataset size, features and class balance |
| GET | `/dataset/split-info` | sizes and class balance of the train/test split |
| POST | `/model/train` | train, evaluate and save the model |
| GET | `/model/status` | is a model trained, when and with what metrics |
| GET | `/model/metrics` | latest, best and recent trainings |
| GET | `/model/schema` | features `/predict` expects, with limits and known categories |
| POST | `/predict` | churn prediction for one customer or a list |

Interactive documentation with request examples and documented error
responses: `/docs` (Swagger UI) and `/redoc`.

## Dataset format

`data/churn_dataset.csv` is a CSV file with a header row. One row is one
customer; the file in the repository has 2000 rows, 20% of them churned.

| Column | Type | Allowed values | Meaning |
| --- | --- | --- | --- |
| `monthly_fee` | float | ≥ 0 | monthly tariff price |
| `usage_hours` | float | ≥ 0 | service usage over the last month, hours |
| `support_requests` | int | ≥ 0 | number of support requests |
| `account_age_months` | int | ≥ 0 | account age, months |
| `failed_payments` | int | ≥ 0 | number of failed payments |
| `region` | text | `europe`, `asia`, `america`, `africa` | customer region |
| `device_type` | text | `mobile`, `desktop`, `tablet` | main device |
| `payment_method` | text | `card`, `paypal`, `crypto` | payment method |
| `autopay_enabled` | int | `0` or `1` | autopay is on |
| `churn` | int | `0` or `1` | **target**: 1 if the customer left |

```csv
monthly_fee,usage_hours,support_requests,account_age_months,failed_payments,region,device_type,payment_method,autopay_enabled,churn
9.99,27.92,1,14,1,america,desktop,card,1,1
19.99,21.48,2,1,0,america,mobile,card,1,0
```

Rules applied when the file is loaded:

- every row is validated; a file with invalid rows is rejected as a
  whole, and the error lists the first invalid rows and fields;
- extra columns (for example a customer id) are ignored, missing columns
  are an error;
- training needs both classes and enough rows for a stratified split.

## Running locally

Requirements: Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # create .venv with all dependencies
make run           # start the server with auto-reload on port 8000
```

Then open http://127.0.0.1:8000/docs. A fresh service has no model yet:
call `POST /model/train` first, then `POST /predict`.

### Settings

All settings have defaults and can be changed with environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHURN_DATASET_PATH` | `data/churn_dataset.csv` | training dataset |
| `CHURN_TEST_SIZE` | `0.2` | share of rows held out for the test split |
| `CHURN_RANDOM_STATE` | `42` | seed of the train/test split |
| `CHURN_MODEL_PATH` | `models/churn_model.joblib` | where the trained model is saved |
| `CHURN_HISTORY_PATH` | `models/training_history.jsonl` | training history file |
| `CHURN_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`, any case |

Relative paths are resolved from the directory the server is started in.

## Running in Docker

```bash
make docker-build  # docker build -t itg-fastapi-churn .
make docker-run    # run on port 8000, models kept in the churn-models volume
```

`make docker-run` is the same as:

```bash
docker run --rm -p 8000:8000 -v churn-models:/app/models --name churn itg-fastapi-churn
```

- The image contains only the installed package and `data/`; tests and
  development tools are not included. The service runs as a non-root user.
- `/app/models` is a volume, so the trained model and the training history
  survive container restarts and re-creation.
- Docker checks `GET /health` every 30 seconds; `docker ps` shows the
  container as `healthy` when the service answers.
- Settings are passed with `-e`, for example
  `-e CHURN_LOG_LEVEL=warning`; another dataset can be mounted with
  `-v /path/to/data:/app/data`.
- Logs: `docker logs churn`.

If the build fails with `docker-credential-desktop: executable file not
found in $PATH`, the Docker Desktop helpers are missing from `PATH`:

```bash
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
```

## Request examples

The responses below come from the service trained on the dataset from the
repository.

### Train the model — `POST /model/train`

Without a body, logistic regression with the default settings is trained:

```bash
curl -X POST http://127.0.0.1:8000/model/train
```

```json
{"accuracy": 0.59, "f1": 0.3543307086614173, "roc_auc": 0.6111691628933009, "warnings": []}
```

Choose the model and override its scikit-learn hyperparameters:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 6}}'
```

```json
{"accuracy": 0.6125, "f1": 0.30493273542600896, "roc_auc": 0.6102403343782655, "warnings": []}
```

- `model_type`: `logreg` (default) or `random_forest`.
- `hyperparameters`: parameters of
  [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  or [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
  Unknown names and invalid values are rejected with 422. `n_estimators`
  is limited to 1000, `max_iter` to 10000 and `n_jobs` to 16.
- `warnings` lists problems that did not stop training, for example
  that the model did not converge.

### Check the model — `GET /model/status`

```bash
curl http://127.0.0.1:8000/model/status
```

```json
{
  "is_trained": true,
  "trained_at": "2026-10-06T19:25:44.352545Z",
  "metrics": {"accuracy": 0.5875, "f1": 0.34782608695652173, "roc_auc": 0.6113626688339332},
  "model_type": "logreg",
  "hyperparameters": {"max_iter": 1000, "class_weight": "balanced", "C": 0.5}
}
```

### Predict churn — `POST /predict`

One customer:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"monthly_fee": 19.99, "usage_hours": 42.5, "support_requests": 1,
       "account_age_months": 14, "failed_payments": 0, "region": "europe",
       "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1}'
```

```json
{"predictions": [{"churn": 0, "probabilities": {"0": 0.8089232443237894, "1": 0.19107675567621066}}]}
```

A list of up to 1000 customers; predictions come back in the same order:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"monthly_fee": 19.99, "usage_hours": 42.5, "support_requests": 1,
        "account_age_months": 14, "failed_payments": 0, "region": "europe",
        "device_type": "mobile", "payment_method": "card", "autopay_enabled": 1},
       {"monthly_fee": 19.99, "usage_hours": 3.5, "support_requests": 4,
        "account_age_months": 14, "failed_payments": 2, "region": "europe",
        "device_type": "mobile", "payment_method": "card", "autopay_enabled": 0}]'
```

```json
{"predictions": [
  {"churn": 0, "probabilities": {"0": 0.8089232443237894, "1": 0.19107675567621066}},
  {"churn": 1, "probabilities": {"0": 0.10213978154071635, "1": 0.8978602184592837}}
]}
```

`churn` is the predicted class, `probabilities` gives the probability of
each class. All nine features are required, types are strict (a number
sent as text is an error) and unknown fields are rejected. A category the
model has not seen during training is accepted and simply does not add to
the prediction; `GET /model/schema` lists the known categories.

### Compare trainings — `GET /model/metrics`

```bash
curl "http://127.0.0.1:8000/model/metrics?limit=5&model_type=random_forest"
```

Returns `total` (number of matching trainings), `latest`, `best` (highest
F1) and `recent` (up to `limit` latest trainings, 1–100, default 10). Each
record has the training time, model type, hyperparameters and metrics.
`model_type` is optional.

### Health — `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model": {"available": true, "model_type": "logreg", "trained_at": "2026-10-06T19:25:44.352545Z"},
  "dataset": {"available": true, "problem": null}
}
```

`status` is `degraded` when the model is not trained yet or the dataset
cannot be read; `dataset.problem` then holds the error code. The answer is
always 200 while the service runs. The dataset check reads only the header
and the first row, so a broken row further down is found on training.

## Errors

Every error has the same format:

```json
{"code": "model_not_trained", "message": "Model is not trained yet, call POST /model/train first", "details": null}
```

| Status | Codes | When |
| --- | --- | --- |
| 404 | `dataset_not_found` | the dataset file does not exist |
| 409 | `dataset_empty`, `dataset_invalid`, `not_enough_data` | the dataset cannot be used for training |
| 409 | `model_not_trained`, `incompatible_model` | `/predict` without a suitable model |
| 422 | `validation_error` | the request is invalid; `details` lists the fields |
| 422 | `invalid_hyperparameters` | scikit-learn rejected the hyperparameters |
| 500 | `prediction_failed`, `history_unavailable`, `internal_error` | server-side failure; details are only in the server log |

Example of `validation_error`:

```json
{"code": "validation_error", "message": "Request data is invalid",
 "details": [{"location": ["body", "client", "monthly_fee"], "message": "Input should be a valid number",
              "type": "float_type", "input": "19.99"}]}
```

## Model

- **Pipeline**: numeric features are scaled with `StandardScaler`,
  categorical ones are one-hot encoded (unknown categories are ignored),
  then the classifier. Preprocessing is part of the saved model, so it is
  fitted on the training rows only.
- **Split**: stratified 80/20 train/test split with a fixed seed, so the
  class balance is the same in both parts and results are reproducible.
- **Defaults**: logistic regression with `class_weight="balanced"` and
  `max_iter=1000`; random forest with 100 trees,
  `class_weight="balanced"` and `random_state=42`.
- **Metrics** on the test split: accuracy, F1 of the churn class and ROC
  AUC. Only 20% of customers churn, so accuracy alone is misleading: a
  model that always answers "stays" has accuracy 0.8 and F1 0. Compare
  models by F1 and ROC AUC.
- **Storage**: the model is saved with joblib (atomic write), the
  training history is a JSON Lines file. Only load model files written by
  this service: joblib runs code from the file while loading.

## Logging

Service events are written to stdout, one line each:

```text
2026-10-06 21:41:03,860 INFO itg_fastapi_churn.dataset.loader: Dataset loaded from data/churn_dataset.csv: 2000 rows
2026-10-06 21:41:03,872 INFO itg_fastapi_churn.api.routers.model: Training logreg model
2026-10-06 21:41:03,887 INFO itg_fastapi_churn.api.routers.model: Model logreg trained in 0.01 s: accuracy=0.59 f1=0.3543307086614173 roc_auc=0.6111691628933009
2026-10-06 21:41:03,895 INFO itg_fastapi_churn.api.routers.prediction: Predicted churn for 1 clients in 2.0 ms: 1 likely to leave
2026-10-06 21:41:03,911 WARNING itg_fastapi_churn.api.errors: POST /model/train failed with invalid_hyperparameters: ...
```

Logged events: model loading on start, dataset loading, training start
and result, predictions (count only, never customer data), client errors
as `WARNING` and server errors as `ERROR` with the traceback. Request
access lines come from uvicorn.

## Project structure

```text
src/itg_fastapi_churn/
├── main.py              # create_app(): settings, logging, routers, model loading on start
├── cli.py, __main__.py  # `python -m itg_fastapi_churn` starts uvicorn
├── core/                # shared by all layers
│   ├── config.py        # Settings from CHURN_* environment variables
│   ├── errors.py        # ServiceError hierarchy with stable error codes
│   └── logging_config.py
├── api/                 # HTTP layer
│   ├── routers/         # root, health, dataset, model, prediction endpoints
│   ├── dependencies.py  # settings, dataset, split, model store, history
│   ├── errors.py        # error responses in the common format
│   └── error_docs.py    # documented error examples for /docs
├── dataset/             # reading and validating the CSV
├── ml/                  # churn pipeline: features, split, model, metrics,
│                        # prediction, persistence, model store, history
└── schemas/             # pydantic request and response models
tests/                   # pytest suite, tests/data/churn_sample.csv
data/churn_dataset.csv   # training dataset
Dockerfile, .dockerignore
```

The layers depend in one direction: `api` → `dataset` → `ml` → `schemas`
/ `core`. Only `api` knows about FastAPI; `dataset`, `ml`, `schemas` and
`core` can be used and tested without it.

## Development

```bash
uv sync            # install dev dependencies too
make check         # lint + format check + type check + tests
```

| Command | Description |
| --- | --- |
| `make run` | start the dev server with auto-reload |
| `make test` | run pytest |
| `make lint` | run ruff check |
| `make format` | run ruff format |
| `make fix` | auto-fix lint issues and format |
| `make typecheck` | run ty check |
| `make check` | run all of the above (`scripts/check.sh`) |
| `make docker-build` | build the Docker image |
| `make docker-run` | run the image on port 8000 |

Tests run without network and without the real model files:

- unit tests of data loading, features, split, training, metrics and
  persistence, without FastAPI;
- API tests with `TestClient`, including the full flow "read CSV → train
  → status → predict" on `tests/data/churn_sample.csv`, an 80-row
  stratified sample of the dataset;
- error handling, reproducibility of training, logging and health.

pytest treats warnings as errors and rejects unknown markers.
