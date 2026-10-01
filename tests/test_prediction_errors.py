import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES

WITHOUT_REGION = {k: v for k, v in EXAMPLE_FEATURES.items() if k != "region"}


def error_types(body: dict) -> list[tuple[list, str]]:
    return [(item["location"], item["type"]) for item in body["details"]]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {**EXAMPLE_FEATURES, "extra": 1},
            [(["body", "client", "extra"], "extra_forbidden")],
        ),
        (WITHOUT_REGION, [(["body", "client", "region"], "missing")]),
        (
            {**EXAMPLE_FEATURES, "monthly_fee": "abc", "region": 5},
            [
                (["body", "client", "monthly_fee"], "float_type"),
                (["body", "client", "region"], "string_type"),
            ],
        ),
        (
            [EXAMPLE_FEATURES, {**EXAMPLE_FEATURES, "usage_hours": -1}],
            [(["body", "clients", 1, "usage_hours"], "greater_than_equal")],
        ),
        ("not a client", [(["body", "client"], "model_attributes_type")]),
    ],
)
def test_invalid_request_is_reported_field_by_field(
    trained_client: TestClient,
    payload: JsonValue,
    expected: list[tuple[list, str]],
) -> None:
    response = trained_client.post("/predict", json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert error_types(response.json()) == expected


def test_invalid_request_wins_over_missing_model(client: TestClient) -> None:
    response = client.post("/predict", json=WITHOUT_REGION)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_valid_request_without_model(client: TestClient) -> None:
    response = client.post("/predict", json=EXAMPLE_FEATURES)

    assert response.status_code == 409
    assert response.json()["code"] == "model_not_trained"


@pytest.mark.parametrize(
    ("name", "value", "error_type"),
    [
        ("monthly_fee", "19.99", "float_type"),
        ("monthly_fee", True, "float_type"),
        ("autopay_enabled", True, "int_type"),
        ("support_requests", 1.5, "int_type"),
        ("region", 5, "string_type"),
    ],
)
def test_wrong_value_types_are_not_converted(
    trained_client: TestClient, name: str, value: JsonValue, error_type: str
) -> None:
    response = trained_client.post(
        "/predict", json={**EXAMPLE_FEATURES, name: value}
    )

    assert response.status_code == 422
    assert error_types(response.json()) == [
        (["body", "client", name], error_type)
    ]


def test_schema_limits_are_integers_for_integer_fields(
    client: TestClient,
) -> None:
    specs = {
        spec["name"]: spec
        for spec in client.get("/model/schema").json()["features"]
    }

    assert specs["autopay_enabled"]["minimum"] == 0
    assert isinstance(specs["autopay_enabled"]["maximum"], int)
