from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import JsonDict

EXAMPLE_FEATURES: JsonDict = {
    "monthly_fee": 19.99,
    "usage_hours": 42.5,
    "support_requests": 1,
    "account_age_months": 14,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}


class FeatureVectorChurn(BaseModel):
    """
    Client features used to predict churn

    :monthly_fee: float - monthly tariff price
    :usage_hours: float - service usage hours over the last month
    :support_requests: int - number of support requests
    :account_age_months: int - account age in months
    :failed_payments: int - number of failed payments
    :region: str - client region (europe, asia, america, africa)
    :device_type: str - main device type (mobile, desktop, tablet)
    :payment_method: str - payment method (card, paypal, crypto)
    :autopay_enabled: int - whether autopay is enabled (0 or 1)

    Unknown fields are rejected, so a request with extra or misspelled
    features fails instead of being silently trimmed. Types are strict:
    a number sent as text or true/false sent as a number is an error.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        strict=True,
        json_schema_extra={"examples": [EXAMPLE_FEATURES]},
    )

    monthly_fee: float = Field(ge=0)
    usage_hours: float = Field(ge=0)
    support_requests: int = Field(ge=0)
    account_age_months: int = Field(ge=0)
    failed_payments: int = Field(ge=0)
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int = Field(ge=0, le=1)


class DatasetRowChurn(FeatureVectorChurn):
    """
    Training dataset row: client features with the churn target

    :churn: int - target, 1 if the client left and 0 if stayed

    Extra CSV columns, like a client id, are ignored. Types are not
    strict here, because pandas may read whole numbers as 14.0.
    """

    model_config = ConfigDict(
        extra="ignore",
        strict=False,
        json_schema_extra={"examples": [{**EXAMPLE_FEATURES, "churn": 0}]},
    )

    churn: int = Field(ge=0, le=1)
