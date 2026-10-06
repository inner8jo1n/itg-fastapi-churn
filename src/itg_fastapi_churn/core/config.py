from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """
    Application settings, overridable by CHURN_-prefixed env variables

    :dataset_path: Path - path to the training dataset CSV file
    :test_size: float - share of rows held out for the test split
    :random_state: int - seed that makes the train/test split reproducible
    :model_path: Path - file where the trained model is saved
    :history_path: Path - JSON Lines file with the training history
    :log_level: LogLevel - lowest level of service log messages; given
        in any letter case
    """

    model_config = SettingsConfigDict(env_prefix="CHURN_")

    dataset_path: Path = Path("data/churn_dataset.csv")
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = 42
    model_path: Path = Path("models/churn_model.joblib")
    history_path: Path = Path("models/training_history.jsonl")
    log_level: LogLevel = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        """
        Accept the log level in any letter case, like "info"

        :value: object - raw value from the environment or arguments

        :return: upper-cased text, other values unchanged
        """
        return value.upper() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    """
    Create the settings once and reuse them on later calls

    :return: cached application settings
    """
    return Settings()
