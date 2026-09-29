from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings, overridable by CHURN_-prefixed env variables

    :dataset_path: Path - path to the training dataset CSV file
    :test_size: float - share of rows held out for the test split
    :random_state: int - seed that makes the train/test split reproducible
    """

    model_config = SettingsConfigDict(env_prefix="CHURN_")

    dataset_path: Path = Path("data/churn_dataset.csv")
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = 42


@lru_cache
def get_settings() -> Settings:
    """
    Create the settings once and reuse them on later calls

    :return: cached application settings
    """
    return Settings()
