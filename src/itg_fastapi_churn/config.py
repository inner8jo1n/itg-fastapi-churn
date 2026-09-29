from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings, overridable by CHURN_-prefixed env variables

    :dataset_path: Path - path to the training dataset CSV file
    """

    model_config = SettingsConfigDict(env_prefix="CHURN_")

    dataset_path: Path = Path("data/churn_dataset.csv")


@lru_cache
def get_settings() -> Settings:
    """
    Create the settings once and reuse them on later calls

    :return: cached application settings
    """
    return Settings()
