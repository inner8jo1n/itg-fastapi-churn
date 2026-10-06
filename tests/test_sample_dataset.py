from pathlib import Path

import pandas as pd
import pytest

from itg_fastapi_churn.dataset.loader import load_dataset
from itg_fastapi_churn.ml.features import CATEGORICAL_FEATURES, TARGET_COLUMN

FULL_DATASET_PATH = Path(__file__).parents[1] / "data" / "churn_dataset.csv"


@pytest.fixture
def full_dataset() -> pd.DataFrame:
    return load_dataset(FULL_DATASET_PATH)


def test_sample_is_a_valid_dataset(sample_dataset_path: Path) -> None:
    sample = load_dataset(sample_dataset_path)

    assert len(sample) == 80


def test_sample_has_columns_of_full_dataset(sample_dataset_path: Path) -> None:
    sample_columns = pd.read_csv(sample_dataset_path, nrows=0).columns
    full_columns = pd.read_csv(FULL_DATASET_PATH, nrows=0).columns

    assert list(sample_columns) == list(full_columns)


def test_sample_keeps_churn_share(
    sample_dataset_path: Path, full_dataset: pd.DataFrame
) -> None:
    sample = load_dataset(sample_dataset_path)

    assert sample[TARGET_COLUMN].mean() == pytest.approx(
        full_dataset[TARGET_COLUMN].mean(), abs=0.01
    )


@pytest.mark.parametrize("column", CATEGORICAL_FEATURES)
def test_sample_has_every_category_of_full_dataset(
    sample_dataset_path: Path, full_dataset: pd.DataFrame, column: str
) -> None:
    sample = load_dataset(sample_dataset_path)

    assert set(sample[column]) == set(full_dataset[column])
