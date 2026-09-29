import pandas as pd
import pytest

from itg_fastapi_churn.ml.features import class_distribution
from itg_fastapi_churn.ml.split import DatasetSplit, split_dataset


def make_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame({"x": range(20)})
    target = pd.Series([0] * 16 + [1] * 4)
    return features, target


@pytest.fixture
def split() -> DatasetSplit:
    features, target = make_data()
    return split_dataset(features, target, test_size=0.25, random_state=0)


def test_split_dataset_uses_test_size(split: DatasetSplit) -> None:
    assert len(split.x_train) == 15
    assert len(split.x_test) == 5
    assert len(split.y_train) == 15
    assert len(split.y_test) == 5


def test_split_dataset_keeps_class_balance(split: DatasetSplit) -> None:
    assert class_distribution(split.y_train) == {0: 12, 1: 3}
    assert class_distribution(split.y_test) == {0: 4, 1: 1}


def test_split_dataset_is_reproducible() -> None:
    features, target = make_data()

    first = split_dataset(features, target, test_size=0.25, random_state=0)
    second = split_dataset(features, target, test_size=0.25, random_state=0)

    assert list(first.x_test.index) == list(second.x_test.index)


def test_split_info_reports_sizes_and_balance(split: DatasetSplit) -> None:
    info = split.info()

    assert info.train_rows == 15
    assert info.test_rows == 5
    assert info.train_churn_distribution == {0: 12, 1: 3}
    assert info.test_churn_distribution == {0: 4, 1: 1}
