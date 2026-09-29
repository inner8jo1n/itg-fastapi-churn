from itg_fastapi_churn.dataset.churn_dataset import ChurnDataset
from itg_fastapi_churn.schemas.churn import EXAMPLE_FEATURES


def test_preview_returns_first_rows(dataset: ChurnDataset) -> None:
    rows = dataset.preview(2)

    assert len(rows) == 2
    assert rows[1].churn == 1


def test_preview_returns_all_rows_when_n_exceeds_size(
    dataset: ChurnDataset,
) -> None:
    rows = dataset.preview(10)

    assert len(rows) == 3


def test_info_describes_dataset(dataset: ChurnDataset) -> None:
    info = dataset.info()

    assert info.n_rows == 3
    assert info.n_columns == 10
    assert info.feature_names == list(EXAMPLE_FEATURES)
    assert info.churn_distribution == {0: 2, 1: 1}
