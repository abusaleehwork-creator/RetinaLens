from __future__ import annotations

from src.data import DATA_PATH, fetch_dataset, load_dataset


def test_fetch_dataset_shape() -> None:
    features, target = fetch_dataset()
    assert features.shape[0] == target.shape[0]
    assert features.shape[1] == 19


def test_dataset_file_loads() -> None:
    df = load_dataset(DATA_PATH)
    assert "class" in df.columns
    assert df.shape[0] > 0
