from __future__ import annotations

import pandas as pd

from src.data import load_dataset
from src.preprocessing import build_preprocessor, prepare_dataframe


def test_prepare_dataframe_renames_target() -> None:
    df = load_dataset()
    prepared = prepare_dataframe(df)
    assert "target" in prepared.columns
    assert "class" not in prepared.columns


def test_preprocessor_handles_numeric_data() -> None:
    df = load_dataset().head(10)
    prepared = prepare_dataframe(df)
    X = prepared.drop(columns=["target"])
    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == X.shape[0]
    assert transformed.shape[1] == X.shape[1]
