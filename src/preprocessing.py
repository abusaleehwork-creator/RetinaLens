"""Preprocessing pipeline for RetinaLens."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def drop_duplicate_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the duplicate exudate3 column in the UCI dataset to keep a unique schema."""
    result = df.copy()
    duplicated = result.columns[result.columns.duplicated()].tolist()
    if "exudate3" in duplicated:
        result = result.rename(columns={"exudate3": "exudate3_raw"})
        result = result.rename(columns={"exudate3": "exudate4"})
    return result


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the raw data into a clean modeling DataFrame."""
    result = df.copy()
    result.columns = [str(col).strip().lower() for col in result.columns]
    result = result.rename(columns={"class": "target"})
    if "exudate3" in result.columns and "exudate4" not in result.columns:
        result = result.rename(columns={"exudate3": "exudate4"})
    if "exudate3_raw" in result.columns:
        result = result.drop(columns=["exudate3_raw"])
    if "target" not in result.columns:
        raise ValueError("Target column 'target' is required.")
    return result


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline that scales numeric features and fills missing values."""
    numeric_features = X.columns.tolist()
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(transformers=[("numeric", numeric_transformer, numeric_features)])


def build_model_pipeline(X: pd.DataFrame, model: Any) -> Pipeline:
    """Create a scikit-learn pipeline that combines preprocessing and a model."""
    return Pipeline(steps=[("preprocessor", build_preprocessor(X)), ("model", model)])
