"""Prediction pipeline utilities for deployed inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data import explain_features
from src.preprocessing import prepare_dataframe

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "retinalens_model.joblib"


def validate_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Validate feature values and return a DataFrame for prediction."""
    expected = explain_features()
    missing = [col for col in expected if col not in payload]
    if missing:
        raise ValueError(f"Missing feature values: {missing}")
    data = {col: float(payload[col]) for col in expected}
    df = pd.DataFrame([data])
    return df


def load_saved_pipeline(path: Path | str = DEFAULT_MODEL_PATH) -> Any:
    """Load a serialized training pipeline from disk."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def predict_from_payload(payload: dict[str, Any], model: Any | None = None) -> tuple[int, float]:
    """Produce a class prediction and probability score from a raw feature dict."""
    feature_frame = validate_input(payload)
    if model is None:
        model = load_saved_pipeline()
    prediction = model.predict(feature_frame)[0]
    probability = model.predict_proba(feature_frame)[0, 1]
    return int(prediction), float(probability)
