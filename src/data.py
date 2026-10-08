"""Dataset loading and schema utilities for RetinaLens."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATASET_ID = 329
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "retina_data.csv"


def _normalize_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Normalize the dataset column names and resolve the duplicated exudate3 field."""
    df = features.copy()
    raw_columns = [str(col).strip().lower() for col in df.columns]
    normalized: list[str] = []
    seen_exudate3 = 0

    for col in raw_columns:
        if col == "exudate3":
            seen_exudate3 += 1
            if seen_exudate3 == 2:
                normalized.append("exudate4")
            else:
                normalized.append("exudate3")
        else:
            normalized.append(col)

    df.columns = normalized
    return df


def fetch_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch the UCI dataset and return feature and target data with normalized schema."""
    dataset = fetch_ucirepo(id=DATASET_ID)
    features = _normalize_feature_columns(dataset.data.features.copy())
    targets = dataset.data.targets.copy()
    targets.columns = [str(col).strip().lower() for col in targets.columns]
    if "class" not in targets.columns:
        raise ValueError("Expected a 'Class' target column in the fetched dataset.")
    return features, targets["class"]


def save_raw_dataset(features: pd.DataFrame, target: pd.Series, path: Path | str = DATA_PATH) -> Path:
    """Persist the dataset to disk in CSV format for reproducible local use."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = features.copy()
    combined["class"] = target.reset_index(drop=True)
    combined.to_csv(output, index=False)
    return output


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the stored dataset from disk and return a single DataFrame."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    return pd.read_csv(file_path)


def explain_features() -> list[str]:
    """Return the canonical feature names used by the project."""
    return [
        "quality",
        "pre_screening",
        "ma1",
        "ma2",
        "ma3",
        "ma4",
        "ma5",
        "ma6",
        "exudate1",
        "exudate2",
        "exudate3",
        "exudate4",
        "exudate5",
        "exudate6",
        "exudate7",
        "exudate8",
        "macula_opticdisc_distance",
        "opticdisc_diameter",
        "am_fm_classification",
    ]


def get_feature_sets() -> dict[str, list[str]]:
    """Return the full feature set and a leakage-aware restricted set for experiments."""
    full = explain_features()
    restricted = [
        "ma1",
        "ma2",
        "ma3",
        "ma4",
        "ma5",
        "ma6",
        "exudate1",
        "exudate2",
        "exudate3",
        "exudate4",
        "exudate5",
        "exudate6",
        "exudate7",
        "exudate8",
        "macula_opticdisc_distance",
        "opticdisc_diameter",
    ]
    return {"full": full, "restricted": restricted}
