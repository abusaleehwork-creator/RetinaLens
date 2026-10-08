"""Training and evaluation utilities for RetinaLens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.data import get_feature_sets, load_dataset
from src.preprocessing import build_model_pipeline, prepare_dataframe

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def split_features_target(df: pd.DataFrame, feature_set: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Create a train-ready X/y split from a feature subset."""
    X = df[feature_set].copy()
    y = df["target"].copy()
    return X, y


def get_model_specs() -> dict[str, Any]:
    """Return a dictionary of candidate models and their training settings."""
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42, max_depth=5, learning_rate=0.08),
    }


def train_model(model_name: str, X: pd.DataFrame, y: pd.Series, feature_set: str) -> dict[str, Any]:
    """Fit a model and return metadata about the training run."""
    model = get_model_specs()[model_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    pipeline = build_model_pipeline(X_train, model)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    result = {
        "model_name": model_name,
        "feature_set": feature_set,
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds),
    }
    return result


def cross_validate_model(model_name: str, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Estimate performance using 5-fold cross-validation on the training data."""
    model = get_model_specs()[model_name]
    pipeline = build_model_pipeline(X, model)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return {"mean_roc_auc": float(np.mean(scores)), "std_roc_auc": float(np.std(scores))}


def train_all_models() -> dict[str, dict[str, Any]]:
    """Train and compare the configured models on the full and restricted feature sets."""
    df = prepare_dataframe(load_dataset())
    results: dict[str, dict[str, Any]] = {}
    for feature_group_name, features in get_feature_sets().items():
        X, y = split_features_target(df[features + ["target"]], features)
        for model_name in get_model_specs():
            key = f"{feature_group_name}:{model_name}"
            results[key] = train_model(model_name, X, y, feature_group_name)
    return results


def save_model(pipeline: Any, name: str = "retinalens_model.joblib") -> Path:
    """Persist a trained pipeline object to disk."""
    path = MODEL_DIR / name
    joblib.dump(pipeline, path)
    return path
