"""Reporting utilities for RetinaLens model evaluation."""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


def format_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Return a readable metrics dictionary."""
    return {
        "accuracy": float(result["accuracy"]),
        "precision": float(result["precision"]),
        "recall": float(result["recall"]),
        "f1": float(result["f1"]),
        "roc_auc": float(result["roc_auc"]),
    }


def plot_confusion_matrix(cm: Any, title: str = "Confusion matrix") -> None:
    """Plot a confusion matrix for a model."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: pd.Series, y_prob: pd.Series, title: str = "ROC curve") -> None:
    """Plot the ROC curve for model probability scores."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="navy", lw=2)
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title(title)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tight_layout()
    plt.show()


def summarize_model_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a concise model comparison table."""
    rows = []
    for key, result in results.items():
        feature_set, model_name = key.split(":", 1)
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
            }
        )
    return pd.DataFrame(rows)
