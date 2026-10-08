"""Train and save the RetinaLens model pipeline."""

from __future__ import annotations

from src.data import fetch_dataset, save_raw_dataset
from src.train import save_model, train_all_models


def main() -> None:
    features, target = fetch_dataset()
    save_raw_dataset(features, target)
    results = train_all_models()
    best_key = sorted(
        results,
        key=lambda key: results[key]["roc_auc"],
        reverse=True,
    )[0]
    pipeline = results[best_key]["pipeline"]
    save_model(pipeline, name="retinalens_model.joblib")
    print(f"Best model by ROC-AUC: {best_key}")
    print(results[best_key])


if __name__ == "__main__":
    main()
