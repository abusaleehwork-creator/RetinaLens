# Model comparison summary

The table below summarizes the project’s comparison of baseline and alternative classifiers on the retinal feature dataset.

| Model | Feature set | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | full | 0.753 | 0.893 | 0.610 | 0.725 | 0.828 |
| Random Forest | full | 0.706 | 0.767 | 0.642 | 0.699 | 0.788 |
| HistGradientBoosting | full | 0.732 | 0.775 | 0.699 | 0.735 | 0.778 |

## Interpretation

This is a binary classification task with a moderate class imbalance and a strong need to consider the trade-off between false negatives and false positives. Logistic regression offers a clear baseline with strong interpretability and competitive ROC-AUC, while tree-based models may improve recall but can be less transparent. The exact ranking depends on the chosen feature subset and threshold policy.

## Important note

This project is educational and not a medical device. It should not be used to diagnose or treat patients.
