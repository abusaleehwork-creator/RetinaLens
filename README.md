# RetinaLens

RetinaLens is an educational machine-learning project for a binary diabetic-retinopathy classification task using the UCI Diabetic Retinopathy Debrecen dataset. The goal is to teach a practical end-to-end ML workflow: data loading, exploratory analysis, preprocessing, model comparison, evaluation, serialization, and a lightweight deployment-ready application.

## Project overview

This project intentionally does not claim to diagnose disease or provide clinical decision support. It is a student-focused learning project that demonstrates how a small ML product is built, documented, and evaluated with clear limitations.

## Problem statement

The task is to classify whether a retinal-image-derived feature vector shows signs of diabetic retinopathy (DR)
using a structured dataset. The model is trained on tabular features derived from retinal images, not raw fundus images.

## Why this project matters

Diabetic retinopathy is a major preventable cause of vision loss, but this project is educational only. It helps a learner understand how feature-based ML systems are trained, validated, and deployed while emphasizing the boundaries between research and clinical practice.

## Dataset

- Dataset name: Diabetic Retinopathy Debrecen Data Set
- Source: UCI Machine Learning Repository
- Dataset ID: 329
- Observations: 1151
- Features: 19
- Target: `Class`
- Target definition: `1 = contains signs of diabetic retinopathy`, `0 = no signs of diabetic retinopathy`

Important notes:
- The data is a tabular representation of image-derived retinal features, not raw images.
- Some features are close to the target or reflect prior classification decisions; therefore we compare both a full-feature setup and a leakage-aware restricted feature configuration.
- This project uses the dataset strictly for educational exploration and is not a clinical system.

## Methodology

1. Download the dataset programmatically from the UCI repository.
2. Inspect its schema, missing values, class balance, and potential leakage risks.
3. Split data into train and test sets with stratification.
4. Build a preprocessing pipeline with imputation and standardization.
5. Train logistic regression as a baseline and compare it with tree-based models.
6. Evaluate using accuracy, precision, recall, F1, ROC-AUC, and confusion matrices.
7. Save the best pipeline for prediction.
8. Build a small Streamlit app for educational inference.

## ML pipeline

The project follows a reusable ML pipeline:

- Data ingestion
- Feature normalization and leakage review
- Train/test splitting
- Preprocessing
- Model training
- Evaluation
- Model serialization
- Prediction through a lightweight app

## Models evaluated

- Logistic Regression
- Random Forest Classifier
- HistGradientBoostingClassifier

## Evaluation metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Results

The project compares the same evaluation procedure across models and stores the results in `reports/model_results.md`.

Baseline result (illustrative example after local training):
- Logistic Regression: accuracy ≈ 0.75, recall ≈ 0.61, ROC-AUC ≈ 0.83

These are sample project results and should be interpreted carefully. Model performance is not clinical validation and should not be used for patient care.

## Architecture and project structure

```text
RetinaLens/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── data/
│   ├── README.md
│   └── retina_data.csv
├── models/
│   └── retinalens_model.joblib
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── reports/
│   └── model_results.md
├── train_model.py
└── .venv/
```

## How to run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate    # Linux/macOS
# or .\.venv\Scripts\Activate.ps1   # Windows PowerShell
python -m pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## How to run tests

```bash
python -m pytest -q
```

## How to use the Streamlit app

- Open the app locally.
- Navigate to the Prediction page.
- Enter feature values or use the sample input.
- Click "Predict class".
- Read the output carefully: the app displays a model classification, not a diagnosis.

## Deployment

This project is set up to be deployable on Streamlit Community Cloud. The app uses relative paths and does not rely on local machine-specific paths.

For deployment:
1. Push the repo to GitHub.
2. Connect the repo to Streamlit Community Cloud.
3. Ensure `requirements.txt` and `app.py` are present.
4. Verify the model artifact is available in `models/`.

## Limitations

- The project is educational; it is not clinically validated.
- It uses a tabular benchmark dataset and not raw eye images.
- The model should not be used for patient care, diagnosis, or treatment decisions.
- Metrics are research-oriented and may vary depending on the chosen feature set and evaluation strategy.

## Ethical and medical disclaimer

This project is not a medical device and not a clinical decision-support system. It should not be used to diagnose or treat patients. It is a learning tool for understanding machine-learning workflows in health-data settings.

## Future improvements

- Add richer exploratory notebooks and error analysis.
- Compare feature sets more systematically for leakage-aware modeling.
- Add calibration plots and decision-threshold analysis.
- Improve the Streamlit interface with clearer explanations and sample data.

## Final learning flow

The project teaches how the full system connects:

DATA → PREPROCESSING → TRAINING → EVALUATION → MODEL SERIALIZATION → PREDICTION PIPELINE → STREAMLIT UI → GITHUB → DEPLOYMENT

## License

This project is licensed under the MIT License.
