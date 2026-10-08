from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import get_feature_sets
from src.predict import predict_from_payload

st.set_page_config(page_title="RetinaLens", page_icon="👁️", layout="wide")

FEATURE_LABELS = {
    "quality": "Quality assessment",
    "pre_screening": "Pre-screening result",
    "ma1": "Microaneurysm count alpha 0.5",
    "ma2": "Microaneurysm count alpha 0.6",
    "ma3": "Microaneurysm count alpha 0.7",
    "ma4": "Microaneurysm count alpha 0.8",
    "ma5": "Microaneurysm count alpha 0.9",
    "ma6": "Microaneurysm count alpha 1.0",
    "exudate1": "Exudate score 1",
    "exudate2": "Exudate score 2",
    "exudate3": "Exudate score 3",
    "exudate4": "Exudate score 4",
    "exudate5": "Exudate score 5",
    "exudate6": "Exudate score 6",
    "exudate7": "Exudate score 7",
    "exudate8": "Exudate score 8",
    "macula_opticdisc_distance": "Macula-optic-disc distance",
    "opticdisc_diameter": "Optic-disc diameter",
    "am_fm_classification": "AM/FM classification output",
}


@st.cache_data
def get_example_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "quality": 1,
            "pre_screening": 1,
            "ma1": 22,
            "ma2": 22,
            "ma3": 22,
            "ma4": 19,
            "ma5": 18,
            "ma6": 14,
            "exudate1": 49.895756,
            "exudate2": 17.775994,
            "exudate3": 5.270920,
            "exudate4": 5.270920,
            "exudate5": 0.018632,
            "exudate6": 0.006864,
            "exudate7": 0.003923,
            "exudate8": 0.003923,
            "macula_opticdisc_distance": 0.486903,
            "opticdisc_diameter": 0.100025,
            "am_fm_classification": 1,
        }]
    )


st.title("RetinaLens")
st.caption("An educational machine-learning explorer for diabetic retinopathy classification")

st.markdown(
    """
    This project is for learning and educational exploration only. It does not diagnose disease, replace a clinician,
    or provide clinical decision support. The model is a research prototype built on a public benchmark dataset.
    """
)

pages = ["Home", "Prediction", "Model info", "About dataset"]
page = st.sidebar.radio("Navigate", pages)

if page == "Home":
    st.subheader("Project purpose")
    st.write(
        "RetinaLens demonstrates an end-to-end machine-learning workflow for a diabetic retinopathy classification task. "
        "The goal is educational understanding: data cleaning, feature engineering, model comparison, evaluation, and deployment."
    )
    st.info("The app intentionally avoids clinical claims and uses the model only as a research exercise.")
    st.subheader("Dataset")
    st.write("UCI Machine Learning Repository — Diabetic Retinopathy Debrecen dataset (dataset ID 329).")
    st.subheader("Model summary")
    st.write("The app currently uses a logistic-regression baseline with preprocessing and compares multiple classifiers for educational review.")

elif page == "Prediction":
    st.subheader("Enter retinal-feature values")
    st.caption("Use the sample input or fill in the values below.")
    example = get_example_inputs().iloc[0].to_dict()

    with st.form("prediction_form"):
        values = {}
        for key in list(example.keys()):
            values[key] = st.number_input(
                FEATURE_LABELS[key],
                value=float(example[key]),
                step=0.001 if key.startswith("exudate") or "distance" in key or "diameter" in key else 1.0,
                format="%.6f" if key.startswith("exudate") or "distance" in key or "diameter" in key else "%.0f",
            )
        submitted = st.form_submit_button("Predict class")

    if submitted:
        prediction, probability = predict_from_payload({k: v for k, v in values.items()})
        st.success(f"The model classified this input as Class {prediction}.")
        st.write(f"Probability of positive class: {probability:.3f}")
        st.warning("This is a machine-learning output, not a diagnosis or medical recommendation.")

elif page == "Model info":
    st.subheader("Selected model")
    st.write("Baseline model: Logistic Regression")
    st.write("Why it is useful: it is interpretable, fast, and provides a strong baseline for binary classification.")
    st.write("The project compares logistic regression, random forest, and histogram gradient boosting using the same evaluation procedure.")

elif page == "About dataset":
    st.subheader("About the dataset")
    st.write(
        "The dataset is hosted by the UCI Machine Learning Repository and consists of retinal-image-derived features. "
        "It includes numerical measurements associated with microaneurysm counts, exudate features, optic-disc metrics, and a binary target column."
    )
    st.write("Target definition: 1 indicates signs of diabetic retinopathy; 0 indicates no signs.")
    st.warning("The dataset is used for educational model exploration only and is not clinical validation.")

st.sidebar.write("Key limitation: this is not a medical diagnostic system.")
