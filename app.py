"""Streamlit app: evaluate pre-trained classifiers on the Spambase test data.

The models are loaded from model/*.joblib and are NOT retrained on the uploaded data.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "spam"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Spam Email Classifier Evaluation", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def compute_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:  # only one class present in the uploaded file
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }, pred, proba


st.title("Spam Email Classification - Model Evaluation")
st.write(
    "Upload the **test CSV** (all 57 feature columns plus the `spam` target column). "
    "The five pre-trained models are loaded from disk and evaluated on it; "
    "nothing is retrained."
)

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None)
    model_name = st.selectbox("Select model", list(MODEL_FILES))

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv(BASE_DIR / "test_data.csv")
else:
    st.info("Upload a CSV file or tick the bundled test data option in the sidebar.")
    st.stop()

model = load_model(model_name)
features = list(model.feature_names_in_)
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The CSV is missing required column(s): {missing}")
    st.stop()

X, y = df[features], df[TARGET]
st.subheader("Data preview")
st.write(f"{len(df)} rows, {len(features)} features. Class counts: "
         f"{y.value_counts().sort_index().to_dict()}")
st.dataframe(df.head(), width="stretch")

metrics, pred, proba = compute_metrics(model, X, y)

st.subheader(f"Evaluation metrics - {model_name}")
cols = st.columns(len(metrics))
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not spam", "Spam"], yticklabels=["Not spam", "Spam"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("ROC curve")
    if y.nunique() > 1:
        fpr, tpr, _ = roc_curve(y, proba)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("ROC curve needs both classes in the uploaded data.")

st.subheader("Classification report")
report = classification_report(y, pred, target_names=["Not spam", "Spam"],
                               output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

st.subheader("All models on this data")
rows = {name: compute_metrics(load_model(name), X, y)[0] for name in MODEL_FILES}
st.dataframe(pd.DataFrame(rows).T.round(4), width="stretch")
