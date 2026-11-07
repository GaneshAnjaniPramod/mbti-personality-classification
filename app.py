"""
MBTI Personality Classifier - Streamlit app (ML Assignment 2)

Loads the five pre-trained models from model/, evaluates them on an uploaded
test CSV (no retraining), and shows metrics, confusion matrix and
classification report.
"""
import hashlib
import io
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_CSV = os.path.join(BASE_DIR, "test_data.csv")

st.set_page_config(page_title="MBTI Personality Classifier", page_icon="🧭", layout="wide")


# ---------- loading helpers ----------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "model_info.json")) as f:
        info = json.load(f)
    models = {name: joblib.load(os.path.join(MODEL_DIR, fname))
              for name, fname in info["model_files"].items()}
    return info, models


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("The file could not be decoded. Save it as UTF-8 CSV and upload again.")


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    present = set(y.unique())
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": np.nan,
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    # One-vs-rest AUC needs at least 2 classes in the uploaded file
    if len(present) > 1:
        try:
            metrics["AUC"] = roc_auc_score(y, y_prob, multi_class="ovr",
                                           average="macro", labels=model.classes_)
        except ValueError:
            pass  # happens if the file does not contain every class
    return metrics, y_pred


@st.cache_data(show_spinner=False)
def evaluate_all(file_hash: str, _X: pd.DataFrame, _y: pd.Series):
    """Cached per uploaded file so switching models does not recompute everything."""
    _, models = load_artifacts()
    out = {}
    for name, model in models.items():
        out[name] = compute_metrics(model, _X, _y)
    return out


def letter_accuracy(y_true, y_pred):
    """Accuracy on each MBTI axis separately (E/I, S/N, T/F, J/P)."""
    yt = pd.Series(y_true).astype(str).str[:4]
    yp = pd.Series(y_pred).astype(str).str[:4]
    if not (yt.str.len() == 4).all():
        return None
    axes = ["Energy (E/I)", "Mind (S/N)", "Nature (T/F)", "Tactics (J/P)"]
    return pd.DataFrame({
        "Axis": axes,
        "Accuracy": [(yt.str[i] == yp.str[i]).mean() for i in range(4)],
    })


# ---------- header ----------
st.title("🧭 MBTI Personality Classifier")
st.write(
    "Predicts one of the 16 personality types from answers to 60 questionnaire items "
    "(each scored −3 to +3). Upload the held-out test set, choose a model, and see how it performs. "
    "Models were trained offline; nothing is retrained here."
)

try:
    info, models = load_artifacts()
except FileNotFoundError:
    st.error("Model files are missing. Make sure the model/ folder with the .pkl files and "
             "model_info.json is committed to the repository.")
    st.stop()

TARGET = info["target_col"]
FEATURES = info["feature_cols"]

# ---------- sidebar ----------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help="Must contain the 60 question columns and the "
                                     f"'{TARGET}' column.")
    if os.path.exists(SAMPLE_CSV):
        with open(SAMPLE_CSV, "rb") as f:
            st.download_button("Download sample test_data.csv", f.read(),
                               file_name="test_data.csv", mime="text/csv")
        use_sample = st.checkbox("Use bundled test_data.csv instead", value=False)
    else:
        use_sample = False

    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))

    st.divider()
    st.caption(f"Trained on {info['n_train']:,} rows · tested on {info['n_test']:,} rows · "
               f"scikit-learn {info['sklearn_version']}")

# ---------- get data ----------
raw = None
if uploaded is not None:
    raw = uploaded.getvalue()
    source = uploaded.name
elif use_sample:
    with open(SAMPLE_CSV, "rb") as f:
        raw = f.read()
    source = "test_data.csv (bundled)"

if raw is None:
    st.info("Upload a test CSV in the sidebar to start, or tick **Use bundled test_data.csv**. "
            "You can download the sample file from the sidebar first.")
    st.stop()

try:
    df = read_csv_bytes(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

missing = [c for c in FEATURES if c not in df.columns]
if TARGET not in df.columns:
    st.error(f"The file has no '{TARGET}' column. The test CSV must include the true labels.")
    st.stop()
if missing:
    st.error(f"{len(missing)} required feature column(s) are missing, e.g. “{missing[0]}”. "
             "Upload the test_data.csv produced by the training notebook.")
    st.stop()

df = df.dropna(subset=FEATURES + [TARGET])
X = df[FEATURES]
y = df[TARGET].astype(str)
unknown = sorted(set(y) - set(info["classes"]))
if unknown:
    st.warning(f"Labels not seen in training will always be counted as errors: {unknown}")

file_hash = hashlib.md5(raw).hexdigest()
with st.spinner("Scoring all five models on your file..."):
    all_results = evaluate_all(file_hash, X, y)

st.success(f"Loaded **{source}**: {len(df):,} rows, {y.nunique()} personality types.")

tab_model, tab_compare, tab_data = st.tabs(
    ["Selected model", "Compare all models", "Data preview"])

# ---------- tab 1: selected model ----------
with tab_model:
    metrics, y_pred = all_results[model_name]
    st.subheader(model_name)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")
    st.caption("Precision, Recall and F1 are macro-averaged over the 16 classes; "
               "AUC is one-vs-rest macro AUC.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Confusion matrix**")
        labels = sorted(set(y) | set(y_pred))
        cm = confusion_matrix(y, y_pred, labels=labels)
        normalize = st.toggle("Show as row percentages", value=False)
        fig, ax = plt.subplots(figsize=(9, 7.5))
        if normalize:
            cm_show = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            sns.heatmap(cm_show, annot=True, fmt=".0%", cmap="Purples", cbar=False,
                        xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 7})
        else:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                        xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 7})
        ax.set_xlabel("Predicted type")
        ax.set_ylabel("Actual type")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).T
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                          "f1-score": "{:.3f}"}),
                     height=560)

    la = letter_accuracy(y, y_pred)
    if la is not None:
        st.markdown("**Accuracy on each personality axis**")
        st.caption("A wrong type is often wrong on just one letter. This shows which axis the "
                   "model finds hardest.")
        st.dataframe(la.style.format({"Accuracy": "{:.2%}"}), hide_index=True)

    pred_df = df[[TARGET]].copy()
    pred_df["Predicted"] = y_pred
    pred_df["Correct"] = pred_df[TARGET] == pred_df["Predicted"]
    st.download_button("Download predictions (CSV)", pred_df.to_csv(index=False),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                       mime="text/csv")

# ---------- tab 2: comparison ----------
with tab_compare:
    st.subheader("All five models on this file")
    comp = pd.DataFrame({name: m for name, (m, _) in all_results.items()}).T
    comp.index.name = "ML Model Name"
    best = comp["MCC"].idxmax()
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#DCD5EE"))
    st.write(f"Best model by MCC on this file: **{best}**")

    metric_pick = st.radio("Metric to plot", list(comp.columns), horizontal=True, index=5)
    fig2, ax2 = plt.subplots(figsize=(9, 3.6))
    vals = comp[metric_pick].sort_values()
    colors = ["#5B4B8A" if n == vals.idxmax() else "#B9B0D6" for n in vals.index]
    ax2.barh(vals.index, vals.values, color=colors)
    for i, v in enumerate(vals.values):
        ax2.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax2.set_xlim(0, max(1.0, vals.max() * 1.1))
    ax2.set_xlabel(metric_pick)
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)
    plt.close(fig2)

# ---------- tab 3: data ----------
with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(df.head(50))
    st.markdown("**Class distribution in this file**")
    st.bar_chart(y.value_counts().sort_index())
