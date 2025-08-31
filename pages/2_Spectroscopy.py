

import streamlit as st
import tempfile, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import sparse
from scipy.sparse.linalg import spsolve
from src.core.ai_insights import generate_gemini_insight  

st.title("Raman Spectroscopy")


MODEL_DIR = "src/models/spectroscopy/weights"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
WAVE_PATH = os.path.join(MODEL_DIR, "wave.npy") 


scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)
wave = np.load(WAVE_PATH) if os.path.exists(WAVE_PATH) else None


def baseline_correction(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = D.dot(D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z


def predict_raman(file_path, model_type='rf'):
    df = pd.read_csv(file_path, sep=r"\s+", engine="python", skiprows=1, names=["wave", "intensity"])
    intensity = np.interp(wave, df["wave"].values, df["intensity"].values) if wave is not None else df["intensity"].values
    intensity_corr = baseline_correction(intensity)
    intensity_scaled = scaler.transform([intensity_corr])
    intensity_pca = pca.transform(intensity_scaled)

    model = rf_model if model_type == 'rf' else xgb_model
    pred_class = model.predict(intensity_pca)[0]


    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(intensity_pca)[0]
        confidence = proba[pred_class]
    else:
        confidence = 1.0
        proba = [1.0 if pred_class==0 else 0.0, 1.0 if pred_class==1 else 0.0]

    label = "Healthy" if pred_class == 0 else "Cancer"
    return label, confidence, proba


model_choice = st.sidebar.selectbox("Select model", ["Random Forest", "XGBoost"])
uploaded = st.file_uploader("Upload spectrum (.txt/.csv, wavenumber,intensity)", type=["txt", "csv"])

if uploaded is None:
    st.info("Upload a Raman spectrum to classify.")
    st.stop()

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1] or ".txt")
tmp.write(uploaded.read())
tmp.flush()


model_type = 'rf' if model_choice == "Random Forest" else 'xgb'
label, confidence, proba = predict_raman(tmp.name, model_type=model_type)

st.success(f"Predicted label: **{label}** (confidence: {confidence:.2f})")


try:
    insight_text = generate_gemini_insight(label, confidence)
    st.info(insight_text)
except Exception as e:
    st.warning(f"Failed to generate Gemini AI insight: {e}")


try:
    df = pd.read_csv(tmp.name, sep=r"\s+", engine="python", skiprows=1, names=["wave", "intensity"])
    intensity_corr = baseline_correction(df["intensity"].values)
    plt.figure(figsize=(10,4))
    plt.plot(df["wave"], df["intensity"], label="Raw")
    plt.plot(df["wave"], intensity_corr, label="Baseline-corrected")
    plt.xlabel("Wavenumber")
    plt.ylabel("Intensity")
    plt.title(f"Raman Spectrum ({label}, confidence={confidence:.2f})")
    plt.legend()
    st.pyplot(plt)
except Exception as e:
    st.warning(f"Failed to plot spectrum: {e}")


if hasattr(proba, "__len__"):
    st.write("Class probabilities:")
    st.write(f"- Healthy: {proba[0]:.3f}")
    st.write(f"- Cancer: {proba[1]:.3f}")
