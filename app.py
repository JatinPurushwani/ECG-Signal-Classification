
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="ECG Classification", layout="wide")

st.title("💓 ECG Signal Classification Dashboard")
st.markdown("""
This app predicts cardiac health condition based on ECG signal features.  
The model classifies inputs into:
- **0 → Normal**
- **1 → Mild Abnormalities**
- **2 → Severe Abnormalities**
""")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except Exception as e:
    st.error("Model files not found. Train and save rf_model.pkl and scaler.pkl first.")
    st.stop()

# File uploader
uploaded = st.file_uploader("Upload ECG data (CSV format)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Preprocess
    X = df.select_dtypes(include=["number"])
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    df["Predicted_Class"] = preds
    df["Prediction_Label"] = df["Predicted_Class"].map({0: "Normal", 1: "Mild", 2: "Severe"})

    st.subheader("🧠 Predictions")
    st.dataframe(df[["Predicted_Class", "Prediction_Label"]].head())

    st.subheader("📈 Class Distribution")
    st.bar_chart(df["Prediction_Label"].value_counts())

    st.download_button(
        label="Download Predictions as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ecg_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload a CSV file containing ECG features for prediction.")
