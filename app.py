import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="EV Battery Capacity Prediction",
    page_icon="🔋",
    layout="centered",
)


# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
@st.cache_resource
def load_files():
    model = load_model("models/battery_life_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


model, scaler = load_files()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🔋 EV Battery Capacity Prediction")

st.write("""
Predict the remaining battery capacity using battery voltage,
current and temperature characteristics.
""")

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    voltage_mean = st.number_input("Voltage Mean", value=3.70, format="%.2f")

    voltage_std = st.number_input("Voltage Std", value=0.05, format="%.2f")

    current_mean = st.number_input("Current Mean", value=1.50, format="%.2f")

    current_std = st.number_input("Current Std", value=0.20, format="%.2f")

    temperature_mean = st.number_input("Temperature Mean (°C)", value=30.0)

with col2:
    temperature_max = st.number_input("Temperature Max (°C)", value=35.0)

    current_load_mean = st.number_input("Current Load Mean", value=2.00, format="%.2f")

    voltage_load_mean = st.number_input("Voltage Load Mean", value=3.65, format="%.2f")

    time_max = st.number_input("Time Max", value=1000.0)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Battery Capacity", use_container_width=True):

    features = np.array(
        [
            [
                voltage_mean,
                voltage_std,
                current_mean,
                current_std,
                temperature_mean,
                temperature_max,
                current_load_mean,
                voltage_load_mean,
                time_max,
            ]
        ],
        dtype=np.float32,
    )

    features = scaler.transform(features)

    with st.spinner("Predicting battery capacity..."):
        prediction = float(model.predict(features, verbose=0)[0][0])

    prediction = max(0.0, min(prediction, 100.0))

    st.success("Prediction completed successfully!")

    st.metric("Predicted Battery Capacity", f"{prediction:.2f}%")

    st.progress(prediction / 100)

    if prediction >= 80:
        st.success("🟢 Battery Health: Excellent")

    elif prediction >= 60:
        st.info("🟡 Battery Health: Good")

    elif prediction >= 40:
        st.warning("🟠 Battery Health: Moderate")

    else:
        st.error("🔴 Battery Health: Poor")

st.divider()

with st.expander("Model Performance"):

    st.write("**Evaluation Metrics**")

    c1, c2 = st.columns(2)

    c1.metric("MAE", "0.0363")
    c1.metric("RMSE", "0.0494")

    c2.metric("MSE", "0.0024")
    c2.metric("R² Score", "0.9892")

st.caption("Built with TensorFlow, Scikit-Learn, NumPy and Streamlit.")
