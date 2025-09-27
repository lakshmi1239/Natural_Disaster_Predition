import streamlit as st
import numpy as np
from joblib import load

# ---------------------------
# Page Config (must be first Streamlit command)
# ---------------------------
st.set_page_config(page_title="🌊 Flood Prediction", page_icon="🌊", layout="centered")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model(path):
    return load(path)

flood_model = load_model("models/flood_model.pkl")

# ---------------------------
# UI
# ---------------------------
st.title("🌊 Flood Prediction App")

st.write("Enter environmental data to predict flood risk:")

rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=0.1)
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, step=0.1)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
river_level = st.number_input("🌊 River Water Level (m)", min_value=0.0, step=0.1)

if st.button("🔍 Predict"):
    input_data = np.array([[rainfall, temperature, humidity, river_level]])
    prediction = flood_model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Flood!")
    else:
        st.success("✅ Low Risk of Flood.")
