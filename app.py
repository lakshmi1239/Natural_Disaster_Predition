import streamlit as st
import numpy as np
from joblib import load

# ---------------------------
# Page Config
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
st.write("This app predicts the risk of flood occurrence.")

# Debug: Show how many features model expects
st.info(f"ℹ️ Model expects **{flood_model.n_features_in_}** features.")

# Example: let’s create input fields dynamically
inputs = []
for i in range(flood_model.n_features_in_):
    value = st.number_input(f"Feature {i+1}", value=0.0, step=0.1)
    inputs.append(value)

if st.button("🔍 Predict"):
    input_data = np.array([inputs])  # 2D array
    prediction = flood_model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Flood!")
    else:
        st.success("✅ Low Risk of Flood.")
