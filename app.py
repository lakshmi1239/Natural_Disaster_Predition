import streamlit as st
import numpy as np
from joblib import load

# ---------------------------
# Load the trained model
# ---------------------------
@st.cache_resource
def load_model(path):
    return load(path)

flood_model = load_model("models/flood_model.pkl")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🌊 Natural Disaster Prediction", page_icon="🌊", layout="centered")

st.title("🌊 Natural Disaster Prediction App")
st.write("This app predicts the risk of **flood occurrence** based on input data.")

# Example input fields (you can adjust according to your dataset features)
rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, step=0.1)
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, step=0.1)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
river_level = st.number_input("🌊 River Water Level (m)", min_value=0.0, step=0.1)

# Predict button
if st.button("🔍 Predict Flood Risk"):
    try:
        # Create input array
        input_data = np.array([[rainfall, temperature, humidity, river_level]])

        # Make prediction
        prediction = flood_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Flood!")
        else:
            st.success("✅ Low Risk of Flood.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
