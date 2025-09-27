import streamlit as st
import numpy as np
from joblib import load





import streamlit as st
import pandas as pd
import numpy as np
import pickle
from geopy.geocoders import Nominatim
from sklearn.preprocessing import StandardScaler

def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

try:
    flood_model = load_model('models/flood_model.pkl')
    landscape_model = load_model('models/landscape_model.pkl')
    storm_model = load_model('models/storm_model.pkl')
    scaler = load_model('models/scaler.pkl')
except FileNotFoundError as e:
    st.error(f"Model file missing: {str(e)}")
    st.stop()

def predict_flood(rainwater_level, soil_moisture, dam_releases, temperature, humidity):
    features = np.array([[rainwater_level, soil_moisture, dam_releases, temperature, humidity]])
    scaled = scaler.transform(features)
    return flood_model.predict(scaled)[0]

def predict_landscape(elevation, rainfall, temperature, vegetation_index, soil_moisture):
    features = np.array([[elevation, rainfall, temperature, vegetation_index, soil_moisture]])
    scaled = scaler.transform(features)
    return landscape_model.predict(scaled)[0]

def predict_storm(temperature, humidity, pressure, wind_speed, wind_direction):
    features = np.array([[temperature, humidity, pressure, wind_speed, wind_direction]])
    scaled = scaler.transform(features)
    return storm_model.predict(scaled)[0]

def get_location_name(latitude, longitude):
    geolocator = Nominatim(user_agent="prediction_app")
    try:
        location = geolocator.reverse((latitude, longitude), language='en')
        return location.address if location else "Location not found"
    except Exception as e:
        return f"Error: {str(e)}"

st.title("Environmental Prediction System")
st.write("This system predicts Flood Risk, Landscape Type, or Storm based on environmental data.")

prediction_type = st.selectbox("Select Prediction Type", ["Flood Prediction", "Landscape Prediction", "Storm Prediction"])

if prediction_type == "Flood Prediction":
    st.set_page_config(page_title="🌊 Flood Prediction", page_icon="🌊", layout="centered")


@st.cache_resource
def load_model(path):
    return load(path)

flood_model = load_model("models/flood_model.pkl")


st.title("🌊 Flood Prediction App")
st.write("This app predicts the risk of flood occurrence.")


st.info(f"ℹ️ Model expects **{flood_model.n_features_in_}** features.")


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

elif prediction_type == "Landscape Prediction":
    st.subheader("Landscape Type Prediction")
    elevation = st.number_input("Elevation (m)", min_value=0.0, max_value=3000.0)
    rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=3000.0)
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0)
    vegetation_index = st.number_input("Vegetation Index (NDVI)", min_value=0.0, max_value=1.0)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0)
    if st.button("Predict Landscape Type"):
        result = predict_landscape(elevation, rainfall, temperature, vegetation_index, soil_moisture)
        st.success(f"Predicted Landscape Type: {result}")

elif prediction_type == "Storm Prediction":
    st.subheader("Storm Prediction")
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100)
    pressure = st.number_input("Atmospheric Pressure (hPa)", min_value=900, max_value=1100)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0, max_value=150)
    wind_direction = st.number_input("Wind Direction (°)", min_value=0, max_value=360)
    # clouds = st.number_input("Cloud Cover (%)", min_value=0, max_value=100)
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.704060)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.102493)
    if st.button("Predict Storm"):
        result = predict_storm(temperature, humidity, pressure, wind_speed, wind_direction)
        location_name = get_location_name(latitude, longitude)
        if result == 1:
            st.error("Warning: Storm Predicted!")
        else:
            st.success("No Storm Predicted.")
        st.write(f"Location: {location_name} (Lat: {latitude}, Lon: {longitude})")
        st.map(pd.DataFrame({'latitude': [latitude], 'longitude': [longitude]}))