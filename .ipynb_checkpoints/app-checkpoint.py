import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("bike_model.pkl", "rb"))

st.title("🚲 Bike Rental Demand Prediction")

season = st.selectbox("Season (1-4)", [1,2,3,4])
holiday = st.selectbox("Holiday (0=No, 1=Yes)", [0,1])
workingday = st.selectbox("Working Day (0=No, 1=Yes)", [0,1])
weather = st.selectbox("Weather (1-4)", [1,2,3,4])
temp = st.number_input("Temperature")
humidity = st.number_input("Humidity")
windspeed = st.number_input("Wind Speed")
hour = st.slider("Hour", 0, 23)
month = st.slider("Month", 1, 12)

# Arrange features in correct order
features = np.array([[season, holiday, workingday, weather,
                      temp, humidity, windspeed, hour, month, casual, registered]])

prediction_log = model.predict(features)

# Convert back from log
prediction = np.exp(prediction_log)

st.subheader("Predicted Bike Rentals:")
st.write(int(prediction[0]))