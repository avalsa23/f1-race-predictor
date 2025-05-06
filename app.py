
import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("f1_model.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="F1 Race Prediction Bot", page_icon="🏁")
st.title("🏎️ F1 Race Position Predictor")
st.write("Select a driver and constructor to predict finishing position.")

# Sample options
drivers = ['Lewis Hamilton', 'Max Verstappen', 'Charles Leclerc', 'Lando Norris']
constructors = ['Mercedes', 'Red Bull', 'Ferrari', 'McLaren']

driver = st.selectbox("Driver", drivers)
constructor = st.selectbox("Constructor", constructors)
grid = st.slider("Grid Position", 1, 20, 5)
qualifying = st.slider("Qualifying Position", 1, 20, 5)

if st.button("Predict Finish Position"):
    df = pd.DataFrame({
        'driver_name': [driver],
        'constructor_name': [constructor],
        'grid_position': [grid],
        'qualifying_position': [qualifying]
    })
    df_encoded = encoder.transform(df)
    prediction = model.predict(df_encoded)[0]
    st.success(f"🏁 Predicted finishing position: **{int(prediction)}**")
