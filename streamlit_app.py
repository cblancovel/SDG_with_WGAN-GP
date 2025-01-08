import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# App title
st.title("Adherence to Lifestyle Prediction")

# Input form
st.header("Enter weekly data")
total_active_minutes = st.number_input("Total Active Minutes", min_value=0)
total_steps = st.number_input("Total Steps", min_value=0)
sedentary_minutes = st.number_input("Sedentary Minutes", min_value=0)

# Predict button
if st.button("Predict Adherence"):
    # Prepare the input data
    input_data = np.array([[total_active_minutes, total_steps, sedentary_minutes]])
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    adherence_prediction = rf_model.predict(input_data_scaled)
    adherence_proba = rf_model.predict_proba(input_data_scaled)[:, 1]

    # Display the result
    if adherence_prediction[0] == 1:
        st.success(f"Prediction: Adherent (Probability: {adherence_proba[0]:.2f})")
    else:
        st.error(f"Prediction: Non-Adherent (Probability: {adherence_proba[0]:.2f})")
