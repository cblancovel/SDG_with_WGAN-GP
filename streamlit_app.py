import streamlit as st
import pickle
import pandas as pd

# Load the saved model and scaler
with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit app layout
st.title("Adherence to Lifestyle Prediction App")
st.write("Predict adherence based on weekly activity data.")

# Input fields for the user
st.sidebar.header("Input Features")
active_minutes = st.sidebar.number_input("Total Active Minutes", min_value=0, value=500)
total_steps = st.sidebar.number_input("Total Steps", min_value=0, value=8500)
sedentary_minutes = st.sidebar.number_input("Sedentary Minutes", min_value=0, value=400)

# Create input data for prediction
input_data = pd.DataFrame({
    'TotalActiveMinutes': [active_minutes],
    'TotalSteps': [total_steps],
    'SedentaryMinutes': [sedentary_minutes]
})

# Scale the input data
scaled_input = scaler.transform(input_data)

# Make prediction
prediction = rf_model.predict(scaled_input)
probability = rf_model.predict_proba(scaled_input)[:, 1]

# Display results
st.subheader("Prediction Results")
st.write("Adherent" if prediction[0] == 1 else "Non-Adherent")
st.write(f"Probability of Adherence: {probability[0]:.2f}")
