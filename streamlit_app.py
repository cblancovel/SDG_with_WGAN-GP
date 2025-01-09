import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
with open("rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit App
st.title("Adherence to lifestyle prediction App")
st.write("Predict weekly adherence based on activity data.")

# Input form
st.sidebar.header("Input Data per week")
total_active_minutes = st.sidebar.number_input("Total Active Minutes", min_value=0, value=300)
total_steps = st.sidebar.number_input("Total Steps", min_value=0, value=8000)
sedentary_minutes = st.sidebar.number_input("Sedentary Minutes", min_value=0, value=400)
active_days = st.sidebar.number_input("Active Days (>=10 min activity)", min_value=0, max_value=7, value=5)

# Prepare input data
new_data = pd.DataFrame({
    'TotalActiveMinutes': [total_active_minutes],
    'TotalSteps': [total_steps],
    'SedentaryMinutes': [sedentary_minutes],
    'ActiveDays': [active_days]
})

# Scale input data
new_data_scaled = scaler.transform(new_data)

# Predict adherence
adherence_prediction = rf_model.predict(new_data_scaled)
adherence_proba = rf_model.predict_proba(new_data_scaled)[:, 1]

# Display results
st.subheader("Prediction Results")
if adherence_prediction[0] == 1:
    st.success("Adherent")
else:
    st.error("Non-Adherent")

st.write(f"Probability of Adherence: {adherence_proba[0]:.4f}")

