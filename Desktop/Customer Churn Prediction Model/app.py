import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Customer Churn Prediction App")

# Replace this with your actual column names
feature_names = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

inputs = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

if st.button("Predict"):

    input_df = pd.DataFrame([inputs], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.write("Churn Probability:", round(probability, 2))
