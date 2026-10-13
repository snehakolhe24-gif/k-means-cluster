import streamlit as st
import pandas as pd
import numpy as np


# Configure page style
st.set_page_config(
    page_title="Health Classification Predictor",
    page_icon="⚕ ",
    layout="centered"
)

st.title("⚕ Health Classification Predictor")
st.write("This interactive app uses a K-Nearest Neighbors model to classify health status based on clinical features.")

# Load artifacts safely
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler artifacts: {e}")
    st.stop()

# Input Forms grouped visually
st.header("✅ Input Clinical Measurements")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=48)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=23.5, step=0.1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=30.0, max_value=300.0, value=70.0, step=1.0)
    insulin = st.number_input("Insulin (IU/mL)", min_value=1.0, max_value=100.0, value=2.7, step=0.1)
    homa = st.number_input("HOMA", min_value=0.1, max_value=30.0, value=0.46, step=0.01)

with col2:
    leptin = st.number_input("Leptin (ng/mL)", min_value=1.0, max_value=120.0, value=8.8, step=0.1)
    adiponectin = st.number_input("Adiponectin (g/mL)", min_value=1.0, max_value=50.0, value=9.7, step=0.1)
    resistin = st.number_input("Resistin (ng/mL)", min_value=1.0, max_value=100.0, value=8.0, step=0.1)
    mcp_1 = st.number_input("MCP-1 (pg/mL)", min_value=10.0, max_value=2000.0, value=417.1, step=1.0)

# Feature alignment (strictly matching training features)
features_list = ['age', 'bmi', 'glucose', 'insulin', 'homa', 'leptin', 'adiponectin', 'resistin', 'mcp_1']

# Perform Prediction
if st.button("Run Diagnostic Classification", type="primary"):
    # Format inputs
    input_data = pd.DataFrame([[age, bmi, glucose, insulin, homa, leptin, adiponectin, resistin, mcp_1]], 
                              columns=features_list)
    
    # Scale inputs
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    st.subheader(" Results")
    if prediction == 1:
        st.success("✅ Class 1: Healthy Control")
        st.write(f"Confidence: {probabilities[0]*100:.1f}%")
    else:
        st.error("☑ Class 2: Patient Group")
        st.write(f"Confidence: {probabilities[1]*100:.1f}%")
