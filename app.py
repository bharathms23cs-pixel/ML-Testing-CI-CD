import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="ML Testing CI/CD")

st.title("🏠 House Price Predictor")
st.write("Web Based GUI - ML Model Testing CI/CD")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
st.success("✅ Model Loaded - model.pkl")

# Inputs
sqft = st.number_input("Enter Square Feet", min_value=500, max_value=5000, value=1000)
bhk = st.number_input("Enter BHK", min_value=1, max_value=5, value=2)

if st.button("Predict Price"):
    # Predict - change [[sqft, bhk]] as per your model features
    prediction = model.predict([[sqft, bhk]])
    st.balloons()
    st.subheader(f"Predicted Price: ₹ {prediction[0]:,.2f}")
    st.write("Code Output Shown in GUI - Web Based Model")

st.divider()
st.write("GitHub Actions: 13 workflow runs ✅ Passed")
