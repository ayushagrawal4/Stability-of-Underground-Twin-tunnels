import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Twin Tunnel Collapse Multiplier Predictor")

st.markdown("Enter the parameters below. All values must be real positive numbers (decimals allowed).")
st.markdown("**Note:** For L2/L1, L1 is considered as 8 units.")

# Input fields for each parameter (no upper limit, min_value=0.0)
S_h = st.number_input("S/h", min_value=0.0, value=1.0, step=0.01, format="%.4f", help="Any positive real number (decimals allowed)")
H_h = st.number_input("H/h", min_value=0.0, value=1.0, step=0.01, format="%.4f", help="Any positive real number (decimals allowed)")
mi = st.number_input("mi", min_value=0.0, value=10.0, step=0.01, format="%.4f", help="Any positive real number (decimals allowed)")
GSI = st.number_input("GSI", min_value=0.0, value=50.0, step=0.01, format="%.4f", help="Any positive real number (decimals allowed)")
L2_L1 = st.number_input("L2/L1 (L1 is 8 units)", min_value=0.0, value=1.0, step=0.01, format="%.4f", help="Any positive real number (decimals allowed)")

if st.button("Predict Collapse Multiplier"):
    input_data = np.array([[H_h, S_h, GSI, mi, L2_L1]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Collapse Multiplier: {prediction[0]:,.2f}")
