import streamlit as st
import pandas as pd
import joblib

# --- PHASE 1: UI AND SETUP ---
st.set_page_config(page_title="Credit Wise", page_icon="🏦")
st.title("Credit Wise Loan Approval App 🏦")
st.write("Enter the applicant's details below to predict loan approval.")

# Load the trained model
# (Streamlit caches this so it doesn't reload the model every time a slider moves)
@st.cache_resource
def load_model():
    return joblib.load('loan_model.pkl')

model = load_model()

# --- PHASE 2: THE USER INTERFACE ---
st.header("Applicant Information")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=10000, step=500)

with col2:
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
    # For categorical data, use a selectbox (dropdown)
    employment_status = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])

# Convert categorical inputs to numbers if your model requires it (Example logic)
emp_status_encoded = 1 if employment_status in ["Employed", "Self-Employed"] else 0

# --- PHASE 3: THE PREDICTION LOGIC ---
if st.button("Predict Loan Approval", type="primary"):
    
    # 1. Format the inputs into the exact structure your model expects (a 2D array or DataFrame)
    # WARNING: The order of these variables MUST match the order of columns used during training!
    input_data = pd.DataFrame({
        'Income': [applicant_income],
        'LoanAmount': [loan_amount],
        'CreditScore': [credit_score],
        'Employed': [emp_status_encoded]
    })
    
    # 2. Make the prediction
    prediction = model.predict(input_data)
    
    # 3. Display the results!
    st.divider()
    if prediction[0] == 1: # Assuming 1 means Approved
        st.success("🎉 CONGRATULATIONS! The loan is APPROVED.")
        st.balloons() # Adds a fun animation
    else:
        st.error("⚠️ We're sorry, but the loan is REJECTED based on the current criteria.")