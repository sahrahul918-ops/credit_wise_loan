import streamlit as st
import pandas as pd
import joblib

# 1. SETUP AND LOAD ASSETS
# These must be loaded once to keep the app fast
@st.cache_resource
def load_assets():
    model = joblib.load('loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# 2. USER INTERFACE
st.set_page_config(page_title="Credit Wise", page_icon="🏦")
st.title("Credit Wise Loan Approval App 🏦")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income", value=50000)
    coapplicant_income = st.number_input("Co-Applicant Income", value=0)
    age = st.slider("Age", 18, 100, 30)
    credit_score = st.slider("Credit Score", 300, 850, 700)
    loan_amount = st.number_input("Loan Amount", value=10000)
    loan_term = st.number_input("Loan Term (months)", value=60)

with col2:
    marital = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    emp_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    emp_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Unemployed"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business"])

# 3. PREDICTION LOGIC
if st.button("Predict Loan Approval", type="primary"):
    # Create the raw dataframe with the 18 original features
    input_df = pd.DataFrame({
        'Applicant_Income': [applicant_income],
        'Coapplicant_Income': [coapplicant_income],
        'Age': [age],
        'Marital_Status': [1 if marital == 'Married' else 0],
        'Dependents': [0], # Default value
        'Credit_Score': [credit_score],
        'Existing_Loans': [1], # Default value
        'DTI_Ratio': [0.3], # Default value
        'Savings': [5000], # Default value
        'Collateral_Value': [20000], # Default value
        'Loan_Amount': [loan_amount],
        'Loan_Term': [loan_term],
        'Loan_Purpose': [loan_purpose],
        'Property_Area': [property_area],
        'Education_Level': [1 if education == 'Graduate' else 0],
        'Gender': [1 if gender == 'Male' else 0],
        'Employer_Category': [emp_category],
        'Employment_Status': [emp_status]
    })

    # Convert categories to dummy variables
    input_df = pd.get_dummies(input_df, columns=["Loan_Purpose", "Property_Area", "Employer_Category", "Employment_Status"], dtype=int)

    # REINDEX: This is the SOLUTION for the 27-column issue.
    # It adds missing columns with 0 and drops extra ones to match the model exactly.
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale and predict
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.success("🎉 Loan APPROVED!")
    else:
        st.error("⚠️ Loan REJECTED.")