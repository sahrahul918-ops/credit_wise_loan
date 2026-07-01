# --- 3. PREDICTION LOGIC ---
if st.button("Predict Loan Approval", type="primary"):
    
    # A. Create a dataframe with ALL 18 columns that you had in your original EDA
    raw_data = pd.DataFrame({
        'Applicant_Income': [income],
        'Coapplicant_Income': [co_income],
        'Age': [age],
        'Marital_Status': [1 if marital == 'Married' else 0],
        'Dependents': [dependents],
        'Credit_Score': [credit_score],
        'Existing_Loans': [existing_loans],
        'DTI_Ratio': [dti_ratio],
        'Savings': [savings],
        'Collateral_Value': [collateral],
        'Loan_Amount': [loan_amount],
        'Loan_Term': [loan_term],
        'Loan_Purpose': [loan_purpose],
        'Property_Area': [property_area],
        'Education_Level': [1 if education == 'Graduate' else 0],
        'Gender': [1 if gender == 'Male' else 0],
        'Employer_Category': [emp_category],
        'Employment_Status': [emp_status]
    })
    
    # B. Generate the dummy columns (This will expand to the 27 columns)
    processed_data = pd.get_dummies(raw_data, columns=["Loan_Purpose", "Property_Area", "Employer_Category", "Employment_Status"], dtype=int)
    
    # C. Reindex to ensure the columns are in the EXACT order the model expects
    processed_data = processed_data.reindex(columns=model_columns, fill_value=0)
    
    # D. Scale and Predict
    scaled_data = scaler.transform(processed_data)
    prediction = model.predict(scaled_data)
    
    # E. Output
    st.divider()
    if prediction[0] == 1:
        st.success("🎉 CONGRATULATIONS! The loan is APPROVED.")
    else:
        st.error("⚠️ We're sorry, the loan is REJECTED.")