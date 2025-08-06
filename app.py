import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import plotly.graph_objects as go

# Load model and columns
model = joblib.load("models/xgb_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.set_page_config(page_title="Credit Risk Detector", layout="centered")
st.title("💳 Credit Risk Predictor")
st.markdown("Enter the applicant details below:")

# --- Input Form ---
with st.form("user_input_form"):
    CODE_GENDER = st.selectbox("Gender", ["M", "F"])
    FLAG_OWN_CAR = st.selectbox("Owns a Car?", ["Y", "N"])
    FLAG_OWN_REALTY = st.selectbox("Owns Realty?", ["Y", "N"])
    CNT_CHILDREN = st.number_input("Number of Children", min_value=0, value=0)
    AMT_INCOME_TOTAL = st.number_input("Annual Income", min_value=0.0, value=100000.0)

    NAME_INCOME_TYPE = st.selectbox("Employment Type", [
        "Working", "Commercial associate", "Pensioner", "State servant", "Unemployed",
        "Student", "Businessman", "Maternity leave", "Entrepreneur", "Serviceman",
        "Self-employed", "Freelancer", "Other"
    ])
    NAME_EDUCATION_TYPE = st.selectbox("Education", [
        "Higher education", "Secondary / secondary special", "Incomplete higher",
        "Lower secondary", "Academic degree"
    ])
    NAME_FAMILY_STATUS = st.selectbox("Family Status", [
        "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
    ])
    NAME_HOUSING_TYPE = st.selectbox("Housing Type", [
        "House / apartment", "With parents", "Municipal apartment", "Rented apartment",
        "Office apartment", "Co-op apartment"
    ])
    AGE_YEARS = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
    YEARS_EMPLOYED = st.number_input("Years Employed", min_value=0, max_value=60, value=5)
    DAYS_BIRTH = -AGE_YEARS * 365
    DAYS_EMPLOYED = -YEARS_EMPLOYED * 365

    OCCUPATION_TYPE = st.selectbox("Occupation", [
        "Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff",
        "Cleaning staff", "Cooking staff", "Security staff", "Waiters/barmen staff",
        "Medicine staff", "Private service staff", "Low-skill Laborers", "Realty agents",
        "Secretaries", "HR staff", "IT staff", "High skill tech staff", "Other"
    ])

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction ---
if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "CODE_GENDER": CODE_GENDER,
        "FLAG_OWN_CAR": FLAG_OWN_CAR,
        "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
        "CNT_CHILDREN": CNT_CHILDREN,
        "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
        "NAME_INCOME_TYPE": NAME_INCOME_TYPE,
        "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
        "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
        "NAME_HOUSING_TYPE": NAME_HOUSING_TYPE,
        "DAYS_BIRTH": DAYS_BIRTH,
        "DAYS_EMPLOYED": DAYS_EMPLOYED,
        "OCCUPATION_TYPE": OCCUPATION_TYPE
    }])

    # One-hot encode and align
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    # Display risk
    if prediction == 1:
        st.error(f"⚠️ High Credit Risk (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Credit Risk (Probability: {probability:.2f})")

    # Circular gauge
    st.subheader("📈 Risk Probability Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "red" if prediction == 1 else "green"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probability %"}
    ))
    st.plotly_chart(fig)

    # SHAP explainability
    st.subheader("🔍 SHAP Explainability")
    explainer = shap.Explainer(model)
    shap_values = explainer(input_encoded)

    # Generate and show SHAP plot
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())

    # PDF Report generation
    def generate_pdf_report(pred, prob):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Credit Risk Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Prediction: {'High Risk' if pred else 'Low Risk'}", ln=True)
        pdf.cell(200, 10, txt=f"Probability: {prob:.2f}", ln=True)
        pdf.ln(10)
        for col, val in input_data.iloc[0].items():
            pdf.cell(200, 10, txt=f"{col}: {val}", ln=True)

        buffer = BytesIO()
        pdf_output = pdf.output(dest='S').encode('latin1')
        buffer.write(pdf_output)
        buffer.seek(0)
        return buffer

    # Download button
    st.download_button(
        label="📄 Download PDF Report",
        data=generate_pdf_report(prediction, probability),
        file_name="credit_report.pdf",
        mime="application/pdf"
    )
