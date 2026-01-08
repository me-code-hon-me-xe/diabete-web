import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# LOAD MODEL & SCALER
# =========================================================
MODEL_PATH = "diabetes_logreg_model_08012026.pkl"
SCALER_PATH = "diabetes_logreg_scaler_08012026.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = list(model.feature_names_in_)

st.set_page_config(page_title="Diabetes Screening Tool", layout="centered")

st.title("🩺 Diabetes Risk Screening Tool")
st.write("Enter patient data manually **or** upload an Excel/CSV file for batch prediction.")

# =========================================================
# OPTION 1: MANUAL INPUT
# =========================================================
st.header("Option 1: Manual Input")

input_data = {}

col1, col2 = st.columns(2)
with col1:
    input_data['Age'] = st.number_input("Age (years)", 1, 120, 40)
    input_data['Gender'] = st.selectbox("Gender", {"Male":1, "Female":0})
    input_data['BMI'] = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0)
    input_data['Waist_Circumference'] = st.number_input("Waist Circumference (cm)", 50.0, 160.0, 90.0)
    input_data['Systolic_BP'] = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
    input_data['Diastolic_BP'] = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)

with col2:
    input_data['Family_History_Diabetes'] = st.selectbox("Family History of Diabetes", {"No":0, "Yes":1})
    input_data['History_Hypertension'] = st.selectbox("History of Hypertension", {"No":0, "Yes":1})
    input_data['History_Dyslipidemia'] = st.selectbox("History of Dyslipidemia", {"No":0, "Yes":1})
    input_data['Physical_Activity'] = st.selectbox("Regular Physical Activity", {"No":0, "Yes":1})
    input_data['Education_Level'] = st.slider("Education Level (1–5)", 1, 5, 3)
    input_data['Race_Ethnicity'] = st.selectbox(
        "Race / Ethnicity",
        {"Mexican":1, "Other Hispanic":2, "White":3, "Black":4, "Asian":6, "Other":7}
    )

if st.button("🔍 Predict Risk (Manual)"):
    df_input = pd.DataFrame([input_data])[FEATURES]
    X_scaled = scaler.transform(df_input)
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("Result")
    st.metric("Diabetes Risk Probability", f"{prob*100:.2f}%")

    if prob >= 0.65:
        st.error("Very high risk – Immediate medical testing recommended")
    elif prob >= 0.30:
        st.warning("Moderate risk – Lifestyle intervention advised")
    else:
        st.success("Low risk – Maintain healthy lifestyle")

# =========================================================
# OPTION 2: EXCEL / CSV UPLOAD
# =========================================================
st.header("Option 2: Upload Excel / CSV")

uploaded_file = st.file_uploader("Upload file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
    else:
        df_upload = pd.read_excel(uploaded_file)

    st.write("Preview uploaded data:")
    st.dataframe(df_upload.head())

    if st.button("📊 Predict from File"):
        df_upload = df_upload[FEATURES]
        X_scaled = scaler.transform(df_upload)
        probs = model.predict_proba(X_scaled)[:,1]

        df_upload['Diabetes_Risk_Probability'] = probs
        df_upload['Risk_Level'] = np.where(
            probs >= 0.65, "High",
            np.where(probs >= 0.30, "Medium", "Low")
        )

        st.success("Prediction completed!")
        st.dataframe(df_upload)

        st.download_button(
            "⬇️ Download Results",
            df_upload.to_csv(index=False),
            file_name="diabetes_screening_results.csv",
            mime="text/csv"
        )
