%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# CONFIGURATION & LOAD MODEL
# =========================================================
st.set_page_config(page_title="Diabetes Screening Tool", layout="centered")

MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/Diabete/nhanes_data/model/diabetes_logreg_model_08012026.pkl"
SCALER_PATH = "/content/drive/MyDrive/Colab Notebooks/Diabete/nhanes_data/model/diabetes_logreg_scaler_08012026.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define features in the exact order the model expects (Vietnamese names mapped from English)
FEATURES_VN = [
    'Tuoi', 'Gioi_Tinh', 'Sac_Toc', 'Trinh_Do_Hoc_Van',
    'Chi_So_BMI', 'Vong_Eo', 'HA_Tam_Thu', 'HA_Tam_Truong', 'Di_Truyen_Gia_Dinh',
    'Van_Dong_The_Chat', 'Tien_Su_Cao_HA', 'Tien_Su_Mo_Mau'
]

st.title("🩺 Diabetes Risk Screening Tool")
st.write("Enter patient data below to predict Type 2 Diabetes risk.")

# =========================================================
# MANUAL INPUT FORM
# =========================================================
with st.form("prediction_form"):
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", 1, 120, 40)
        
        # FIX: Handle Gender Mapping
        gender_sb = st.selectbox("Gender", ["Male", "Female"])
        gender = 1 if gender_sb == "Male" else 0
        
        # FIX: Handle Race Mapping
        race_sb = st.selectbox("Race/Ethnicity", 
                               ["Mexican", "Other Hispanic", "White", "Black", "Asian", "Other"])
        race_map = {"Mexican": 1, "Other Hispanic": 2, "White": 3, "Black": 4, "Asian": 6, "Other": 7}
        race = race_map[race_sb]
        
        # FIX: Handle Education Mapping
        edu_sb = st.selectbox("Education", 
                              ["< 9th Grade", "9-11th Grade", "High School", "Some College", "College Grad"])
        edu_map = {"< 9th Grade": 1, "9-11th Grade": 2, "High School": 3, "Some College": 4, "College Grad": 5}
        education = edu_map[edu_sb]

    with col2:
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        
        # Auto-calculate BMI
        bmi_calc = weight / ((height/100)**2)
        bmi = st.number_input(f"BMI (Auto: {bmi_calc:.1f})", 10.0, 60.0, float(f"{bmi_calc:.1f}"))
        
        waist = st.number_input("Waist Circumference (cm)", 50.0, 200.0, 85.0)

    st.subheader("Medical History & Vitals")
    col3, col4 = st.columns(2)
    
    with col3:
        sys_bp = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
        dia_bp = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
        
        # FIX: Physical Activity
        active_sb = st.selectbox("Physical Activity (>30m/day)", ["Yes", "No"])
        activity = 1 if active_sb == "Yes" else 0

    with col4:
        # FIX: Yes/No Questions
        fam_sb = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        family = 1 if fam_sb == "Yes" else 0
        
        bp_sb = st.selectbox("History of Hypertension", ["No", "Yes"])
        hbp_hist = 1 if bp_sb == "Yes" else 0
        
        lip_sb = st.selectbox("History of High Cholesterol", ["No", "Yes"])
        lipid_hist = 1 if lip_sb == "Yes" else 0

    submit_btn = st.form_submit_button("🔍 PREDICT RISK")

# =========================================================
# PREDICTION LOGIC
# =========================================================
if submit_btn:
    # 1. Prepare Dataframe with Correct Vietnamese Column Names
    # Order must match FEATURES_VN list exactly
    input_data = pd.DataFrame([{
        'Tuoi': age,
        'Gioi_Tinh': gender,
        'Sac_Toc': race,
        'Trinh_Do_Hoc_Van': education,
        'Chi_So_BMI': bmi,
        'Vong_Eo': waist,
        'HA_Tam_Thu': sys_bp,
        'HA_Tam_Truong': dia_bp,
        'Di_Truyen_Gia_Dinh': family,
        'Van_Dong_The_Chat': activity,
        'Tien_Su_Cao_HA': hbp_hist,
        'Tien_Su_Mo_Mau': lipid_hist
    }])
    
    # 2. Scale Data
    try:
        X_scaled = scaler.transform(input_data)
        
        # 3. Predict
        prob = model.predict_proba(X_scaled)[0][1]
        percent = prob * 100
        
        # 4. Show Result
        st.divider()
        st.subheader("📊 Analysis Result")
        st.progress(prob)
        st.metric("Diabetes Risk Probability", f"{percent:.2f}%")
        
        if prob >= 0.65:
            st.error("⚠️ HIGH RISK: Immediate medical check-up (HbA1c) recommended.")
        elif prob >= 0.30:
            st.warning("🟡 MODERATE RISK: Lifestyle changes (Diet/Exercise) required.")
        else:
            st.success("✅ LOW RISK: Keep up the healthy lifestyle!")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug info: Check if input data types are numeric.")
        st.write(input_data)
