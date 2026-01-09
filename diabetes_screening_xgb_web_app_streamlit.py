import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import xgboost as xgb # Required to load the XGBoost model

# =========================================================
# 1. SETUP & LOAD MODEL
# =========================================================
st.set_page_config(page_title="Diabetes Screening Tool", layout="centered", page_icon="🩺")

# Update this filename if you renamed it manually after downloading
MODEL_PATH = "diabetes_xgb_model_08012026.pkl"

@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File not found: {MODEL_PATH}")
        return None, None
    
    # Load the XGBoost model
    model = joblib.load(MODEL_PATH)
    
    # Get feature names directly from the XGBoost model
    features = list(model.feature_names_in_) 
    return model, features

model, FEATURES = load_prediction_model()

if model is None:
    st.stop()

# Helper function to export Excel
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

st.title("🩺 Diabetes Risk Screening Tool (XGBoost)")
st.write("Enter patient data manually **or** upload an Excel/CSV file.")

# =========================================================
# 2. OPTION 1: MANUAL INPUT
# =========================================================
st.header("Option 1: Manual Input")

with st.form("manual_form"):
    data = {}
    col1, col2 = st.columns(2)
    
    with col1:
        data['Age'] = st.number_input("Age (years)", 1, 120, 40)
        g_display = st.selectbox("Gender", ["Male", "Female"])
        data['Gender'] = 1 if g_display == "Male" else 0
        data['BMI'] = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0)
        data['Waist_Circumference'] = st.number_input("Waist Circumference (cm)", 50.0, 160.0, 90.0)
        data['Systolic_BP'] = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
        data['Diastolic_BP'] = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)

    with col2:
        fam_disp = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        data['Family_History_Diabetes'] = 1 if fam_disp == "Yes" else 0
        bp_hist_disp = st.selectbox("History of Hypertension", ["No", "Yes"])
        data['History_Hypertension'] = 1 if bp_hist_disp == "Yes" else 0
        chol_hist_disp = st.selectbox("History of Dyslipidemia", ["No", "Yes"])
        data['History_Dyslipidemia'] = 1 if chol_hist_disp == "Yes" else 0
        act_disp = st.selectbox("Regular Physical Activity", ["No", "Yes"])
        data['Physical_Activity'] = 1 if act_disp == "Yes" else 0
        
        edu_disp = st.selectbox("Education Level", 
            ["< 9th Grade", "9-11th Grade", "High School", "Some College", "College Grad"])
        edu_map = {"< 9th Grade":1, "9-11th Grade":2, "High School":3, "Some College":4, "College Grad":5}
        data['Education_Level'] = edu_map[edu_disp]
        
        race_disp = st.selectbox("Race / Ethnicity", 
            ["Mexican", "Other Hispanic", "White", "Black", "Asian", "Other"])
        race_map = {"Mexican":1, "Other Hispanic":2, "White":3, "Black":4, "Asian":6, "Other":7}
        data['Race_Ethnicity'] = race_map[race_disp]

    submit_btn = st.form_submit_button("🔍 Predict Risk")

if submit_btn:
    try:
        # Create DataFrame and sort columns to match model training order
        df_input = pd.DataFrame([data])
        df_input = df_input[FEATURES]
        
        # XGBoost handles unscaled data, so we predict directly
        prob = model.predict_proba(df_input)[0][1]
        
        st.divider()
        st.subheader("📊 Result Analysis")
        st.progress(float(prob))
        st.metric("Risk Score", f"{prob*100:.2f}%")
        
        HIGH_RISK_THRESHOLD = 0.65
        WARNING_THRESHOLD = 0.30

        if prob >= HIGH_RISK_THRESHOLD:
            st.error(f"⚠️ VERY HIGH RISK ({prob:.1%})")
            st.markdown("**👉 Action:** Immediate HbA1c and fasting glucose tests are recommended.")
        elif prob >= WARNING_THRESHOLD:
            st.warning(f"🟡 WARNING SIGNS ({prob:.1%})")
            st.markdown("**👉 Action:** Pre-diabetes risk. Reduce sugar/carbohydrates and increase physical activity.")
        else:
            st.success(f"✅ LOW RISK ({prob:.1%})")
            st.markdown("**👉 Action:** Maintain current lifestyle. Re-evaluate in 6 months.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# =========================================================
# 3. OPTION 2: UPLOAD FILE (EXCEL FORMAT)
# =========================================================
st.header("Option 2: Upload Excel / CSV")

# --- A. CREATE TEMPLATE ---
st.markdown("#### Step 1: Download Excel Template")
st.write("Please download this template, fill in data, and upload it back.")

example_data = {
    'Age':                     [25, 30, 45, 50, 60, 65, 70, 40, 55, 35, 24],
    'Gender':                  [0,  1,  1,  0,  1,  0,  1,  0,  1,  0,  1], 
    'BMI':                     [22.5, 24.0, 28.5, 31.2, 33.5, 29.0, 26.5, 35.1, 27.8, 21.0, 34.5], # High BMI
    'Waist_Circumference':     [75.0, 82.0, 95.0, 102.0, 108.0, 98.0, 92.0, 110.0, 96.0, 70.0, 105.0], # High Waist
    'Systolic_BP':             [110, 115, 128, 145, 150, 138, 130, 160, 125, 112, 138], # Stage 1 HTN
    'Diastolic_BP':            [70, 75, 82, 90, 95, 85, 80, 100, 78, 72, 88],
    'Family_History_Diabetes': [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0], # No family history (Lifestyle only)
    'History_Hypertension':    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1], # Already has history
    'History_Dyslipidemia':    [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    'Physical_Activity':       [1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0], # No activity
    'Education_Level':         [5, 4, 3, 2, 3, 2, 4, 3, 5, 4, 3], 
    'Race_Ethnicity':          [3, 3, 4, 1, 6, 7, 3, 4, 3, 6, 3]  
}

# Ensure template has correct columns
df_template = pd.DataFrame(example_data)
# Fill missing columns with default 0 if example_data doesn't cover all FEATURES
for col in FEATURES:
    if col not in df_template.columns:
        df_template[col] = 0 
df_template = df_template[FEATURES]

excel_template = convert_df_to_excel(df_template)

st.download_button(
    label="⬇️ Download Template (.xlsx)",
    data=excel_template,
    file_name="diabetes_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Click to download the Excel template."
)

st.markdown("---")

# --- B. UPLOAD AND PREDICT ---
st.markdown("#### Step 2: Upload Your File")
uploaded_file = st.file_uploader("Upload filled template", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
    else:
        df_upload = pd.read_excel(uploaded_file)

    st.write("Preview:")
    st.dataframe(df_upload.head())
    
    if st.button("📊 Predict from File"):
        try:
            # Align columns
            df_predict = df_upload[FEATURES]
            
            # Predict directly (No Scaler)
            probs = model.predict_proba(df_predict)[:,1]
            
            df_upload['Risk_Probability'] = probs
            df_upload['Risk_Level'] = np.where(probs >= 0.65, "High (Immediate Test)", 
                                              np.where(probs >= 0.30, "Warning (Pre-diabetes)", "Low"))
            
            st.success("Prediction completed!")
            st.dataframe(df_upload)
            
            # --- EXPORT RESULTS ---
            excel_result = convert_df_to_excel(df_upload)
            
            st.download_button(
                label="⬇️ Download Results (.xlsx)",
                data=excel_result,
                file_name="diabetes_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except KeyError as e:
            st.error(f"❌ Column mismatch! Please use the Template above.")
            st.error(f"Missing columns: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
