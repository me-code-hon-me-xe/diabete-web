import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io  # <--- Thư viện cần thiết để xuất file Excel

# =========================================================
# 1. SETUP & LOAD MODEL
# =========================================================
st.set_page_config(page_title="Diabetes Screening Tool", layout="centered", page_icon="🩺")

MODEL_PATH = "diabetes_logreg_model_08012026.pkl"
SCALER_PATH = "diabetes_logreg_scaler_08012026.pkl"

@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File not found: {MODEL_PATH}")
        return None, None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = list(model.feature_names_in_) 
    return model, scaler, features

model, scaler, FEATURES = load_prediction_model()

if model is None:
    st.stop()

# Hàm hỗ trợ xuất file Excel để download
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

st.title("🩺 Diabetes Risk Screening Tool")
st.markdown(f"**Model Features:** `{', '.join(FEATURES)}`")
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
        df_input = pd.DataFrame([data])
        df_input = df_input[FEATURES]
        X_scaled = scaler.transform(df_input)
        prob = model.predict_proba(X_scaled)[0][1]
        
        st.divider()
        st.subheader("📊 Result Analysis")
        st.progress(prob)
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

# --- A. TẠO FILE MẪU EXCEL (.xlsx) ---
st.markdown("#### Step 1: Download Excel Template")
st.write("Please download this template, fill in data, and upload it back.")

example_data = {
    'Age': [45, 60],
    'Gender': [1, 0],
    'BMI': [24.5, 31.2],
    'Waist_Circumference': [85.0, 102.0],
    'Systolic_BP': [120, 145],
    'Diastolic_BP': [80, 90],
    'Family_History_Diabetes': [0, 1],
    'History_Hypertension': [0, 1],
    'History_Dyslipidemia': [0, 1],
    'Physical_Activity': [1, 0],
    'Education_Level': [4, 2],
    'Race_Ethnicity': [3, 4]
}

df_template = pd.DataFrame(example_data)
df_template = df_template[FEATURES]

# Chuyển đổi sang Excel Buffer
excel_template = convert_df_to_excel(df_template)

st.download_button(
    label="⬇️ Download Template (.xlsx)",
    data=excel_template,
    file_name="diabetes_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # MIME chuẩn của Excel
    help="Click to download the Excel template."
)

st.markdown("---")

# --- B. UPLOAD VÀ DỰ ĐOÁN ---
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
            df_predict = df_upload[FEATURES]
            
            X_scaled_up = scaler.transform(df_predict)
            probs = model.predict_proba(X_scaled_up)[:,1]
            
            df_upload['Risk_Probability'] = probs*100%
            df_upload['Risk_Level'] = np.where(probs >= 0.65, "High (Immediate Test)", 
                                      np.where(probs >= 0.30, "Warning (Pre-diabetes)", "Low"))
         
            st.success("Prediction completed!")
            st.dataframe(df_upload)
            
            # --- CHUYỂN KẾT QUẢ SANG EXCEL ---
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
