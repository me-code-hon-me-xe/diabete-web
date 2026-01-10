import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io

# =========================================================
# 1. SETUP & LOAD MODEL
# =========================================================
st.set_page_config(page_title="Diabetes Screening Tool (RF)", layout="centered", page_icon="🩺")

# The filename for your exported Random Forest model
MODEL_PATH = "diabetes_rdfc_model_08012026.pkl"

@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}. Please ensure you have trained and saved the model first.")
        return None, None
    
    # Load the Random Forest model
    model = joblib.load(MODEL_PATH)
    
    # Random Forest models saved via joblib usually store features if they were trained on a DataFrame
    # If not, we define them manually based on your NHANES dataset
    features = [
        'Age', 'Gender', 'Race_Ethnicity', 'Education_Level', 'BMI', 
        'Waist_Circumference', 'Systolic_BP', 'Diastolic_BP', 
        'Family_History_Diabetes', 'Physical_Activity'
    ]
    return model, features

model, FEATURES = load_prediction_model()

# Helper function to export Excel
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

st.title("🩺 Diabetes Risk Screening (Random Forest)")
st.info("This model uses a Random Forest classifier tuned to minimize overfitting and maximize recall.")

if model is None:
    st.stop()

# =========================================================
# 2. OPTION 1: MANUAL INPUT
# =========================================================
st.header("Option 1: Individual Patient Assessment")

with st.form("manual_form"):
    data = {}
    col1, col2 = st.columns(2)
    
    with col1:
        data['Age'] = st.number_input("Age (years)", 1, 100, 45)
        g_display = st.selectbox("Gender", ["Male", "Female"])
        data['Gender'] = 1 if g_display == "Male" else 0
        data['BMI'] = st.number_input("BMI (kg/m²)", 10.0, 60.0, 28.0)
        data['Waist_Circumference'] = st.number_input("Waist Circumference (cm)", 50.0, 160.0, 95.0)
        data['Systolic_BP'] = st.number_input("Systolic BP (mmHg)", 80, 220, 125)

    with col2:
        data['Diastolic_BP'] = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)
        fam_disp = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        data['Family_History_Diabetes'] = 1 if fam_disp == "Yes" else 0
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

    submit_btn = st.form_submit_button("🔍 Calculate Risk Score")

if submit_btn:
    df_input = pd.DataFrame([data])[FEATURES]
    prob = model.predict_proba(df_input)[0][1]
    
    st.divider()
    st.subheader("📊 Results")
    
    # We use the 0.06 threshold calculated earlier for high sensitivity
    OPTIMAL_THRESHOLD = 0.06 
    
    st.metric("Probability Score", f"{prob*100:.1f}%")
    
    if prob >= 0.50:
        st.error("🚨 HIGH RISK: Probability significantly above baseline.")
    elif prob >= OPTIMAL_THRESHOLD:
        st.warning("⚠️ ELEVATED RISK: Screening suggested based on optimized sensitivity.")
    else:
        st.success("✅ LOW RISK: Probability is within normal screening range.")

# =========================================================
# 3. OPTION 2: BATCH PROCESSING
# =========================================================
st.header("Option 2: Batch Processing (Excel/CSV)")

uploaded_file = st.file_uploader("Upload patient list", type=["xlsx", "csv"])

if uploaded_file:
    df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    
    if st.button("🚀 Run Batch Prediction"):
        try:
            # Check if all required features exist in the file
            df_predict = df_upload[FEATURES]
            probs = model.predict_proba(df_predict)[:,1]
            
            df_upload['Risk_Score'] = probs
            df_upload['Recommendation'] = np.where(probs >= 0.06, "Follow-up Required", "Routine Check")
            
            st.write("Results Preview:")
            st.dataframe(df_upload.head())
            
            # Export
            st.download_button(
                label="⬇️ Download Full Results",
                data=convert_df_to_excel(df_upload),
                file_name="diabetes_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error processing file: {e}. Ensure columns match: {FEATURES}")
