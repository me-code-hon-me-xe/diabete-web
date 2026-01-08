import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# 1. SETUP & LOAD MODEL
# =========================================================
st.set_page_config(page_title="Diabetes Screening Tool", layout="centered", page_icon="🩺")

# Path to your English-trained model
MODEL_PATH = "diabetes_logreg_model_08012026.pkl"
SCALER_PATH = "diabetes_logreg_scaler_08012026.pkl"

@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File not found: {MODEL_PATH}")
        return None, None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Get feature names from the model (Expecting English names now)
    # e.g., ['Age', 'Gender', 'BMI', 'Waist_Circumference'...]
    features = list(model.feature_names_in_) 
    return model, scaler, features

model, scaler, FEATURES = load_prediction_model()

if model is None:
    st.stop()

st.title("🩺 Diabetes Risk Screening Tool")
st.markdown(f"**Model Features:** `{', '.join(FEATURES)}`")
st.write("Enter patient data manually **or** upload an Excel/CSV file.")

# =========================================================
# 2. OPTION 1: MANUAL INPUT
# =========================================================
st.header("Option 1: Manual Input")

with st.form("manual_form"):
    # We store data directly into English keys
    data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        data['Age'] = st.number_input("Age (years)", 1, 120, 40)
        
        # Gender: Male=1, Female=0
        g_display = st.selectbox("Gender", ["Male", "Female"])
        data['Gender'] = 1 if g_display == "Male" else 0
        
        data['BMI'] = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0)
        
        data['Waist_Circumference'] = st.number_input("Waist Circumference (cm)", 50.0, 160.0, 90.0)
        
        data['Systolic_BP'] = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
        data['Diastolic_BP'] = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)

    with col2:
        # Family History
        fam_disp = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        data['Family_History_Diabetes'] = 1 if fam_disp == "Yes" else 0
        
        # Hypertension
        bp_hist_disp = st.selectbox("History of Hypertension", ["No", "Yes"])
        data['History_Hypertension'] = 1 if bp_hist_disp == "Yes" else 0
        
        # Dyslipidemia
        chol_hist_disp = st.selectbox("History of Dyslipidemia", ["No", "Yes"])
        data['History_Dyslipidemia'] = 1 if chol_hist_disp == "Yes" else 0
        
        # Physical Activity
        act_disp = st.selectbox("Regular Physical Activity", ["No", "Yes"])
        data['Physical_Activity'] = 1 if act_disp == "Yes" else 0
        
        # Education Level
        # 1: <9th, 2: 9-11th, 3: High school, 4: Some college, 5: College+
        edu_disp = st.selectbox("Education Level", 
            ["< 9th Grade", "9-11th Grade", "High School", "Some College", "College Grad"])
        edu_map = {"< 9th Grade":1, "9-11th Grade":2, "High School":3, "Some College":4, "College Grad":5}
        data['Education_Level'] = edu_map[edu_disp]
        
        # Race / Ethnicity
        # 1: Mexican, 2: Other Hispanic, 3: White, 4: Black, 6: Asian, 7: Other
        race_disp = st.selectbox("Race / Ethnicity", 
            ["Mexican", "Other Hispanic", "White", "Black", "Asian", "Other"])
        race_map = {"Mexican":1, "Other Hispanic":2, "White":3, "Black":4, "Asian":6, "Other":7}
        data['Race_Ethnicity'] = race_map[race_disp]

    submit_btn = st.form_submit_button("🔍 Predict Risk")

if submit_btn:
    try:
        # Create DataFrame
        df_input = pd.DataFrame([data])
        
        # Ensure column order matches the model
        df_input = df_input[FEATURES]
        
        # Scale & Predict
        X_scaled = scaler.transform(df_input)
        prob = model.predict_proba(X_scaled)[0][1]
        
        st.divider()
        st.subheader("📊 Result Analysis")
        st.metric("Diabetes Risk Probability", f"{prob*100:.2f}%")
        st.progress(prob)
        
        
        if prob >= HIGH_RISK_THRESHOLD:
            # Thay print bằng st.error và st.markdown
            st.error(f"⚠️ VERY HIGH RISK ({prob:.1%})")
            st.markdown("**👉 Action:** Immediate HbA1c and fasting glucose tests are recommended.")
            
        elif prob >= WARNING_THRESHOLD:
            # Thay print bằng st.warning
            st.warning(f"🟡 WARNING SIGNS ({prob:.1%})")
            st.markdown("**👉 Action:** Pre-diabetes risk. Reduce sugar/carbohydrates and increase physical activity.")
            
        else:
            # Thay print bằng st.success
            st.success(f"✅ LOW RISK ({prob:.1%})")
            st.markdown("**👉 Action:** Maintain current lifestyle. Re-evaluate in 6 months.")
            
    except KeyError as e:
        st.error(f"❌ Key Error: The model expects feature '{e}'. Please check your English feature names.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# =========================================================
# 3. OPTION 2: UPLOAD FILE
# =========================================================
st.header("Option 2: Upload Excel / CSV")
st.info(f"Required columns: {', '.join(FEATURES)}")

uploaded_file = st.file_uploader("Upload file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
    else:
        df_upload = pd.read_excel(uploaded_file)

    st.write("Preview:")
    st.dataframe(df_upload.head())
    
    if st.button("📊 Predict from File"):
        try:
            # Prepare data
            df_predict = df_upload[FEATURES]
            
            # Predict
            X_scaled_up = scaler.transform(df_predict)
            probs = model.predict_proba(X_scaled_up)[:,1]
            
            # Add results
            df_upload['Risk_Probability'] = probs
            df_upload['Risk_Level'] = np.where(probs >= 0.65, "High", 
                                      np.where(probs >= 0.30, "Medium", "Low"))
            
            st.success("Prediction completed!")
            st.dataframe(df_upload)
            
            # Download
            csv = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Result CSV", csv, "results.csv", "text/csv")
            
        except KeyError as e:
            st.error(f"❌ Missing columns! The model requires: {FEATURES}")
            st.error(f"Missing: {e}")
        except Exception as e:
            st.error(f"Error: {e}")
