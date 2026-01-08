%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# 1. SETUP & LOAD MODEL
# =========================================================
st.set_page_config(page_title="Diabetes Screening Tool", layout="centered")

# Vì bạn lấy từ GIT về, file sẽ nằm ngay thư mục hiện tại
# (Không cần đường dẫn /content/drive/... nữa)
MODEL_PATH = "diabetes_logreg_model_08012026.pkl"
SCALER_PATH = "diabetes_logreg_scaler_08012026.pkl"

# Hàm load model an toàn
@st.cache_resource
def load_prediction_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File not found: {MODEL_PATH}. Make sure you pulled from Git.")
        return None, None, None
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = list(model.feature_names_in_) # Lấy tên cột gốc (VD: Tuoi, Gioi_Tinh...)
    return model, scaler, features

model, scaler, FEATURES = load_prediction_model()

if model is None:
    st.stop()

st.title("🩺 Diabetes Risk Screening Tool")
st.write("Enter patient data manually **or** upload an Excel/CSV file.")

# =========================================================
# 2. OPTION 1: MANUAL INPUT
# =========================================================
st.header("Option 1: Manual Input")

with st.form("manual_form"):
    # Dictionary để lưu dữ liệu (Key phải khớp với tên cột lúc train model)
    # Chúng ta dùng biến tạm để nhập, sau đó gán vào dict này
    data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Age -> Tuoi
        data['Tuoi'] = st.number_input("Age (years)", 1, 120, 40)
        
        # Gender -> Gioi_Tinh
        # Sửa lỗi selectbox: Dùng list options, sau đó map thủ công
        g_display = st.selectbox("Gender", ["Male", "Female"])
        data['Gioi_Tinh'] = 1 if g_display == "Male" else 0
        
        # BMI -> Chi_So_BMI
        data['Chi_So_BMI'] = st.number_input("BMI (kg/m²)", 10.0, 60.0, 25.0)
        
        # Waist -> Vong_Eo
        data['Vong_Eo'] = st.number_input("Waist Circumference (cm)", 50.0, 160.0, 90.0)
        
        # BP -> HA
        data['HA_Tam_Thu'] = st.number_input("Systolic BP (mmHg)", 80, 220, 120)
        data['HA_Tam_Truong'] = st.number_input("Diastolic BP (mmHg)", 40, 140, 80)

    with col2:
        # Family History -> Di_Truyen_Gia_Dinh
        fam_disp = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        data['Di_Truyen_Gia_Dinh'] = 1 if fam_disp == "Yes" else 0
        
        # Hypertension History -> Tien_Su_Cao_HA
        bp_hist_disp = st.selectbox("History of Hypertension", ["No", "Yes"])
        data['Tien_Su_Cao_HA'] = 1 if bp_hist_disp == "Yes" else 0
        
        # Dyslipidemia History -> Tien_Su_Mo_Mau
        chol_hist_disp = st.selectbox("History of Dyslipidemia", ["No", "Yes"])
        data['Tien_Su_Mo_Mau'] = 1 if chol_hist_disp == "Yes" else 0
        
        # Physical Activity -> Van_Dong_The_Chat
        act_disp = st.selectbox("Regular Physical Activity", ["No", "Yes"])
        data['Van_Dong_The_Chat'] = 1 if act_disp == "Yes" else 0
        
        # Education -> Trinh_Do_Hoc_Van
        edu_disp = st.selectbox("Education Level", 
            ["< 9th Grade", "9-11th Grade", "High School", "Some College", "College Grad"])
        edu_map = {"< 9th Grade":1, "9-11th Grade":2, "High School":3, "Some College":4, "College Grad":5}
        data['Trinh_Do_Hoc_Van'] = edu_map[edu_disp]
        
        # Race -> Sac_Toc
        race_disp = st.selectbox("Race / Ethnicity", 
            ["Mexican", "Other Hispanic", "White", "Black", "Asian", "Other"])
        race_map = {"Mexican":1, "Other Hispanic":2, "White":3, "Black":4, "Asian":6, "Other":7}
        data['Sac_Toc'] = race_map[race_disp]

    submit_btn = st.form_submit_button("🔍 Predict Risk")

if submit_btn:
    # Tạo DataFrame đúng chuẩn model yêu cầu
    df_input = pd.DataFrame([data])
    
    # Đảm bảo đúng thứ tự cột
    df_input = df_input[FEATURES]
    
    # Scale & Predict
    X_scaled = scaler.transform(df_input)
    prob = model.predict_proba(X_scaled)[0][1]
    
    st.divider()
    st.subheader("📊 Result")
    st.metric("Diabetes Risk Probability", f"{prob*100:.2f}%")
    st.progress(prob)
    
    if prob >= 0.65:
        st.error("⚠️ VERY HIGH RISK – Immediate medical testing recommended")
    elif prob >= 0.30:
        st.warning("🟡 MODERATE RISK – Lifestyle intervention advised")
    else:
        st.success("✅ LOW RISK – Maintain healthy lifestyle")

# =========================================================
# 3. OPTION 2: UPLOAD FILE
# =========================================================
st.header("Option 2: Upload Excel / CSV")

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
            # Lưu ý: File upload phải có tiêu đề cột là Tiếng Việt (Tuoi, Vong_Eo...)
            # để khớp với model cũ. Nếu file là tiếng Anh, ta cần đổi tên cột.
            
            # Mapping tên cột tiếng Anh sang tiếng Việt (đề phòng file upload dùng tiếng Anh)
            # Nếu file đã là tiếng Việt thì dòng này không ảnh hưởng
            rename_map = {
                'Age': 'Tuoi', 'Gender': 'Gioi_Tinh', 'BMI': 'Chi_So_BMI',
                'Waist_Circumference': 'Vong_Eo', 'Systolic_BP': 'HA_Tam_Thu',
                'Diastolic_BP': 'HA_Tam_Truong', 'Family_History': 'Di_Truyen_Gia_Dinh',
                'Hypertension_History': 'Tien_Su_Cao_HA', 'Dyslipidemia_History': 'Tien_Su_Mo_Mau',
                'Physical_Activity': 'Van_Dong_The_Chat', 'Education': 'Trinh_Do_Hoc_Van',
                'Race': 'Sac_Toc'
            }
            df_upload.rename(columns=rename_map, inplace=True)
            
            # Lọc cột và dự đoán
            df_predict = df_upload[FEATURES]
            X_scaled_up = scaler.transform(df_predict)
            probs = model.predict_proba(X_scaled_up)[:,1]
            
            df_upload['Risk_Probability'] = probs
            df_upload['Risk_Level'] = np.where(probs >= 0.65, "High", 
                                      np.where(probs >= 0.30, "Medium", "Low"))
            
            st.success("Done!")
            st.dataframe(df_upload)
            
            # Download
            csv = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Result CSV", csv, "results.csv", "text/csv")
            
        except KeyError as e:
            st.error(f"❌ Column mismatch! The model expects these columns: {FEATURES}")
            st.error(f"Missing: {e}")
