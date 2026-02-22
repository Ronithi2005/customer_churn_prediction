import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Churn Intelligence System",
    layout="wide"
)

# ================= LOAD MODEL =================
model = joblib.load("churn_xgboost_model.pkl")

# ================= PREMIUM CSS =================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500&display=swap" rel="stylesheet">

<style>

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #f5f7fb;
    color: #111;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 72px;
    font-weight: 700;
    text-align: center;
    margin-top: 30px;
    letter-spacing: 1px;
    background: linear-gradient(90deg,#000,#444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #6b6b6b;
    margin-bottom: 50px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    margin-top: 20px;
    margin-bottom: 20px;
}

.stSelectbox, .stNumberInput, .stSlider {
    background: white;
    border-radius: 12px;
    padding: 8px;
}

div.stButton > button {
    background: linear-gradient(135deg, #000000, #434343);
    color: white !important;
    font-size: 18px;
    padding: 16px 40px;
    border-radius: 12px;
    border: none;
    display: block;
    margin: 60px auto;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.05);
    background: black;
    color: white !important;
}

.result-box {
    background: white;
    padding: 50px;
    border-radius: 20px;
    text-align: center;
    font-size: 32px;
    font-family: 'Playfair Display', serif;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
    margin-top: 40px;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="main-title">Customer Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict customer retention behavior using Machine Learning</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">Customer Information</div>', unsafe_allow_html=True)

# ================= INPUT UI =================
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])

with col3:
    device = st.selectbox("Device Protection", ["No", "Yes"])
    tech = st.selectbox("Tech Support", ["No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

col4, col5 = st.columns(2)

with col4:
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check","Mailed check",
         "Bank transfer (automatic)","Credit card (automatic)"]
    )

with col5:
    monthly = st.slider("Monthly Charges ($)",0.0,500.0,70.0)
    total = st.slider("Total Charges ($)",0.0,20000.0,1000.0)

# ================= PREDICTION =================
if st.button("Predict Customer Churn"):

    # ===== Encoding =====
    gender = 1 if gender=="Male" else 0
    senior = 1 if senior=="Yes" else 0
    partner = 1 if partner=="Yes" else 0
    dependents = 1 if dependents=="Yes" else 0
    phone = 1 if phone=="Yes" else 0
    multiple = 1 if multiple=="Yes" else 0
    online_security = 1 if online_security=="Yes" else 0
    online_backup = 1 if online_backup=="Yes" else 0
    device = 1 if device=="Yes" else 0
    tech = 1 if tech=="Yes" else 0
    tv = 1 if tv=="Yes" else 0
    movies = 1 if movies=="Yes" else 0
    paperless = 1 if paperless=="Yes" else 0

    internet = {"DSL":0,"Fiber optic":1,"No":2}[internet]
    contract = {"Month-to-month":0,"One year":1,"Two year":2}[contract]
    payment = {
        "Electronic check":0,
        "Mailed check":1,
        "Bank transfer (automatic)":2,
        "Credit card (automatic)":3
    }[payment]

    input_data = pd.DataFrame([[
        gender, senior, partner, dependents,
        tenure, phone, multiple, internet,
        online_security, online_backup,
        device, tech, tv, movies,
        contract, paperless, payment,
        monthly, total
    ]], columns=[
        'gender','SeniorCitizen','Partner','Dependents',
        'tenure','PhoneService','MultipleLines','InternetService',
        'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies','Contract','PaperlessBilling',
        'PaymentMethod','MonthlyCharges','TotalCharges'
    ])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.markdown('<div class="result-box">Customer is Likely to Churn</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box">Customer is Likely to Stay</div>', unsafe_allow_html=True)

    st.write(f"### Churn Probability: {round(prob*100,2)}%")
    st.progress(float(prob))