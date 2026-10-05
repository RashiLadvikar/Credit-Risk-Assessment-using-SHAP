import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #080b12 0%, #111827 100%);
    }

    /* Main container */
    .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 35px;
    }

    /* Section headings */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #f8fafc;
    }

    /* Labels */
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Inputs */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    /* Assess button */
    .stButton > button {
        width: 100%;
        height: 55px;
        border-radius: 14px;
        border: none;
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
        color: #111827;
        font-size: 17px;
        font-weight: 800;
    }

    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.30);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        padding: 18px;
        border-radius: 15px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748b;
        margin-top: 40px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("credit_risk_model.pkl")
    threshold = joblib.load("best_threshold.pkl")

    return model, threshold


model, threshold = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💳 Credit Risk Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered loan default risk prediction using '
    'machine learning and probability calibration.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    person_income = st.number_input(
        "Annual Income ($)",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )


with col2:

    person_emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0.0,
        max_value=60.0,
        value=5.0,
        step=0.5
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)",
        min_value=0,
        max_value=60,
        value=5,
        step=1
    )

    cb_person_default_on_file = st.selectbox(
        "Previous Default on File",
        ["N", "Y"]
    )


with col3:

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )


# ============================================================
# LOAN DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">💰 Loan Details</div>',
    unsafe_allow_html=True
)

col4, col5 = st.columns(2)

with col4:

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.1
    )


with col5:

    loan_percent_income = st.number_input(
        "Loan / Income Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
        format="%.2f"
    )


# ============================================================
# BUTTON
# ============================================================

st.write("")

left, center, right = st.columns([1, 2, 1])

with center:

    assess = st.button(
        "🔍  Assess Credit Risk"
    )


# ============================================================
# PREDICTION
# ============================================================

if assess:

    input_data = pd.DataFrame([{

        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length

    }])


    # Prediction probability

    probability = float(
        model.predict_proba(input_data)[0][1]
    )


    # Apply optimized threshold

    prediction = int(
        probability >= float(threshold)
    )


    st.write("")
    st.divider()


    # ========================================================
    # RESULT
    # ========================================================

    if prediction == 1:

        st.error(
            f"⚠️ HIGH RISK — Estimated Default Probability: "
            f"{probability * 100:.2f}%"
        )

    else:

        st.success(
            f"✅ LOW RISK — Estimated Default Probability: "
            f"{probability * 100:.2f}%"
        )


    # ========================================================
    # RESULT METRICS
    # ========================================================

    st.write("")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Default Probability",
            f"{probability * 100:.2f}%"
        )

    with metric2:

        st.metric(
            "Decision Threshold",
            f"{float(threshold) * 100:.2f}%"
        )

    with metric3:

        st.metric(
            "Risk Decision",
            "HIGH RISK" if prediction == 1 else "LOW RISK"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Credit Risk Assessment • XGBoost • Probability Calibration • "
    "FastAPI • Streamlit"
)