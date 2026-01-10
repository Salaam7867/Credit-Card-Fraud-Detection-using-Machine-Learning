import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection System")
st.caption("Real-time fraud risk assessment using Machine Learning")

# ===============================
# LOAD MODEL ARTIFACTS (JOBLIB)
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, feature_cols, threshold

model, scaler, FEATURE_COLS, THRESHOLD = load_artifacts()

# ===============================
# BUSINESS CONTEXT
# ===============================
with st.expander("📌 Business Context", expanded=True):
    st.write("""
Banks must detect fraudulent transactions early **without blocking genuine customers**.

This system:
- Prioritizes **high fraud recall**
- Uses **probability thresholding**
- Balances **bank loss vs customer friction**
""")

# ===============================
# SIMPLE REALISTIC INPUTS
# ===============================
st.subheader("🧾 Transaction Details")

amount = st.number_input(
    "Transaction Amount ($)",
    min_value=0.0,
    value=120.0,
    step=10.0
)

time = st.number_input(
    "Transaction Time (seconds since first transaction)",
    min_value=0.0,
    value=40000.0,
    step=1000.0
)

# Feature engineering (same as training)
log_amount = np.log1p(amount)
is_high_amount = int(amount > 2000)

# ===============================
# BUILD MODEL INPUT (HIDE PCA)
# ===============================
X = pd.DataFrame(columns=FEATURE_COLS)
X.loc[0] = 0  # initialize all PCA columns as 0

if "Amount" in FEATURE_COLS:
    X.at[0, "Amount"] = amount
if "Time" in FEATURE_COLS:
    X.at[0, "Time"] = time
if "log_amount" in FEATURE_COLS:
    X.at[0, "log_amount"] = log_amount
if "is_high_amount" in FEATURE_COLS:
    X.at[0, "is_high_amount"] = is_high_amount

X_scaled = scaler.transform(X)
fraud_prob = model.predict_proba(X_scaled)[0][1]

# ===============================
# DECISION OUTPUT
# ===============================
st.subheader("📊 Fraud Risk Assessment")

if fraud_prob >= THRESHOLD:
    st.error(f"""
🟥 **HIGH FRAUD RISK**

**Probability:** {fraud_prob:.2%}  
**Action:** Block transaction & alert customer
""")
else:
    st.success(f"""
🟩 **LOW FRAUD RISK**

**Probability:** {fraud_prob:.2%}  
**Action:** Allow transaction
""")

st.caption("Model: Logistic Regression | Threshold tuned for business cost")
