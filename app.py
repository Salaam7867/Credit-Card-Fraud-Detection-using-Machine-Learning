import streamlit as st
import numpy as np
import pandas as pd
import pickle

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
# LOAD MODEL ARTIFACTS
# ===============================
@st.cache_resource
def load_artifacts():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
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
# QUICK DEMO (FOR RECRUITERS)
# ===============================
st.subheader("⚡ Quick Demo")

col1, col2 = st.columns(2)

def normal_transaction():
    return {
        "Amount": 45,
        "Time": 45000,
        "is_high_amount": 0,
        "log_amount": np.log1p(45)
    }

def high_risk_transaction():
    return {
        "Amount": 2500,
        "Time": 200,
        "is_high_amount": 1,
        "log_amount": np.log1p(2500)
    }

demo_input = None

with col1:
    if st.button("🟢 Simulate Normal Transaction"):
        demo_input = normal_transaction()

with col2:
    if st.button("🔴 Simulate High-Risk Transaction"):
        demo_input = high_risk_transaction()

# ===============================
# MANUAL INPUT (REALISTIC)
# ===============================
st.subheader("🧾 Transaction Details")

amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=120.0)
time = st.number_input("Transaction Time (seconds since first transaction)", min_value=0.0, value=40000.0)

log_amount = np.log1p(amount)
is_high_amount = int(amount > 2000)

# ===============================
# BUILD MODEL INPUT (HIDDEN PCA)
# ===============================
def build_model_input(amount, time, log_amount, is_high_amount):
    data = pd.DataFrame(columns=FEATURE_COLS)
    data.loc[0] = 0  # initialize all PCA features as 0

    # Inject engineered features
    if "Amount" in FEATURE_COLS:
        data.at[0, "Amount"] = amount
    if "Time" in FEATURE_COLS:
        data.at[0, "Time"] = time
    if "log_amount" in FEATURE_COLS:
        data.at[0, "log_amount"] = log_amount
    if "is_high_amount" in FEATURE_COLS:
        data.at[0, "is_high_amount"] = is_high_amount

    return data

# Use demo or manual input
if demo_input:
    X_input = build_model_input(**demo_input)
else:
    X_input = build_model_input(amount, time, log_amount, is_high_amount)

X_scaled = scaler.transform(X_input)
fraud_prob = model.predict_proba(X_scaled)[0][1]

# ===============================
# DECISION OUTPUT
# ===============================
st.subheader("📊 Fraud Risk Assessment")

if fraud_prob >= THRESHOLD:
    st.error(f"""
🟥 **HIGH FRAUD RISK**

**Probability:** {fraud_prob:.2%}  
**Action:** Block transaction & notify customer
""")
else:
    st.success(f"""
🟩 **LOW FRAUD RISK**

**Probability:** {fraud_prob:.2%}  
**Action:** Allow transaction
""")

# ===============================
# FOOTER
# ===============================
st.caption("Model: Logistic Regression | Threshold tuned for business cost")
