import numpy as np
import pandas as pd
import joblib
import streamlit as st


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ===============================
# LOAD MODEL ARTIFACTS (SAFE)
# ===============================
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    threshold = joblib.load("threshold.pkl")
    return model, scaler, feature_cols, threshold

model, scaler, feature_cols, THRESHOLD = load_artifacts()

# ===============================
# HEADER
# ===============================
st.markdown(
    """
    <h1 style='text-align: center;'>💳 Credit Card Fraud Detection System</h1>
    <p style='text-align: center; color: gray;'>
    Real-time fraud risk assessment using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ===============================
# BUSINESS CONTEXT
# ===============================
with st.expander("📌 Business Context", expanded=True):
    st.write(
        """
        Banks must **detect fraudulent transactions early** while 
        **minimizing false alerts** that annoy genuine customers.

        This model:
        - Prioritizes **high fraud recall**
        - Uses **probability thresholding**
        - Balances **bank loss vs customer friction**
        """
    )

# ===============================
# INPUT SECTION
# ===============================
st.subheader("🧾 Transaction Details")

input_data = {}
cols = st.columns(2)

for i, col in enumerate(feature_cols):
    with cols[i % 2]:
        input_data[col] = st.number_input(
            label=col,
            value=0.0,
            format="%.4f"
        )

# ===============================
# PREDICTION
# ===============================
st.markdown("---")

if st.button("🔍 Analyze Transaction", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    fraud_prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    st.metric(
        label="Fraud Probability",
        value=f"{fraud_prob:.2%}"
    )

    if fraud_prob >= THRESHOLD:
        st.error("🚨 **High Risk: Fraudulent Transaction Detected**")
        st.caption("Action recommended: Block or manually verify transaction")
    else:
        st.success("✅ **Low Risk: Legitimate Transaction**")
        st.caption("Transaction can be safely approved")

    st.markdown("---")
    st.caption(
        f"Decision Threshold: {THRESHOLD:.2f} | Model: Logistic Regression"
    )

# ===============================
# FOOTER
# ===============================
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:12px; color:gray;'>
    Built by Mohd Abdul Salaam | Machine Learning Project
    </p>
    """,
    unsafe_allow_html=True
)
