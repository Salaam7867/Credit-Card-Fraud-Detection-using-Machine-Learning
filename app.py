import streamlit as st
import numpy as np
import pickle

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Bank Fraud Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

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
# HEADER
# ===============================
st.title("💳 Credit Card Fraud Risk System")
st.caption("Real-time fraud risk assessment used by banks")

st.info(
    "Banks must detect fraudulent transactions early while minimizing false alerts that "
    "annoy genuine customers."
)

# ===============================
# LAYOUT
# ===============================
left, right = st.columns([1, 1])

# ===============================
# INPUT PANEL (BUSINESS FRIENDLY)
# ===============================
with left:
    st.subheader("🧾 Transaction Details")

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        value=2500.0,
        step=100.0
    )

    hour = st.slider(
        "Transaction Hour",
        0, 23, 14
    )

    international = st.selectbox(
        "International Transaction?",
        ["No", "Yes"]
    )

    customer_risk = st.selectbox(
        "Customer Risk Profile",
        ["Low", "Medium", "High"]
    )

    predict_btn = st.button("🔍 Assess Fraud Risk")

# ===============================
# FEATURE MAPPING (IMPORTANT)
# ===============================
def build_model_input():
    """
    Converts business inputs → model-ready feature vector
    PCA features are auto-filled (real systems do this)
    """

    data = dict.fromkeys(FEATURE_COLS, 0.0)

    # Core features
    data["Amount"] = amount
    data["Time"] = hour * 3600  # convert hour → seconds

    # Derived logic
    data["log_amount"] = np.log1p(amount)
    data["is_high_amount"] = int(amount > 5000)

    # Risk heuristics
    if international == "Yes":
        data["V1"] = -2.5
        data["V2"] = 2.0

    if customer_risk == "High":
        data["V3"] = -3.0
    elif customer_risk == "Medium":
        data["V3"] = -1.5

    return np.array([data[col] for col in FEATURE_COLS]).reshape(1, -1)

# ===============================
# PREDICTION + DECISION
# ===============================
with right:
    st.subheader("📊 Fraud Risk Assessment")

    if predict_btn:
        X = build_model_input()
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]
        decision = prob >= THRESHOLD

        st.metric(
            "Fraud Probability",
            f"{prob:.2%}"
        )

        if decision:
            st.error("🟥 HIGH RISK — BLOCK TRANSACTION")
            st.write(
                "⚠️ Recommended Action: **Block transaction and notify customer**"
            )
        else:
            st.success("🟢 LOW RISK — APPROVE TRANSACTION")
            st.write(
                "✅ Recommended Action: **Approve transaction**"
            )

        st.caption(
            f"Decision threshold set at {THRESHOLD:.2f} to balance bank loss vs customer friction."
        )

# ===============================
# ADVANCED (OPTIONAL)
# ===============================
with st.expander("⚙️ Advanced: Model Internals"):
    st.write("PCA-based features and scaling are handled internally.")
    st.write("This abstraction mirrors real banking systems.")
