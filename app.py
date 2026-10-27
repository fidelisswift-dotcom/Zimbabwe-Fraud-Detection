
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Mobile Money Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "fraud_detection_random_forest.joblib"

@st.cache_resource
def load_fraud_model():
    return joblib.load(MODEL_PATH)

st.title("🛡️ Mobile Money Fraud Detection")
st.subheader("Machine Learning-Based Fraud Risk Analysis")

st.info(
    "Educational prototype using synthetic PaySim data. "
    "Predictions are risk indicators, not proof of fraud."
)

if not MODEL_PATH.exists():
    st.error("The saved model file is missing. We need to locate it before analysing transactions.")
    st.stop()

try:
    saved_model = load_fraud_model()
    model = saved_model["model"]
    threshold = saved_model["threshold"]
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.stop()

with st.form("transaction_form"):
    transaction_type = st.selectbox(
        "Transaction Type",
        ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    )
    amount = st.number_input("Transaction Amount", min_value=0.0, value=50000.0)
    step = st.number_input("Transaction Time Step", min_value=1, value=100)
    oldbalance_org = st.number_input(
        "Sender's Balance Before Transaction", min_value=0.0, value=50000.0
    )
    oldbalance_dest = st.number_input(
        "Recipient's Balance Before Transaction", min_value=0.0, value=0.0
    )
    submitted = st.form_submit_button("Analyse Transaction", use_container_width=True)

if submitted:
    transaction = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest
    }])

    probability = model.predict_proba(transaction)[0, 1]

    st.subheader("Analysis Result")
    st.metric("Estimated Fraud Probability", f"{probability:.2%}")
    st.progress(float(probability))

    if probability >= threshold:
        st.error("HIGH RISK: Potentially Fraudulent")
    else:
        st.success("BELOW THRESHOLD: Likely Legitimate")

    st.caption(f"Decision threshold: {threshold:.0%}")
    st.warning(
        "This prediction is not proof of fraud. Do not automatically "
        "block a real transaction based solely on this prototype."
    )

st.divider()
st.caption("University project prototype | PaySim synthetic dataset")
