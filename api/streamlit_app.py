import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go

# -------------------------------
# Fix import path
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.preprocess import preprocess_data

# -------------------------------
# Load model
# -------------------------------
model = joblib.load("models/kmeans/model.pkl")
scaler = joblib.load("models/kmeans/scaler.pkl")
features = joblib.load("models/kmeans/features.pkl")
pca = joblib.load("models/kmeans/pca.pkl")

# -------------------------------
# UI STYLE
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.block {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# INTERPRETATION (FIXED LOGIC)
# -------------------------------
def interpret_customer(cluster, risk_score):
    if cluster == 1 and risk_score > 70:
        return "⚠️ Risky High Spender", "High usage but dangerous repayment pattern"
    elif cluster == 1:
        return "🟢 High Value Customer", "Active spender with good repayment"
    else:
        return "🔴 Low Activity Customer", "Low engagement user"

# -------------------------------
# UI
# -------------------------------
st.title("💳 Credit Card Customer Intelligence")
st.caption("AI-driven segmentation + risk analytics")

# Sidebar
st.sidebar.header("Customer Input")

balance = st.sidebar.number_input("Balance", 0.0, 100000.0, 5000.0)
purchases = st.sidebar.number_input("Purchases", 0.0, 100000.0, 3000.0)
cash_advance = st.sidebar.number_input("Cash Advance", 0.0, 100000.0, 0.0)
credit_limit = st.sidebar.number_input("Credit Limit", 0.0, 100000.0, 10000.0)
payments = st.sidebar.number_input("Payments", 0.0, 100000.0, 4000.0)
min_payments = st.sidebar.number_input("Minimum Payments", 0.0, 10000.0, 500.0)
purchases_trx = st.sidebar.number_input("Purchase Transactions", 0, 500, 30)
cash_trx = st.sidebar.number_input("Cash Advance Transactions", 0, 200, 0)
prc_full = st.sidebar.slider("Full Payment %", 0.0, 1.0, 0.5)
tenure = st.sidebar.slider("Tenure", 1, 12, 12)

# -------------------------------
# Predict
# -------------------------------
if st.button("🚀 Analyze Customer"):

    df = pd.DataFrame([{
        "BALANCE": balance,
        "PURCHASES": purchases,
        "CASH_ADVANCE": cash_advance,
        "CREDIT_LIMIT": credit_limit,
        "PAYMENTS": payments,
        "MINIMUM_PAYMENTS": min_payments,
        "PURCHASES_TRX": purchases_trx,
        "CASHADVANCETRX": cash_trx,
        "PRCFULLPAYMENT": prc_full,
        "TENURE": tenure
    }])

    # preprocessing
    df = preprocess_data(df)
    df = df.reindex(columns=features, fill_value=0)

    X = scaler.transform(df)
    X = pca.transform(X)

    cluster = int(model.predict(X)[0])

    # -------------------------------
    # REALISTIC CONFIDENCE
    # -------------------------------
    distances = model.transform(X)
    confidence = 1 / (1 + np.min(distances)) * 100

    # -------------------------------
    # BUSINESS METRICS
    # -------------------------------
    utilization = balance / (credit_limit + 1e-6)
    payment_ratio = payments / (balance + 1e-6)

    # -------------------------------
    # RISK SCORE (IMPROVED)
    # -------------------------------
    risk_score = (
        (cash_advance / (credit_limit + 1e-6)) * 0.4 +
        (1 - prc_full) * 0.3 +
        utilization * 0.3
    ) * 100

    risk_score = max(0, min(risk_score, 100))

    segment, desc = interpret_customer(cluster, risk_score)

    # -------------------------------
    # MAIN RESULT
    # -------------------------------
    st.markdown(f"""
    <div class="block">
    <h3>{segment}</h3>
    <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    

    # -------------------------------
    # KEY METRICS
    # -------------------------------
    st.subheader("📊 Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Utilization", f"{utilization:.2f}")
    c2.metric("Payment Ratio", f"{payment_ratio:.2f}")
    c3.metric("Tenure", tenure)

    st.info("Utilization > 1 means customer is over limit → high risk")

    # -------------------------------
    # RISK GAUGE (CLEAN)
    # -------------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={'suffix': "%"},
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if risk_score > 70 else "orange" if risk_score > 30 else "green"}
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.info("Risk score combines utilization, repayment, and cash behavior")

    # -------------------------------
    # SPENDING
    # -------------------------------
    st.subheader("💳 Spending vs Credit")

    st.bar_chart(pd.DataFrame({
        "Value": [balance, purchases, credit_limit]
    }, index=["Balance", "Purchases", "Limit"]))

    # -------------------------------
    # PAYMENT
    # -------------------------------
    st.subheader("💰 Payment Behavior")

    st.bar_chart(pd.DataFrame({
        "Value": [payments, min_payments]
    }, index=["Paid", "Minimum"]))

    # -------------------------------
    # RADAR
    # -------------------------------
    st.subheader("📈 Behavioral Profile")

    fig = go.Figure(go.Scatterpolar(
        r=[
            utilization,
            payment_ratio,
            purchases/(credit_limit+1e-6),
            cash_advance/(credit_limit+1e-6)
        ],
        theta=['Utilization','Payment','Spending','Cash'],
        fill='toself'
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # ALERTS
    # -------------------------------
    st.subheader("🚨 Key Alerts")

    alerts = []

    if utilization > 1:
        alerts.append("Customer exceeding credit limit")

    if prc_full < 0.3:
        alerts.append("Low full-payment behavior")

    if cash_advance > purchases:
        alerts.append("Heavy dependency on cash advances")

    if payment_ratio < 0.2:
        alerts.append("Very weak repayment capacity")

    if not alerts:
        st.success("No major risks detected")
    else:
        for a in alerts:
            st.warning(a)

    # -------------------------------
    # FINAL DECISION BLOCK (NEW)
    # -------------------------------
    st.subheader("📌 Final Decision Insight")

    if risk_score > 70:
        st.error("⚠️ High risk → Avoid increasing credit limit")
    elif risk_score > 30:
        st.warning("⚠️ Moderate risk → Monitor customer behavior")
    else:
        st.success("✅ Safe customer → Can offer premium benefits")