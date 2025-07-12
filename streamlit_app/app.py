# streamlit_app/app.py 

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("scripts/best_churn_model.pkl")
scaler = joblib.load("scripts/scaler.pkl")

# App Config
st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📉 Customer Churn Predictor")
st.markdown("Use this tool to assess churn risk based on customer behavior.")

st.sidebar.title("🧠 About the App")
st.sidebar.markdown("Welcome to the **Real-Time Customer Churn Predictor**.")

with st.sidebar.expander("🔍 What It Does", expanded=False):
    st.markdown("""
- Predicts whether a customer is likely to churn  
- Accepts behavioral, transactional, and support input  
- Outputs real-time churn probability  
- Flags high-risk users for proactive retention  
""")

with st.sidebar.expander("⚙️ Technologies Used", expanded=False):
    st.markdown("""
- **Python**, **Scikit-learn** – for machine learning  
- **Streamlit** – for the web interface  
- **Pandas**, **NumPy** – for data processing  
- Trained on simulated CRM-style customer data  
""")

with st.sidebar.expander("🛠️ How to Use", expanded=False):
    st.markdown("""
1. Expand each section under **Customer Info**
2. Input customer data manually
3. Click **Predict Churn**
4. View churn risk + probability
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: gray;'>Built by <b>Aklilu Abera</b><br>Data Analyst | Data Science Professional</div>",
    unsafe_allow_html=True
)

st.markdown("---")
st.header("📋 Customer Info")

with st.form(key="churn_form"):
    # Group inputs into dropdowns for cleaner UI

    with st.expander("🧍 Customer Profile", expanded=True):
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure_days = st.number_input("Tenure (days)", min_value=0, max_value=3000, value=365)

    with st.expander("💸 Financial Behavior", expanded=False):
        total_spent = st.number_input("Total Spent ($)", min_value=0.0, step=10.0)
        avg_txn_value = st.number_input("Avg. Transaction Value", min_value=0.0, step=1.0)
        num_transactions = st.number_input("Number of Transactions", min_value=0)
        days_since_last_txn = st.number_input("Days Since Last Transaction", min_value=0)

    with st.expander("🛠️ Support Interaction", expanded=False):
        num_tickets = st.number_input("Support Tickets", min_value=0)
        avg_resolution_time = st.number_input("Avg. Resolution Time (hrs)", min_value=0.0)
        ticket_rate = st.number_input("Tickets per Month", min_value=0.0)

    with st.expander("🌐 Web Behavior", expanded=False):
        avg_session_duration = st.number_input("Avg. Session Duration (min)", min_value=0.0)
        total_session_time = st.number_input("Total Session Time (min)", min_value=0.0)
        avg_pages = st.number_input("Avg. Pages per Session", min_value=0.0)
        days_since_last_visit = st.number_input("Days Since Last Visit", min_value=0)

    submit = st.form_submit_button("🔮 Predict Churn")

if submit:
    # Prepare input
    input_data = pd.DataFrame([[ 
        age, tenure_days, total_spent, avg_txn_value, num_transactions, days_since_last_txn,
        num_tickets, avg_resolution_time, ticket_rate,
        avg_session_duration, total_session_time, avg_pages, days_since_last_visit
    ]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    pred = model.predict(input_scaled)[0]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.metric("Churn Probability", f"{prob:.2%}")

    if prob > 0.7:
        st.error("⚠️ High Risk of Churn")
    elif prob > 0.4:
        st.warning("🟠 Medium Risk of Churn")
    else:
        st.success("🟢 Low Risk of Churn")


