import streamlit as st
import pandas as pd
from explainable_aml.explainability.explain import explain_transaction
from explainable_aml.nlp.generate_explanation import generate_nlp_explanation
from explainable_aml.utils.logging import log_decision, log_event

# Health check logic
if "health" in st.query_params:
    st.write("OK")
    st.stop()

st.title("Explainable AML System")

st.sidebar.header("Transaction Details")

transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=1000.0)
amount_deviation = st.sidebar.number_input("Amount Deviation", min_value=0.0, value=100.0)
transaction_frequency = st.sidebar.number_input("Transaction Frequency", min_value=1, value=1)
country_risk = st.sidebar.slider("Country Risk", 0.0, 1.0, 0.1)
customer_age = st.sidebar.number_input("Customer Age", min_value=18, value=30)

if st.sidebar.button("Analyze Transaction"):
    features = {
        'transaction_amount': transaction_amount,
        'amount_deviation': amount_deviation,
        'transaction_frequency': transaction_frequency,
        'country_risk': country_risk,
        'customer_age': customer_age
    }

    explanation = explain_transaction(features)
    nlp_text = generate_nlp_explanation(explanation, features)

    # Log the decision
    log_decision(features, explanation['risk_score'], explanation['alert_flag'], explanation, nlp_text)

    st.subheader("Risk Assessment")
    st.write(f"Risk Score: {explanation['risk_score']:.4f}")
    st.write(f"Risk Band: {explanation.get('risk_band', 'Unknown')}")
    st.write(f"Alert Flag: {'Yes' if explanation['alert_flag'] else 'No'}")
    if explanation.get('ood_flag', False):
        st.warning(f"Out-of-Distribution Warning: Features {explanation['ood_features']} are outside training range. Model assessment may be unreliable.")

    st.subheader("Feature Contributions")
    contrib_df = pd.DataFrame(explanation['top_features'])
    contrib_df['contribution'] = contrib_df['contribution'].abs()  # For bar chart
    st.bar_chart(contrib_df.set_index('feature')['contribution'])

    st.subheader("Explanation")
    st.write(nlp_text)

    st.subheader("Analyst Feedback")
    feedback = st.radio("Is this assessment valid?", ("Valid", "Invalid"))
    if st.button("Submit Feedback"):
        # Log feedback event
        log_event('feedback_provided', {
            'features': features,
            'risk_score': explanation['risk_score'],
            'alert_flag': explanation['alert_flag'],
            'feedback': feedback
        })
        st.write("Feedback submitted.")