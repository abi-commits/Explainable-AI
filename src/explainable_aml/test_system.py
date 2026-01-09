from explainable_aml.explainability.explain import explain_transaction
from explainable_aml.nlp.generate_explanation import generate_nlp_explanation
from explainable_aml.utils.logging import log_decision

def test_system():
    # Sample transaction features - borderline risk
    features = {
        'transaction_amount': 6000,
        'amount_deviation': 3500,
        'transaction_frequency': 4,
        'country_risk': 0.5,
        'customer_age': 35
    }

    print("Testing Explainable AML System")
    print("Transaction Features:", features)

    # Get explanation
    explanation = explain_transaction(features)
    print("Explanation:", explanation)

    # Generate NLP
    nlp_text = generate_nlp_explanation(explanation, features)
    print("NLP Explanation:", nlp_text)

    # Log decision
    log_decision(features, explanation['risk_score'], explanation['alert_flag'], explanation, nlp_text)
    print("Decision logged.")

if __name__ == "__main__":
    test_system()