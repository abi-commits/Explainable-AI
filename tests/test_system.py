import pytest
import os
from explainable_aml.explainability.explain import explain_transaction
from explainable_aml.nlp.generate_explanation import generate_nlp_explanation
from explainable_aml.utils.logging import log_event
from explainable_aml.config import CONFIG
from explainable_aml.model.train_model import train_risk_model

@pytest.fixture(scope="module")
def ensure_model_exists():
    if not os.path.exists(CONFIG['model_path']):
        print("Model bundle not found. Training model for system test...")
        train_risk_model()
    return CONFIG['model_path']

def test_full_flow(ensure_model_exists):
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
    # Note: this uses the CONFIG paths by default
    explanation = explain_transaction(features)
    
    assert "risk_score" in explanation
    assert "alert_flag" in explanation
    assert "top_features" in explanation

    # Generate NLP
    nlp_text = generate_nlp_explanation(explanation, features)
    
    assert nlp_text is not None
    assert "text" in nlp_text

    # Log decision
    log_event('decision_logged', {'features': features, 'nlp': nlp_text})
