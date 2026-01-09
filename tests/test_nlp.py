import pytest
from explainable_aml.nlp.generate_explanation import generate_nlp_explanation

def test_generate_nlp_explanation(sample_features):
    explanation_input = {
        "risk_score": 0.8,
        "alert_flag": True,
        "risk_band": "High",
        "top_features": [
            {'feature': 'transaction_amount', 'contribution': 0.5},
            {'feature': 'country_risk', 'contribution': 0.3}
        ],
        "ood_flag": False,
        "ood_features": []
    }
    
    nlp_result = generate_nlp_explanation(explanation_input, sample_features)
    
    assert "text" in nlp_result
    assert "pattern_id" in nlp_result
    assert "0.8" in nlp_result["text"]
    assert "High" in nlp_result["text"]
