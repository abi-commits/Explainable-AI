import pytest
import numpy as np
from explainable_aml.model.train_model import train_risk_model
from explainable_aml.explainability.explain import explain_transaction
from explainable_aml.config import CONFIG

@pytest.fixture
def trained_model_path(sample_data_path, temp_dir):
    model_path = str(temp_dir / "model.pkl")
    # Train with sample data
    train_risk_model(data_path=sample_data_path, model_path=model_path)
    return model_path

def test_explain_transaction(trained_model_path, sample_data_path, sample_features):
    # Reduce sampling for test data (which is small)
    original_samples = CONFIG['shap_background_samples']
    CONFIG['shap_background_samples'] = 5
    
    try:
        explanation = explain_transaction(
            transaction_features=sample_features,
            model_path=trained_model_path,
            data_path=sample_data_path
        )
        
        assert "risk_score" in explanation
        assert "alert_flag" in explanation
        assert "top_features" in explanation
        assert len(explanation["top_features"]) <= 5
        assert isinstance(explanation["risk_score"], float)
        assert isinstance(explanation["alert_flag"], (bool, np.bool_))
    finally:
        CONFIG['shap_background_samples'] = original_samples
