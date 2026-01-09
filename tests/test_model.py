import pytest
import os
from explainable_aml.model.train_model import train_risk_model
from explainable_aml.config import CONFIG

def test_train_risk_model(sample_data_path, temp_dir):
    model_path = str(temp_dir / "model.pkl")
    
    # Run training
    bundle = train_risk_model(data_path=sample_data_path, model_path=model_path)
    
    # Check bundle content
    assert "model" in bundle
    assert "features" in bundle
    assert bundle["threshold"] == CONFIG['threshold']
    assert len(bundle["features"]) == 5
    
    # Check file created
    assert os.path.exists(model_path)
    
    # Check model works
    model = bundle["model"]
    assert hasattr(model, "predict")
