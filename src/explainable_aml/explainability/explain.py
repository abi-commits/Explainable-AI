import joblib
import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from explainable_aml.utils.logging import log_event
from explainable_aml.config import CONFIG
from explainable_aml.utils.validation import validate_file_exists, validate_features, validate_model_bundle

def explain_transaction(transaction_features: Dict[str, Any], model_path: Optional[str] = None, data_path: Optional[str] = None) -> Dict[str, Any]:
    if model_path is None:
        model_path = CONFIG['model_path']
    if data_path is None:
        data_path = CONFIG['data_path']
    """
    Explain a transaction's risk score using SHAP.

    Args:
        transaction_features (dict): Features of the transaction
        model_path (str): Path to the model bundle (containing model, features, threshold, metadata)
        data_path (str): Path to background data for SHAP

    Returns:
        dict: risk_score, alert_flag, top_features (list of dicts with feature, contribution)
    """
    try:
        validate_file_exists(model_path)
        validate_file_exists(data_path)

        # Load model bundle
        bundle = joblib.load(model_path)
        validate_model_bundle(bundle)
        
        model = bundle['model']
        features = bundle['features']
        threshold = bundle['threshold']
        feature_ranges = bundle.get('feature_ranges', {})

        validate_features(transaction_features, features)

        # Load background data for SHAP explainer
        df = pd.read_csv(data_path)
        X_background = df[features].sample(CONFIG['shap_background_samples'], random_state=42)  # Sample for efficiency

        # Create explainer
        explainer = shap.TreeExplainer(model, X_background)

        # Prepare input
        X_input = pd.DataFrame([transaction_features])

        # Predict risk score
        risk_score = float(model.predict_proba(X_input)[0][1])  # Probability of class 1
        alert_flag = bool(risk_score > threshold)  # Use threshold from bundle

        # Determine risk band
        if risk_score < 0.3:
            risk_band = "Low"
        elif risk_score < 0.7:
            risk_band = "Borderline"
        else:
            risk_band = "High"

        # Explain
        shap_values = explainer.shap_values(X_input)

        # For binary classification, shap_values is list of arrays for each class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For positive class

        # Get feature contributions
        feature_contributions = {}
        for i, feature in enumerate(features):
            feature_contributions[feature] = shap_values[0][i]

        # Sort by absolute contribution
        sorted_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        top_features = [{'feature': f, 'contribution': float(c)} for f, c in sorted_features[:5]]  # Top 5

        # Check for out-of-distribution features
        ood_features = []
        for feature in features:
            value = transaction_features[feature]
            if feature in feature_ranges:
                min_val = feature_ranges[feature]['min']
                max_val = feature_ranges[feature]['max']
                if value < min_val or value > max_val:
                    ood_features.append(feature)

        ood_flag = len(ood_features) > 0

        explanation = {
            'risk_score': risk_score,
            'alert_flag': alert_flag,
            'risk_band': risk_band,
            'top_features': top_features,
            'ood_flag': ood_flag,
            'ood_features': ood_features
        }

        # Log transaction scored event
        log_event('transaction_scored', {
            'features': transaction_features,
            'risk_score': risk_score,
            'alert_flag': alert_flag,
            'risk_band': risk_band,
            'top_features': top_features,
            'ood_flag': ood_flag,
            'ood_features': ood_features
        })

        return explanation

    except Exception as e:
        log_event('explanation_failed', {'error': str(e)})
        raise e

if __name__ == "__main__":
    # Example usage
    sample_features = {
        'transaction_amount': 15000,
        'amount_deviation': 10000,
        'transaction_frequency': 8,
        'country_risk': 0.8,
        'customer_age': 35
    }
    explanation = explain_transaction(sample_features)
    print(explanation)