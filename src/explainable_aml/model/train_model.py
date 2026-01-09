import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from explainable_aml.utils.logging import log_event
from explainable_aml.config import CONFIG
from explainable_aml.utils.validation import validate_file_exists

def train_risk_model(data_path: Optional[str] = None, model_path: Optional[str] = None) -> Dict[str, Any]:
    if data_path is None:
        data_path = CONFIG['data_path']
    if model_path is None:
        model_path = CONFIG['model_path']
    """
    Train an XGBoost model to predict money laundering risk and save a model bundle.

    Features: transaction_amount, amount_deviation, transaction_frequency, country_risk, customer_age
    Target: is_money_laundering (binary)

    Returns a bundle containing model, features, threshold, and training metadata.
    """
    try:
        validate_file_exists(data_path)

        # Load data
        df = pd.read_csv(data_path)
        data_version = df['data_version'].iloc[0] if 'data_version' in df.columns else 'unknown'

        # Select features
        features = ['transaction_amount', 'amount_deviation', 'transaction_frequency', 'country_risk', 'customer_age']
        X = df[features]
        y = df['is_money_laundering']

        # Compute feature ranges for OOD detection
        feature_ranges = {}
        for feature in features:
            feature_ranges[feature] = {
                'min': X[feature].min(),
                'max': X[feature].max()
            }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train XGBoost model
        model = xgb.XGBClassifier(**CONFIG['model_params'])
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Create model bundle
        model_bundle = {
            "model": model,
            "features": features,
            "threshold": CONFIG['threshold'],
            "feature_ranges": feature_ranges,
            "training_data_version": f"data_v{data_version}",
            "trained_at": datetime.now().isoformat()
        }

        # Save bundle
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_bundle, model_path)
        print(f"Model bundle saved to {model_path}")

        # Log model trained event
        log_event('model_trained', {
            'model_path': str(model_path),
            'features': features,
            'threshold': model_bundle['threshold'],
            'training_data_version': model_bundle['training_data_version'],
            'trained_at': model_bundle['trained_at']
        })

        return model_bundle

    except Exception as e:
        log_event('model_training_failed', {'error': str(e)})
        raise e

if __name__ == "__main__":
    train_risk_model()