import logging
import json
from datetime import datetime
from explainable_aml.config import CONFIG

# Configure logging
logging.basicConfig(
    filename=str(CONFIG['log_path']),
    level=getattr(logging, CONFIG['logging_level']),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(event_type, data):
    """
    Log an event for governance and auditability.

    Args:
        event_type (str): Type of event (e.g., 'model_trained', 'transaction_scored', 'feedback_provided')
        data (dict): Event-specific data
    """
    # Convert numpy types to python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj

    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'data': convert_types(data)
    }

    logging.info(json.dumps(log_entry))

def log_decision(features, risk_score, alert_flag, explanation, nlp_explanation, feedback=None):
    """
    Log a decision event for governance and auditability.

    Args:
        features (dict): Transaction features
        risk_score (float): Predicted risk score
        alert_flag (bool): Alert flag
        explanation (dict): SHAP explanation
        nlp_explanation (str): NLP explanation
        feedback (str, optional): Analyst feedback
    """
    data = {
        'features': features,
        'risk_score': risk_score,
        'alert_flag': alert_flag,
        'explanation': explanation,
        'nlp_explanation': nlp_explanation,
        'feedback': feedback
    }
    log_event('decision_logged', data)

if __name__ == "__main__":
    # Example events
    log_event('model_trained', {'model_path': 'path/to/model', 'features': ['f1', 'f2']})
    log_event('transaction_scored', {'features': {'amount': 1000}, 'risk_score': 0.1})
    log_event('feedback_provided', {'feedback': 'Valid'})