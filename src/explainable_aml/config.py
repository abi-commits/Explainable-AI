from pathlib import Path
import os
import yaml

# Project root: Explainable-AI directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_config():
    env = os.environ.get('ENV', 'dev')
    config_path = PROJECT_ROOT / 'config' / f'{env}.yaml'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Resolve paths
        for key in ['data_path', 'model_path', 'log_path']:
            if key in config:
                p = Path(config[key])
                if not p.is_absolute():
                     config[key] = PROJECT_ROOT / p
                else:
                     config[key] = p
        
        # Ensure log_path exists
        if 'log_path' not in config:
             config['log_path'] = PROJECT_ROOT / 'src' / 'explainable_aml' / 'logs' / 'aml_events.log'
             
        return config
    else:
        return {
            # File paths
            'data_path': PROJECT_ROOT / 'src' / 'explainable_aml' / 'data' / 'transactions.csv',
            'model_path': PROJECT_ROOT / 'src' / 'explainable_aml' / 'model' / 'risk_model_bundle.pkl',
            'log_path': PROJECT_ROOT / 'src' / 'explainable_aml' / 'logs' / 'aml_events.log',

            # Model hyperparameters
            'model_params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },

            # Decision threshold
            'threshold': 0.35,

            # SHAP parameters
            'shap_background_samples': 100,

            # Logging configuration
            'logging_level': 'INFO'
        }

CONFIG = load_config()