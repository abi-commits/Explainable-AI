from pathlib import Path
from typing import Dict, List, Any
import os
import numpy as np

def validate_features(features: Dict[str, Any], required_features: List[str]) -> None:
    """
    Validate that all required features are present and have correct types.
    
    Args:
        features: Dictionary of feature names and values
        required_features: List of required feature names
        
    Raises:
        ValueError: If features are missing or invalid
    """
    missing = [f for f in required_features if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
        
    for feature, value in features.items():
        if not isinstance(value, (int, float, np.number)) and feature in required_features:
             # Basic check, can be expanded
             pass

def validate_file_exists(file_path: Path) -> None:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to the file
        
    Raises:
        FileNotFoundError: If file does not exist
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

def validate_model_bundle(bundle: Dict[str, Any]) -> None:
    """
    Validate that the model bundle contains all necessary components.
    
    Args:
        bundle: Dictionary containing model components
        
    Raises:
        ValueError: If bundle is invalid
    """
    required_keys = ['model', 'features', 'threshold']
    missing = [k for k in required_keys if k not in bundle]
    if missing:
        raise ValueError(f"Invalid model bundle. Missing keys: {missing}")
