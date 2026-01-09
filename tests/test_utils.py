import pytest
from pathlib import Path
from explainable_aml.utils.validation import validate_features, validate_file_exists

def test_validate_features():
    features = {'a': 1, 'b': 2}
    required = ['a', 'b']
    validate_features(features, required) # Should pass
    
    with pytest.raises(ValueError):
        validate_features(features, ['c'])

def test_validate_file_exists(temp_dir):
    f = temp_dir / "test.txt"
    f.touch()
    validate_file_exists(f)
    
    with pytest.raises(FileNotFoundError):
        validate_file_exists(temp_dir / "nonexistent.txt")
