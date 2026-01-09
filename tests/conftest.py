import pytest
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

@pytest.fixture
def sample_data():
    # Generate larger sample data to satisfy train_test_split requirements
    n_samples = 20
    data = {
        'transaction_amount': np.random.uniform(10, 100000, n_samples),
        'amount_deviation': np.random.uniform(0, 1000, n_samples),
        'transaction_frequency': np.random.randint(1, 20, n_samples),
        'country_risk': np.random.uniform(0, 1, n_samples),
        'customer_age': np.random.randint(18, 90, n_samples),
        'is_money_laundering': np.random.choice([0, 1], n_samples),
        'data_version': ['v1'] * n_samples
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_dir():
    # Create temp dir
    dir_path = Path("tests/temp")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)
    yield dir_path
    # Cleanup
    if dir_path.exists():
        shutil.rmtree(dir_path)

@pytest.fixture
def sample_data_path(temp_dir, sample_data):
    path = temp_dir / "transactions.csv"
    sample_data.to_csv(path, index=False)
    return str(path)

@pytest.fixture
def sample_features():
    return {
        'transaction_amount': 100.0,
        'amount_deviation': 0.0,
        'transaction_frequency': 1,
        'country_risk': 0.1,
        'customer_age': 30
    }
