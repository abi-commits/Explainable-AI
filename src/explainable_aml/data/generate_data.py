import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)  # For reproducibility

def generate_synthetic_data(n_customers=5000, max_tx_per_customer=10, version="1.0"):
    """
    Generate synthetic transaction data for AML prototype.

    Args:
        n_customers (int): Number of customers to generate
        max_tx_per_customer (int): Maximum transactions per customer
        version (str): Version of the data generator

    Features:
    - customer_id: Unique identifier for customer
    - transaction_id: Unique identifier for transaction
    - transaction_amount: Amount of transaction
    - amount_deviation: Absolute deviation from customer's average transaction amount
    - transaction_frequency: Number of transactions per day for the customer
    - country_risk: Risk score of the transaction country (0-1)
    - customer_age: Age of the customer
    - is_money_laundering: Binary target (1 if high risk, 0 otherwise)
    - data_version: Version of the data generator
    - generation_timestamp: Timestamp when data was generated
    - n_customers_param: Number of customers used in generation
    - max_tx_per_customer_param: Max transactions per customer used in generation

    Logic for target: High risk if deviation > 5000, frequency > 5, and country_risk > 0.5
    """
    generation_timestamp = datetime.now().isoformat()
    data = []

    for customer_id in range(1, n_customers + 1):
        # Generate customer attributes
        customer_age = np.random.randint(18, 80)
        customer_avg_amount = np.random.uniform(100, 10000)

        # Number of transactions for this customer
        n_tx = np.random.randint(1, max_tx_per_customer + 1)

        # Generate transaction dates (simulate over a week)
        base_date = datetime.now() - timedelta(days=7)
        tx_dates = [base_date + timedelta(days=np.random.randint(0, 7)) for _ in range(n_tx)]

        # Group by date to calculate frequency
        date_counts = pd.Series(tx_dates).value_counts()
        tx_frequencies = [date_counts[date] for date in tx_dates]

        for i, (tx_date, freq) in enumerate(zip(tx_dates, tx_frequencies)):
            transaction_id = f"{customer_id}_{i+1}"
            transaction_amount = np.random.normal(customer_avg_amount, customer_avg_amount * 0.5)
            transaction_amount = max(10, transaction_amount)  # Ensure positive

            amount_deviation = abs(transaction_amount - customer_avg_amount)

            # Country risk: random, but biased towards higher for some
            country_risk = np.random.beta(2, 5) if np.random.rand() > 0.7 else np.random.beta(1, 2)

            # Determine if money laundering (nuanced logic for varied risk levels)
            # Core high-risk: all three high
            core_risk = (amount_deviation > 4000 and freq > 6 and country_risk > 0.7)

            # Marginal risks: barely qualify
            marginal_risk = (
                (amount_deviation > 3500 and freq > 2) or  # High amount, low freq
                (freq > 5 and amount_deviation > 2000) or  # High freq, low deviation
                (country_risk > 0.4 and amount_deviation > 2500)  # Medium country + deviation
            )

            is_money_laundering = 1 if (core_risk or marginal_risk) else 0

            # Inject 2% label noise
            if np.random.rand() < 0.02:
                is_money_laundering = 1 - is_money_laundering

            data.append({
                'customer_id': customer_id,
                'transaction_id': transaction_id,
                'transaction_amount': transaction_amount,
                'amount_deviation': amount_deviation,
                'transaction_frequency': freq,
                'country_risk': country_risk,
                'customer_age': customer_age,
                'is_money_laundering': is_money_laundering,
                'transaction_date': tx_date,
                'data_version': version,
                'generation_timestamp': generation_timestamp,
                'n_customers_param': n_customers,
                'max_tx_per_customer_param': max_tx_per_customer
            })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(version="1.0")
    df.to_csv('src/explainable_aml/data/transactions.csv', index=False)
    print("Synthetic data generated and saved to src/explainable_aml/data/transactions.csv")