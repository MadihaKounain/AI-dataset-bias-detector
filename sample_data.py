"""
sample_data.py
--------------
Generates a realistic synthetic loan approval dataset with known biases
for demonstration purposes.
"""

import pandas as pd
import numpy as np


def generate_loan_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], n, p=[0.6, 0.4])
    age = rng.integers(22, 65, n)
    race = rng.choice(["White", "Black", "Asian", "Hispanic"], n, p=[0.55, 0.20, 0.15, 0.10])
    income = rng.normal(55000, 20000, n).clip(15000, 200000).round(0)
    credit_score = rng.integers(300, 850, n)
    loan_amount = rng.normal(20000, 8000, n).clip(5000, 80000).round(0)
    employment_years = rng.integers(0, 30, n)
    education = rng.choice(["High School", "Bachelor", "Master", "PhD"], n, p=[0.35, 0.40, 0.18, 0.07])

    # Introduce bias: females & Black/Hispanic applicants have lower approval rate
    base_prob = (
        0.35
        + 0.25 * (credit_score > 650).astype(float)
        + 0.15 * (income > 50000).astype(float)
        + 0.10 * (employment_years > 5).astype(float)
        - 0.08 * (gender == "Female").astype(float)          # gender bias
        - 0.10 * np.isin(race, ["Black", "Hispanic"]).astype(float)  # racial bias
    )
    base_prob = base_prob.clip(0.05, 0.95)
    approved = (rng.random(n) < base_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "race": race,
        "education": education,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "loan_approved": approved,
    })

    # Add 3% missing values to a few columns
    for col in ["income", "credit_score", "employment_years"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    # Add a few duplicate rows
    dupes = df.sample(15, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


if __name__ == "__main__":
    df = generate_loan_dataset()
    df.to_csv("sample_loan_data.csv", index=False)
    print(f"✅ Sample dataset saved: {df.shape[0]} rows × {df.shape[1]} cols")
    print(df.head())
