import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

data = []

for i in range(n):
    user_id = np.random.randint(1, 100)

    amount = np.random.exponential(scale=100)

    is_fraud = 1 if np.random.rand() < 0.05 else 0

    if is_fraud:
        amount *= np.random.uniform(5, 20)

    lat = -23 + np.random.randn()
    long = -46 + np.random.randn()

    merchant_category = np.random.choice([
        "food", "electronics", "transport", "jewelry"
    ])

    device_id = f"dev_{np.random.randint(1, 20)}"

    timestamp = pd.Timestamp("2026-04-01") + pd.to_timedelta(np.random.randint(0, 100000), unit="s")

    data.append([
        user_id, amount, timestamp,
        lat, long, merchant_category,
        device_id, is_fraud
    ])

df = pd.DataFrame(data, columns=[
    "user_id", "amount", "timestamp",
    "lat", "long", "merchant_category",
    "device_id", "is_fraud"
])

df.to_csv("data/transactions.csv", index=False)

print("✅ Dataset gerado com sucesso!")