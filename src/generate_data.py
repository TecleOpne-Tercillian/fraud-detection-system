import pandas as pd
import numpy as np

np.random.seed(42)

n_users = 300
transactions_per_user = 40

fraud_types = [
    "Card Not Present Fraud",
    "Account Takeover",
    "Identity Fraud",
    "Transaction Anomaly",
    "Device Spoofing"
]

categories = [
    "groceries", "travel", "e-commerce",
    "subscriptions", "bank transfer",
    "luxury goods", "crypto exchange"
]

data = []

for user_id in range(1, n_users + 1):

    customer_id = f"U{str(user_id).zfill(5)}"

    base_amount = np.random.uniform(50, 300)
    home_lat = -23.5 + np.random.randn() * 0.1
    home_long = -46.6 + np.random.randn() * 0.1
    main_device = f"dev_{np.random.randint(1, 15)}"

    for _ in range(transactions_per_user):

        is_fraud = np.random.rand() < 0.04

        if is_fraud:
            amount = base_amount * np.random.uniform(4, 20)
            lat = home_lat + np.random.uniform(3, 25)
            long = home_long + np.random.uniform(3, 25)
            device_id = f"dev_{np.random.randint(20, 50)}"
            fraud_type = np.random.choice(fraud_types)
            hour = np.random.choice([0, 1, 2, 3, 4, 23])
        else:
            amount = base_amount * np.random.uniform(0.6, 1.4)
            lat = home_lat + np.random.randn() * 0.01
            long = home_long + np.random.randn() * 0.01
            device_id = main_device
            fraud_type = "None"
            hour = np.random.randint(8, 22)

        timestamp = pd.Timestamp("2026-04-01") + pd.Timedelta(
            days=np.random.randint(0, 15),
            hours=hour,
            minutes=np.random.randint(0, 60)
        )

        data.append([
            user_id,
            customer_id,
            amount,
            timestamp,
            lat,
            long,
            np.random.choice(categories),
            device_id,
            int(is_fraud),
            fraud_type
        ])

df = pd.DataFrame(data, columns=[
    "user_id",
    "customer_id",
    "amount",
    "timestamp",
    "lat",
    "long",
    "merchant_category",
    "device_id",
    "is_fraud",
    "fraud_type"
])

df.to_csv("data/transactions.csv", index=False)

print("✅ Dataset atualizacao")