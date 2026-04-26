import pandas as pd
import numpy as np

np.random.seed(42)

n_users = 200
transactions_per_user = 50

data = []

for user_id in range(1, n_users + 1):

    base_amount = np.random.uniform(20, 200)
    home_lat = -23.5 + np.random.randn() * 0.1
    home_long = -46.6 + np.random.randn() * 0.1
    main_device = f"dev_{np.random.randint(1, 10)}"

    for i in range(transactions_per_user):

        is_fraud = np.random.rand() < 0.03  # 3% fraude

        if is_fraud:
            amount = base_amount * np.random.uniform(5, 15)
            lat = home_lat + np.random.uniform(5, 20)
            long = home_long + np.random.uniform(5, 20)
            device_id = f"dev_{np.random.randint(10, 30)}"
            hour = np.random.choice([1, 2, 3, 4])
        else:
            amount = base_amount * np.random.uniform(0.5, 1.5)
            lat = home_lat + np.random.randn() * 0.01
            long = home_long + np.random.randn() * 0.01
            device_id = main_device
            hour = np.random.randint(8, 22)

        timestamp = pd.Timestamp("2026-04-01") + pd.Timedelta(
            days=np.random.randint(0, 10),
            hours=hour,
            minutes=np.random.randint(0, 60)
        )

        merchant_category = np.random.choice([
            "food", "transport", "shopping", "electronics"
        ])

        data.append([
            user_id, amount, timestamp,
            lat, long, merchant_category,
            device_id, int(is_fraud)
        ])

df = pd.DataFrame(data, columns=[
    "user_id", "amount", "timestamp",
    "lat", "long", "merchant_category",
    "device_id", "is_fraud"
])

df.to_csv("data/transactions.csv", index=False)

print("✅ Dataset REALISTA gerado!")