import pandas as pd

# carregar dataset
df = pd.read_csv("data/transactions.csv")

# ver primeiras linhas
print(df.head())

# estatísticas básicas
print(df.describe())

# quantas fraudes existem
print(df["is_fraud"].value_counts())

# média de valor por usuário
print(df.groupby("user_id")["amount"].mean())