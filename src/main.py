import pandas as pd

df = pd.read_csv('data/transactions.csv')

print(df.head())

print("\nTransações por usuário:")
print(df.groupby('user_id').size())