import pandas as pd
import matplotlib.pyplot as plt

# carregar dataset
df = pd.read_csv("data/transactions.csv")

print("\nPrimeiras linhas:")
print(df.head())

print("\nPrimeiras linhas:")
print(df.info())

print("\nDistribuição de fraude:")
print(df["is_fraud"].value_counts())

# ----------------------------
# 1. Valor médio por fraude
# ----------------------------
print("\nMédia de valores:")
print(df.groupby("is_fraud")["amount"].mean())

# ----------------------------
# 2. Quantidade de fraudes por usuário
# ----------------------------
fraud_by_user = df.groupby("user_id")["is_fraud"].sum().sort_values(ascending=False)

print("\nTop usuários com fraude:")
print(fraud_by_user.head())

# ----------------------------
# 3. Gráfico de fraude vs não fraude
# ----------------------------
df["is_fraud"].value_counts().plot(
    kind="bar",
    title="Fraude vs Não Fraude",
    color=["green", "red"]
)
plt.show()

# ----------------------------
# 4. Distribuição de valores
# ----------------------------
df["amount"].hist(bins=50)
plt.title("Distribuição de valores")
plt.show()

# ----------------------------
# 5. Fraude por categoria
# ----------------------------
df.groupby("merchant_category")["is_fraud"].mean().plot(
    kind="bar",
    title="Taxa de fraude por categoria"
)
plt.show()