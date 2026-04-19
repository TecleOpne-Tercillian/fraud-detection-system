import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# estilo profissional
sns.set(style="darkgrid")

# carregar dataset
df = pd.read_csv("data/transactions.csv")

print("\nPrimeiras linhas:")
print(df.head())

print("\nDistribuição de fraude:")
print(df["is_fraud"].value_counts())

# ----------------------------
# 📊 1. Distribuição de fraude
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="is_fraud", palette="Set2")
plt.title("Fraude vs Não Fraude")
plt.show()

# ----------------------------
# 📊 2. Distribuição de valores
# ----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["amount"], bins=50, kde=True, color="blue")
plt.title("Distribuição dos Valores das Transações")
plt.show()

# ----------------------------
# 📊 3. Fraude por categoria
# ----------------------------
plt.figure(figsize=(10,5))
sns.barplot(
    data=df,
    x="merchant_category",
    y="is_fraud",
    estimator="mean",
    palette="Reds"
)
plt.title("Taxa de Fraude por Categoria")
plt.xticks(rotation=45)
plt.show()

# ----------------------------
# 📊 4. Relação valor vs fraude
# ----------------------------
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="is_fraud", y="amount", palette="Set3")
plt.title("Valor vs Fraude")
plt.show()