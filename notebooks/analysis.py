import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

# ----------------------------
# 📁 caminho robusto
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "transactions.csv")

# ----------------------------
# 📥 carregar dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)

print("\n📌 Primeiras linhas:")
print(df.head())

print("\n📊 Distribuição de fraude:")
print(df["is_fraud"].value_counts(normalize=True))

# ----------------------------
# 🧠 FEATURE ENGINEERING (EDA)
# ----------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(by=["user_id", "timestamp"])

df["time_diff"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0)
df["user_avg_amount"] = df.groupby("user_id")["amount"].transform("mean")
df["amount_vs_avg"] = df["amount"] / (df["user_avg_amount"] + 1)

# ----------------------------
# 📊 1. FRAUDE VS NÃO FRAUDE
# ----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="is_fraud", palette="Set2")
plt.title("Fraude vs Não Fraude")
plt.show()

# ----------------------------
# 📊 2. DISTRIBUIÇÃO DE VALORES
# ----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["amount"], bins=50, kde=True)
plt.title("Distribuição dos Valores")
plt.show()

# ----------------------------
# 📊 3. VALOR vs FRAUDE
# ----------------------------
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="is_fraud", y="amount")
plt.title("Valor vs Fraude")
plt.show()

print("\n🧠 Insight:")
print("Fraudes tendem a ocorrer em valores mais altos (ver boxplot).")

# ----------------------------
# 📊 4. DESVIO DE COMPORTAMENTO
# ----------------------------
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="is_fraud", y="amount_vs_avg")
plt.title("Desvio de comportamento (amount_vs_avg)")
plt.show()

print("\n🧠 Insight:")
print("Fraudes mostram maior desvio em relação ao comportamento médio do usuário.")

# ----------------------------
# 📊 5. TEMPO ENTRE TRANSAÇÕES
# ----------------------------
plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="is_fraud", y="time_diff")
plt.title("Tempo entre transações vs fraude")
plt.show()

print("\n🧠 Insight:")
print("Fraudes tendem a ocorrer com intervalos menores entre transações.")

# ----------------------------
# 📊 6. FRAUDE POR CATEGORIA
# ----------------------------
plt.figure(figsize=(10,5))
sns.barplot(
    data=df,
    x="merchant_category",
    y="is_fraud",
    estimator="mean"
)
plt.title("Taxa de fraude por categoria")
plt.xticks(rotation=45)
plt.show()

# ----------------------------
# 📊 7. USUÁRIOS MAIS SUSPEITOS
# ----------------------------
user_risk = df.groupby("user_id")["is_fraud"].mean().sort_values(ascending=False)

print("\n🚨 Top usuários mais suspeitos:")
print(user_risk.head(10))

# ----------------------------
# 📊 8. CORRELAÇÃO
# ----------------------------
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Correlação entre variáveis")
plt.show()

print("\n🧠 Insight final:")
print("As variáveis comportamentais (amount_vs_avg, time_diff) são fortes indicadoras de fraude.")