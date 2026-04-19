from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 📥 carregar dados
# ----------------------------
df = pd.read_csv("data/transactions.csv")

# ----------------------------
# 🧠 FEATURE ENGINEERING
# ----------------------------

# garantir ordenação temporal por usuário
df = df.sort_values(by=["user_id", "timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# tempo entre transações do mesmo usuário
df["time_diff"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
df["time_diff"] = df["time_diff"].fillna(0)

# média de gasto por usuário
df["user_avg_amount"] = df.groupby("user_id")["amount"].transform("mean")

# desvio do comportamento normal
df["amount_vs_avg"] = df["amount"] / (df["user_avg_amount"] + 1)

# frequência de transações do usuário
df["user_tx_count"] = df.groupby("user_id")["amount"].transform("count")

# ----------------------------
# 🔧 preparar dados categóricos
# ----------------------------
le_category = LabelEncoder()
df["merchant_category"] = le_category.fit_transform(df["merchant_category"])

le_device = LabelEncoder()
df["device_id"] = le_device.fit_transform(df["device_id"])

# ----------------------------
# 📊 features do modelo
# ----------------------------
features = [
    "amount",
    "lat",
    "long",
    "merchant_category",
    "device_id",
    "time_diff",
    "user_avg_amount",
    "amount_vs_avg",
    "user_tx_count"
]

X = df[features]

# ----------------------------
# 🤖 modelo de detecção de anomalias
# ----------------------------
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X)

# prever anomalias
df["anomaly_score"] = model.predict(X)

# converter para fraude (1) e normal (0)
df["predicted_fraud"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

# ----------------------------
# 📊 resultados
# ----------------------------
print("\n📊 Resultado do modelo:")
print(df["predicted_fraud"].value_counts())

print("\n📌 Fraudes reais vs previstas:")
print(pd.crosstab(df["is_fraud"], df["predicted_fraud"]))

# ----------------------------
# 💾 salvar resultado
# ----------------------------
df.to_csv("data/transactions_with_predictions.csv", index=False)

print("\n✅ Modelo executado com sucesso!")

# ----------------------------
# 🧪 avaliação do modelo
# ----------------------------
print("\n📊 Matriz de confusão:")
print(confusion_matrix(df["is_fraud"], df["predicted_fraud"]))

print("\n📈 Relatório de classificação:")
print(classification_report(df["is_fraud"], df["predicted_fraud"]))