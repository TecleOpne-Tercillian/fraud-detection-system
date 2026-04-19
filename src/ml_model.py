import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 📥 carregar dados
# ----------------------------
df = pd.read_csv("data/transactions.csv")

# ----------------------------
# 🧠 FEATURE ENGINEERING
# ----------------------------

df = df.sort_values(by=["user_id", "timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# tempo entre transações
df["time_diff"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
df["time_diff"] = df["time_diff"].fillna(0)

# média de gasto por usuário
df["user_avg_amount"] = df.groupby("user_id")["amount"].transform("mean")

# desvio do comportamento
df["amount_vs_avg"] = df["amount"] / (df["user_avg_amount"] + 1)

# frequência de transações
df["user_tx_count"] = df.groupby("user_id")["amount"].transform("count")

# ----------------------------
# 🔧 preparar dados categóricos
# ----------------------------
le_category = LabelEncoder()
df["merchant_category"] = le_category.fit_transform(df["merchant_category"])

le_device = LabelEncoder()
df["device_id"] = le_device.fit_transform(df["device_id"])

# ----------------------------
# 📊 features e target
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
y = df["is_fraud"]

# ----------------------------
# ✂️ divisão treino/teste
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 🤖 modelo Random Forest
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# previsões
y_pred = model.predict(X_test)

# ----------------------------
# 📊 avaliação do modelo
# ----------------------------
print("\n📊 Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\n📈 Relatório de classificação:")
print(classification_report(y_test, y_pred))

# ----------------------------
# 💾 salvar previsões no dataset completo
# ----------------------------
df["predicted_fraud"] = model.predict(X)

df.to_csv("data/transactions_with_predictions.csv", index=False)

print("\n✅ Modelo Random Forest executado com sucesso!")