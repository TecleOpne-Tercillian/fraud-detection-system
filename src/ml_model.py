import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

# carregar dados
df = pd.read_csv("data/transactions.csv")

# ----------------------------
# 🔧 preparar dados
# ----------------------------

# transformar categorias em números
le_category = LabelEncoder()
df["merchant_category"] = le_category.fit_transform(df["merchant_category"])

le_device = LabelEncoder()
df["device_id"] = le_device.fit_transform(df["device_id"])

# features (o que o modelo vai usar)
features = ["amount", "lat", "long", "merchant_category", "device_id"]

X = df[features]

# ----------------------------
# 🤖 modelo de detecção de anomalias
# ----------------------------
model = IsolationForest(
    contamination=0.05,  # 5% fraude esperada
    random_state=42
)

model.fit(X)

# prever anomalias
df["anomaly_score"] = model.predict(X)

# transformar resultado:
# -1 = fraude (anomaly)
# 1 = normal
df["predicted_fraud"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

# ----------------------------
# 📊 resultados
# ----------------------------
print("\n📊 Resultado do modelo:")
print(df["predicted_fraud"].value_counts())

print("\n📌 Fraudes reais vs previstas:")
print(pd.crosstab(df["is_fraud"], df["predicted_fraud"]))

# salvar resultado
df.to_csv("data/transactions_with_predictions.csv", index=False)

print("\n✅ Modelo executado com sucesso!")