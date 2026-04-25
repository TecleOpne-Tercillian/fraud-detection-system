import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
# ✂️ divisão treino/teste (STRATIFIED)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 🧠 FUNÇÃO DE AVALIAÇÃO
# ----------------------------
def evaluate_model(name, y_test, y_pred):
    print(f"\n📊 {name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ----------------------------
# 🤖 BASELINE - Logistic Regression
# ----------------------------
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)
evaluate_model("Logistic Regression", y_test, y_pred_lr)

# ----------------------------
# 🌲 Random Forest MELHORADO
# ----------------------------
model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ----------------------------
# 🎯 THRESHOLD TUNING
# ----------------------------
y_proba = model_rf.predict_proba(X_test)[:, 1]

threshold = 0.3
y_pred_custom = (y_proba > threshold).astype(int)

evaluate_model(f"Random Forest (Threshold {threshold})", y_test, y_pred_custom)

# ----------------------------
# 🔍 FEATURE IMPORTANCE
# ----------------------------
importances = model_rf.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print("\n🔍 Feature Importance:")
print(feat_imp)

plt.figure(figsize=(10, 5))
feat_imp.plot(kind="bar")
plt.title("Importância das Variáveis")
plt.tight_layout()
plt.show()

# ----------------------------
# 📊 MATRIZ DE CONFUSÃO VISUAL
# ----------------------------
cm = confusion_matrix(y_test, y_pred_custom)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title("Confusion Matrix (Threshold Ajustado)")
plt.show()

# ----------------------------
# 💾 salvar previsões no dataset completo
# ----------------------------
df["fraud_probability"] = model_rf.predict_proba(X)[:, 1]
df["predicted_fraud"] = (df["fraud_probability"] > threshold).astype(int)

df.to_csv("data/transactions_with_predictions.csv", index=False)

print("\n✅ Modelo treinado, avaliado e salvo com sucesso!")