import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# 📁 CAMINHOS ROBUSTOS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "transactions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "transactions_with_predictions.csv")

# ----------------------------
# 📥 carregar dados
# ----------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------
# 🧠 FEATURE ENGINEERING
# ----------------------------
df = df.sort_values(by=["user_id", "timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["time_diff"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0)
df["user_avg_amount"] = df.groupby("user_id")["amount"].transform("mean")
df["amount_vs_avg"] = df["amount"] / (df["user_avg_amount"] + 1)
df["user_tx_count"] = df.groupby("user_id")["amount"].transform("count")

# ----------------------------
# 🔧 preparar categóricos
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
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 🧠 função de avaliação
# ----------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 {name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

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

# avaliação padrão
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
feat_imp = pd.Series(
    model_rf.feature_importances_,
    index=features
).sort_values(ascending=False)

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
plt.tight_layout()
plt.show()

# ----------------------------
# 💾 salvar previsões
# ----------------------------
df["fraud_probability"] = model_rf.predict_proba(X)[:, 1]
df["predicted_fraud"] = (df["fraud_probability"] > threshold).astype(int)

df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Modelo treinado e salvo com sucesso!")