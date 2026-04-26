import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import shap

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
MODEL_PATH = os.path.join(BASE_DIR, "..", "data", "model.pkl")
SHAP_PATH = os.path.join(BASE_DIR, "..", "data", "shap_explainer.pkl")
SHAP_SAMPLE_PATH = os.path.join(BASE_DIR, "..", "data", "shap_sample.csv")

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
# 🔧 categóricos
# ----------------------------
le_category = LabelEncoder()
df["merchant_category"] = le_category.fit_transform(df["merchant_category"])

le_device = LabelEncoder()
df["device_id"] = le_device.fit_transform(df["device_id"])

# ----------------------------
# 📊 FEATURES
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
# ✂️ split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 🤖 MODELOS
# ----------------------------
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model_rf.fit(X_train, y_train)

# ----------------------------
# 📊 avaliação
# ----------------------------
def evaluate(name, y_true, y_pred):
    print(f"\n📊 {name}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

y_pred_rf = model_rf.predict(X_test)
evaluate("Random Forest", y_test, y_pred_rf)

# ----------------------------
# 🎯 threshold tuning
# ----------------------------
y_proba = model_rf.predict_proba(X_test)[:, 1]

threshold = 0.3
y_pred_custom = (y_proba > threshold).astype(int)

evaluate(f"RF Threshold {threshold}", y_test, y_pred_custom)

# ----------------------------
# 💾 salvar previsões
# ----------------------------
df["fraud_probability"] = model_rf.predict_proba(X)[:, 1]
df["predicted_fraud"] = (df["fraud_probability"] > threshold).astype(int)

df.to_csv(OUTPUT_PATH, index=False)

# ----------------------------
# 💾 salvar modelo
# ----------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
pickle.dump(model_rf, open(MODEL_PATH, "wb"))

# ----------------------------
# 🧠 SHAP (EXPLICABILIDADE)
# ----------------------------
# ----------------------------
# 🧠 SHAP (VERSÃO SEGURA PARA STREAMLIT)
# ----------------------------
explainer = shap.TreeExplainer(model_rf)

X_sample = X_test.sample(200, random_state=42)

# shap values
shap_values = explainer.shap_values(X_sample)

# IMPORTANTE: não salvar objeto inteiro (evita crash em deploy)
import json

shap_summary = {
    "features": features,
    "mean_abs_shap": abs(shap_values[1]).mean(axis=0).tolist() if isinstance(shap_values, list) else abs(shap_values).mean(axis=0).tolist()
}

SHAP_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "shap_summary.json")

with open(SHAP_JSON_PATH, "w") as f:
    json.dump(shap_summary, f)

# salvar amostra (ok)
X_sample.to_csv(SHAP_SAMPLE_PATH, index=False)

print("\n✅ Modelo + SHAP seguro salvo com sucesso")