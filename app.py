import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import random
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# ----------------------------
# ⚙️ CONFIG
# ----------------------------
st.set_page_config(page_title="MagnaShield Fraud Intelligence", layout="wide")

st.title("💳 MagnaShield Fraud Intelligence System")
st.caption("Real-time Fraud Detection • TecleOpne System")

# ----------------------------
# 📁 DATA
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_with_predictions.csv")

df = pd.read_csv(DATA_PATH)

# ----------------------------
# 🤖 MODEL (OPCIONAL SHAP SAFE)
# ----------------------------
model_path = os.path.join(BASE_DIR, "data", "model.pkl")

if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
    explainer = shap.TreeExplainer(model)
else:
    model = None
    explainer = None

# ----------------------------
# 🧠 RISK SCORE
# ----------------------------
df["risk_score"] = df["fraud_probability"]

df["risk_level"] = df["risk_score"].apply(
    lambda x: "🔴 Alto" if x > 0.7 else "🟡 Médio" if x > 0.3 else "🟢 Baixo"
)

# ----------------------------
# 🚨 ALERTA
# ----------------------------
def fraud_alert(score):
    if score > 0.8:
        st.error("🚨 ALERTA CRÍTICO: Fraude provável!")
    elif score > 0.6:
        st.warning("⚠️ Transação suspeita")

# ----------------------------
# 🎛️ FILTROS
# ----------------------------
st.sidebar.header("🔎 Filtros")

risk_filter = st.sidebar.selectbox(
    "Nível de risco",
    ["Todos", "🔴 Alto", "🟡 Médio", "🟢 Baixo"]
)

filtered_df = df.copy()

if risk_filter != "Todos":
    filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]

# ----------------------------
# 🚨 ALERT SYSTEM
# ----------------------------
high_risk = filtered_df[filtered_df["risk_score"] > 0.8]

if len(high_risk) > 0:
    st.error(f"🚨 {len(high_risk)} transações críticas")
else:
    st.success("🟢 Sistema normal")

# ----------------------------
# 📊 KPIs
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Transações", len(filtered_df))
col2.metric("Fraudes", int(filtered_df["predicted_fraud"].sum()))
col3.metric("Risco médio", round(filtered_df["risk_score"].mean(), 3))

st.divider()

# ----------------------------
# 📊 RISCO
# ----------------------------
st.subheader("📊 Distribuição de Risco")

fig = px.histogram(
    filtered_df,
    x="risk_score",
    nbins=30,
    color_discrete_sequence=["#7C3AED"]
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 📍 CATEGORIAS
# ----------------------------
st.subheader("📍 Fraude por Categoria")

cat_df = filtered_df.groupby("merchant_category")["is_fraud"].mean().reset_index()

fig2 = px.bar(cat_df, x="merchant_category", y="is_fraud", color="is_fraud")

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# 🚨 USUÁRIOS
# ----------------------------
st.subheader("🚨 Top Usuários")

user_risk = filtered_df.groupby("user_id")["risk_score"].mean().reset_index()

fig3 = px.bar(user_risk.head(10), x="user_id", y="risk_score", color="risk_score")

st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# 🧠 SHAP (SEGURO)
# ----------------------------
st.subheader("🧠 Explicação da Fraude (SHAP)")

if explainer is not None:

    sample = filtered_df.sample(1)

    st.write("Transação analisada:")
    st.dataframe(sample)

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

    X_sample = sample[features]

    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots()

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=features,
        show=False
    )

    st.pyplot(fig)

else:
    st.info("SHAP não disponível (modelo não carregado)")