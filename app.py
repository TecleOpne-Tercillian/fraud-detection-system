import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle
import numpy as np
Import shap
# ----------------------------
# ⚙️ CONFIG
# ----------------------------
st.set_page_config(
    page_title="MagnaShield Fraud Intelligence",
    layout="wide"
)

st.title("💳 MagnaShield Fraud Intelligence System")
st.caption("TecleOpne System • Fraud Detection Platform")

# ----------------------------
# 📁 DATA
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_with_predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "data", "explainer.pkl")

df = pd.read_csv(DATA_PATH)

model = pickle.load(open(MODEL_PATH, "rb"))
explainer = pickle.load(open(EXPLAINER_PATH, "rb"))

# ----------------------------
# 🧠 RISK SCORE
# ----------------------------
df["risk_score"] = df["fraud_probability"]

df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[-1, 0.3, 0.7, 1.0],
    labels=["🟢 Baixo", "🟡 Médio", "🔴 Alto"]
)

# ----------------------------
# 🚨 ALERT SYSTEM
# ----------------------------
def alert(df_local):
    high = df_local[df_local["risk_score"] > 0.8]

    if len(high) > 0:
        st.error(f"🚨 ALERTA CRÍTICO: {len(high)} transações de alto risco")
    elif df_local["risk_score"].mean() > 0.6:
        st.warning("⚠️ comportamento suspeito detectado")
    else:
        st.success("🟢 sistema normal")

# ----------------------------
# 🎛️ FILTERS
# ----------------------------
st.sidebar.header("Filtros")

risk_filter = st.sidebar.selectbox(
    "Risco",
    ["Todos", "🔴 Alto", "🟡 Médio", "🟢 Baixo"]
)

only_fraud = st.sidebar.checkbox("Somente fraude")

min_v, max_v = st.sidebar.slider(
    "Valor",
    float(df["amount"].min()),
    float(df["amount"].max()),
    (float(df["amount"].min()), float(df["amount"].max()))
)

filtered = df[
    (df["amount"] >= min_v) &
    (df["amount"] <= max_v)
]

if only_fraud:
    filtered = filtered[filtered["predicted_fraud"] == 1]

if risk_filter != "Todos":
    filtered = filtered[filtered["risk_level"] == risk_filter]

# ----------------------------
# KPIs
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Transações", len(filtered))
col2.metric("Fraudes reais", int(filtered["is_fraud"].sum()))
col3.metric("Detectadas", int(filtered["predicted_fraud"].sum()))
col4.metric("Risco médio", round(filtered["risk_score"].mean(), 3))

alert(filtered)

st.divider()

# ----------------------------
# 📊 GRÁFICO PROFISSIONAL
# ----------------------------
st.subheader("📊 Distribuição de Risco")

fig = px.histogram(
    filtered,
    x="risk_score",
    nbins=30,
    color_discrete_sequence=["#7C3AED"]
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 💰 VALORES
# ----------------------------
st.subheader("💰 Valores")

st.plotly_chart(
    px.box(filtered, y="amount", color="is_fraud"),
    use_container_width=True
)

# ----------------------------
# 📍 CATEGORIA
# ----------------------------
st.subheader("📍 Fraude por Categoria")

cat = filtered.groupby("merchant_category")["is_fraud"].mean().reset_index()

st.plotly_chart(
    px.bar(cat, x="merchant_category", y="is_fraud"),
    use_container_width=True
)

# ----------------------------
# 🚨 TOP USERS
# ----------------------------
st.subheader("🚨 Usuários suspeitos")

users = filtered.groupby("user_id")["risk_score"].mean().reset_index()

st.plotly_chart(
    px.bar(users.sort_values("risk_score", ascending=False).head(10),
           x="user_id", y="risk_score"),
    use_container_width=True
)

# ----------------------------
# 👤 USER
# ----------------------------
st.subheader("👤 Usuário")

u = st.selectbox("Selecionar", filtered["user_id"].unique())

udf = filtered[filtered["user_id"] == u]

st.metric("Transações", len(udf))
st.metric("Fraudes", int(udf["predicted_fraud"].sum()))
st.metric("Risco médio", round(udf["risk_score"].mean(), 3))

st.dataframe(udf)

# ----------------------------
# 📋 TABLE
# ----------------------------
st.subheader("📋 Dados")

st.dataframe(filtered.sort_values("risk_score", ascending=False))

# ----------------------------
# 🧠 SHAP SIMPLES E ESTÁVEL
# ----------------------------
st.subheader("🧠 Explicação da IA")

try:
    features = [
        "amount","lat","long","merchant_category","device_id",
        "time_diff","user_avg_amount","amount_vs_avg","user_tx_count"
    ]

    sample = filtered.sample(1)
    X = sample[features]

    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        sv = sv[1][0]
    else:
        sv = sv[0]

    impact = pd.Series(sv, index=features).sort_values()

    st.markdown("### 💳 Motivos da decisão")

    st.write("🔴 fatores de risco:")
    for i in impact.tail(3).index:
        st.write(f"- {i}")

    st.write("🟢 fatores que reduziram risco:")
    for i in impact.head(2).index:
        st.write(f"- {i}")

except:
    st.warning("IA explicável indisponível no momento")