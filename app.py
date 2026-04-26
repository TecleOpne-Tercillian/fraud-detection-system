import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Fraud Detection", layout="wide")

# ----------------------------
# 🎨 ESTILO CUSTOM
# ----------------------------
st.markdown("""
<style>
.metric-card {
    background-color: #111;
    padding: 15px;
    border-radius: 10px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Fraud Detection Dashboard")

# ----------------------------
# 📁 PATH
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_with_predictions.csv")

df = pd.read_csv(DATA_PATH)

# ----------------------------
# 🧠 RISCO
# ----------------------------
df["risk_score"] = df["fraud_probability"]

# ----------------------------
# 📊 KPIs
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💸 Transações", len(df))
col2.metric("🚨 Fraudes", int(df["predicted_fraud"].sum()))
col3.metric("📊 Risco médio", round(df["risk_score"].mean(), 2))

st.divider()

# ----------------------------
# 📈 GRÁFICO PRINCIPAL
# ----------------------------
st.subheader("📈 Distribuição de Risco")

fig = px.histogram(
    df,
    x="risk_score",
    nbins=30,
    color="predicted_fraud",
    color_discrete_map={0: "#00cc96", 1: "#EF553B"}
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 🚨 ALERTAS (DIFERENCIAL)
# ----------------------------
st.subheader("🚨 Transações de Alto Risco")

alerts = df[df["risk_score"] > 0.8].sort_values("risk_score", ascending=False)

st.dataframe(alerts.head(20), use_container_width=True)

# ----------------------------
# 👤 USUÁRIOS
# ----------------------------
st.subheader("👤 Usuários Suspeitos")

user_risk = df.groupby("user_id")["risk_score"].mean().reset_index()
user_risk = user_risk.sort_values("risk_score", ascending=False)

fig2 = px.bar(user_risk.head(10), x="user_id", y="risk_score")

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# 📋 TABELA FINAL
# ----------------------------
st.subheader("📋 Todas as transações")

st.dataframe(
    df.sort_values("risk_score", ascending=False),
    use_container_width=True
)