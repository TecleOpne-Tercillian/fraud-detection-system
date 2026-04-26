import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ----------------------------
# ⚙️ CONFIG
# ----------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Fraud Detection System")

# ----------------------------
# 📁 DATA
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_with_predictions.csv")

df = pd.read_csv(DATA_PATH)

# ----------------------------
# 🧠 RISK SCORE
# ----------------------------
df["risk_score"] = df["fraud_probability"]

df["risk_level"] = df["risk_score"].apply(
    lambda x: "🔴 Alto" if x > 0.7 else "🟡 Médio" if x > 0.3 else "🟢 Baixo"
)

# ----------------------------
# 🎛️ SIDEBAR FILTROS
# ----------------------------
st.sidebar.header("🔎 Filtros")

risk_filter = st.sidebar.selectbox(
    "Nível de risco",
    ["Todos", "🔴 Alto", "🟡 Médio", "🟢 Baixo"]
)

show_fraud = st.sidebar.checkbox("Somente fraudes detectadas")

category_filter = st.sidebar.multiselect(
    "Categoria",
    options=df["merchant_category"].unique(),
    default=df["merchant_category"].unique()
)

min_value, max_value = st.sidebar.slider(
    "Valor da transação",
    float(df["amount"].min()),
    float(df["amount"].max()),
    (float(df["amount"].min()), float(df["amount"].max()))
)

# ----------------------------
# 🔍 FILTROS
# ----------------------------
filtered_df = df.copy()

filtered_df = filtered_df[
    (filtered_df["amount"] >= min_value) &
    (filtered_df["amount"] <= max_value)
]

filtered_df = filtered_df[
    filtered_df["merchant_category"].isin(category_filter)
]

if show_fraud:
    filtered_df = filtered_df[filtered_df["predicted_fraud"] == 1]

if risk_filter != "Todos":
    filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]

# ----------------------------
# 📊 KPIs
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Transações", len(filtered_df))
col2.metric("Fraudes reais", int(filtered_df["is_fraud"].sum()))
col3.metric("Fraudes detectadas", int(filtered_df["predicted_fraud"].sum()))
col4.metric("Risco médio", round(filtered_df["risk_score"].mean(), 3))

st.divider()

# ----------------------------
# 📈 FRAUDE DISTRIBUIÇÃO
# ----------------------------
st.subheader("📊 Distribuição de Fraudes")

fig1 = px.histogram(
    filtered_df,
    x="is_fraud",
    color="is_fraud"
)

st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# 💰 VALORES
# ----------------------------
st.subheader("💰 Distribuição de Valores")

fig2 = px.box(
    filtered_df,
    y="amount",
    color="is_fraud"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# 📍 CATEGORIAS
# ----------------------------
st.subheader("📍 Fraude por Categoria")

cat_df = filtered_df.groupby("merchant_category")["is_fraud"].mean().reset_index()

fig3 = px.bar(cat_df, x="merchant_category", y="is_fraud")

st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# 🧠 RISCO DISTRIBUIÇÃO
# ----------------------------
st.subheader("🧠 Score de Risco")

fig4 = px.histogram(filtered_df, x="risk_score", nbins=25)

st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# 🚨 TOP USUÁRIOS
# ----------------------------
st.subheader("🚨 Usuários Mais Suspeitos")

user_risk = (
    filtered_df.groupby("user_id")["risk_score"]
    .mean()
    .reset_index()
    .sort_values("risk_score", ascending=False)
)

fig5 = px.bar(user_risk.head(10), x="user_id", y="risk_score")

st.plotly_chart(fig5, use_container_width=True)

# ----------------------------
# 🚨 ALERTAS
# ----------------------------
st.subheader("🚨 Alertas de Alta Fraude")

alerts = filtered_df[filtered_df["risk_score"] > 0.8]

st.dataframe(
    alerts.sort_values("risk_score", ascending=False),
    use_container_width=True
)

# ----------------------------
# 👤 USUÁRIO
# ----------------------------
st.subheader("👤 Análise por Usuário")

user = st.selectbox("Selecionar usuário", filtered_df["user_id"].unique())

user_df = filtered_df[filtered_df["user_id"] == user]

colu1, colu2, colu3 = st.columns(3)

colu1.metric("Transações", len(user_df))
colu2.metric("Fraudes", int(user_df["predicted_fraud"].sum()))
colu3.metric("Risco médio", round(user_df["risk_score"].mean(), 3))

st.dataframe(
    user_df.sort_values("risk_score", ascending=False),
    use_container_width=True
)

st.divider()

# ----------------------------
# 📋 TABELA FINAL
# ----------------------------
st.subheader("📋 Todas as transações")

st.dataframe(
    filtered_df.sort_values("risk_score", ascending=False),
    use_container_width=True
)