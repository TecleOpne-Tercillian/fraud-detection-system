import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------
# ⚙️ CONFIGURAÇÃO
# ----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

st.title("💳 Fraud Detection System")

# ----------------------------
# 📥 CARREGAR DADOS
# ----------------------------
df = pd.read_csv("data/transactions_with_predictions.csv")

# ----------------------------
# 🧠 SCORE DE RISCO (MELHORADO)
# ----------------------------
df["risk_score"] = df["fraud_probability"]

# classificação de risco
df["risk_level"] = df["risk_score"].apply(
    lambda x: "🔴 Alto" if x > 0.7 else "🟡 Médio" if x > 0.3 else "🟢 Baixo"
)

# ----------------------------
# 🎛️ SIDEBAR (FILTROS)
# ----------------------------
st.sidebar.header("🔎 Filtros")

show_fraud = st.sidebar.checkbox("Mostrar apenas fraudes detectadas")

risk_filter = st.sidebar.selectbox(
    "Nível de risco",
    ["Todos", "🔴 Alto", "🟡 Médio", "🟢 Baixo"]
)

max_value = st.sidebar.slider(
    "Valor máximo da transação",
    0,
    int(df["amount"].max()),
    int(df["amount"].max())
)

# ----------------------------
# 🔍 FILTRAGEM
# ----------------------------
filtered_df = df[df["amount"] <= max_value]

if show_fraud:
    filtered_df = filtered_df[filtered_df["predicted_fraud"] == 1]

if risk_filter != "Todos":
    filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]

# ----------------------------
# 📊 MÉTRICAS
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total de Transações", len(df))
col2.metric("Fraudes Reais", int(df["is_fraud"].sum()))
col3.metric("Fraudes Detectadas", int(df["predicted_fraud"].sum()))
col4.metric("Risco Médio", round(df["risk_score"].mean(), 3))

st.divider()

# ----------------------------
# 📊 GRÁFICO 1 - FRAUDES
# ----------------------------
st.subheader("📊 Distribuição de Fraudes")

fig1 = px.histogram(
    filtered_df,
    x="is_fraud",
    color="is_fraud",
    title="Fraudes vs Não Fraudes"
)

st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# 💰 GRÁFICO 2 - VALORES
# ----------------------------
st.subheader("💰 Distribuição de Valores")

fig2 = px.box(
    filtered_df,
    y="amount",
    color="is_fraud",
    title="Valores das Transações"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# 📍 GRÁFICO 3 - CATEGORIAS
# ----------------------------
st.subheader("📍 Taxa de Fraude por Categoria")

cat_df = filtered_df.groupby("merchant_category")["is_fraud"].mean().reset_index()

fig3 = px.bar(
    cat_df,
    x="merchant_category",
    y="is_fraud",
    title="Fraude por Categoria"
)

st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# 🧠 GRÁFICO 4 - SCORE DE RISCO
# ----------------------------
st.subheader("🧠 Score de Risco")

fig4 = px.histogram(
    filtered_df,
    x="risk_score",
    nbins=20,
    title="Distribuição do Score de Risco"
)

st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# 🔥 RANKING DE USUÁRIOS SUSPEITOS (NOVO DIFERENCIAL)
# ----------------------------
st.subheader("🚨 Top Usuários Mais Suspeitos")

user_risk = df.groupby("user_id")["risk_score"].mean().reset_index()
user_risk = user_risk.sort_values("risk_score", ascending=False)

fig5 = px.bar(
    user_risk.head(10),
    x="user_id",
    y="risk_score",
    title="Top 10 Usuários Mais Suspeitos"
)

st.plotly_chart(fig5, use_container_width=True)

# ----------------------------
# 👤 ANÁLISE POR USUÁRIO
# ----------------------------
st.subheader("👤 Análise por Usuário")

user = st.selectbox("Selecione um usuário", df["user_id"].unique())

user_df = df[df["user_id"] == user]

colu1, colu2, colu3 = st.columns(3)

colu1.metric("Total", len(user_df))
colu2.metric("Fraudes detectadas", int(user_df["predicted_fraud"].sum()))
colu3.metric("Score médio", round(user_df["risk_score"].mean(), 3))

st.write("📋 Transações do usuário:")
st.dataframe(user_df, use_container_width=True)

st.divider()

# ----------------------------
# 📋 TABELA FINAL
# ----------------------------
st.subheader("📋 Dados Filtrados")

st.dataframe(filtered_df, use_container_width=True)