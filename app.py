import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ----------------------------
# ⚙️ CONFIGURAÇÃO
# ----------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

st.title("💳 Fraud Detection System")

# ----------------------------
# 📁 CAMINHO ROBUSTO
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_with_predictions.csv")

# ----------------------------
# 📥 CARREGAR DADOS (COM CACHE)
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ----------------------------
# 🚨 VERIFICAÇÃO DE SEGURANÇA
# ----------------------------
if "fraud_probability" not in df.columns:
    st.error("⚠️ Dataset não contém 'fraud_probability'. Rode o modelo novamente.")
    st.stop()

# ----------------------------
# 🧠 SCORE DE RISCO
# ----------------------------
df["risk_score"] = df["fraud_probability"]

df["risk_level"] = df["risk_score"].apply(
    lambda x: "🔴 Alto" if x > 0.7 else "🟡 Médio" if x > 0.3 else "🟢 Baixo"
)

# ----------------------------
# 🎛️ SIDEBAR
# ----------------------------
st.sidebar.header("🔎 Filtros")

show_fraud = st.sidebar.checkbox("Mostrar apenas fraudes detectadas")

risk_filter = st.sidebar.selectbox(
    "Nível de risco",
    ["Todos", "🔴 Alto", "🟡 Médio", "🟢 Baixo"]
)

max_value = st.sidebar.slider(
    "Valor máximo",
    0,
    int(df["amount"].max()),
    int(df["amount"].max())
)

# ----------------------------
# 🔍 FILTRO
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

col1.metric("Transações", len(df))
col2.metric("Fraudes reais", int(df["is_fraud"].sum()))
col3.metric("Fraudes detectadas", int(df["predicted_fraud"].sum()))
col4.metric("Risco médio", round(df["risk_score"].mean(), 3))

st.divider()

# ----------------------------
# 📊 GRÁFICO FRAUDE
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
st.subheader("💰 Valores das Transações")

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
# 🧠 SCORE
# ----------------------------
st.subheader("🧠 Distribuição de Risco")

fig4 = px.histogram(filtered_df, x="risk_score", nbins=20)
st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# 🚨 RANKING
# ----------------------------
st.subheader("🚨 Top Usuários Suspeitos")

user_risk = (
    df.groupby("user_id")["risk_score"]
    .mean()
    .reset_index()
    .sort_values("risk_score", ascending=False)
)

fig5 = px.bar(user_risk.head(10), x="user_id", y="risk_score")
st.plotly_chart(fig5, use_container_width=True)

# ----------------------------
# 👤 USUÁRIO
# ----------------------------
st.subheader("👤 Análise por Usuário")

user = st.selectbox("Usuário", df["user_id"].unique())

user_df = df[df["user_id"] == user]

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
# 📋 TABELA FINAL (MELHORADA)
# ----------------------------
st.subheader("📋 Transações (ordenadas por risco)")

st.dataframe(
    filtered_df.sort_values("risk_score", ascending=False),
    use_container_width=True
)