import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Fraud Detection Dashboard (ML Project)")

# carregar dados
df = pd.read_csv("data/transactions_with_predictions.csv")

# ----------------------------
# 📊 métricas principais
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total de Transações", len(df))
col2.metric("Fraudes Reais", df["is_fraud"].sum())
col3.metric("Fraudes Previstas", df["predicted_fraud"].sum())

st.divider()

# ----------------------------
# 📊 gráficos melhores (Plotly)
# ----------------------------
st.subheader("📊 Distribuição de Fraudes")

fig1 = px.histogram(df, x="is_fraud", color="is_fraud")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("💰 Valores das Transações")

fig2 = px.box(df, y="amount", color="is_fraud")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📍 Fraude por Categoria")

fig3 = px.bar(
    df.groupby("merchant_category")["is_fraud"].mean().reset_index(),
    x="merchant_category",
    y="is_fraud"
)
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# 📋 tabela interativa
# ----------------------------
st.subheader("📋 Dados")

st.dataframe(df, use_container_width=True)