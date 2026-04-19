import sys
sys.stdout.reconfigure(encoding='utf-8')
import streamlit as st
import pandas as pd

# título
st.title("💳 Fraud Detection Dashboard")

# carregar dados
df = pd.read_csv("data/transactions_with_predictions.csv")

# ----------------------------
# 📊 métricas principais
# ----------------------------
total = len(df)
fraudes_reais = df["is_fraud"].sum()
fraudes_previstas = df["predicted_fraud"].sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total de Transações", total)
col2.metric("Fraudes Reais", fraudes_reais)
col3.metric("Fraudes Previstas", fraudes_previstas)

# ----------------------------
# 📊 gráfico de distribuição
# ----------------------------
st.subheader("Distribuição de Fraudes")

st.bar_chart(df["is_fraud"].value_counts())

# ----------------------------
# 📊 valores das transações
# ----------------------------
st.subheader("Distribuição de Valores")

st.line_chart(df["amount"])

# ----------------------------
# 📊 tabela filtrável
# ----------------------------
st.subheader("Dados")

filtro = st.selectbox(
    "Filtrar por:",
    ["Todos", "Fraudes Reais", "Fraudes Previstas"]
)

if filtro == "Fraudes Reais":
    df = df[df["is_fraud"] == 1]
elif filtro == "Fraudes Previstas":
    df = df[df["predicted_fraud"] == 1]

st.dataframe(df)