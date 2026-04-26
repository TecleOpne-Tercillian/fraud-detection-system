import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

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

df = pd.read_csv(DATA_PATH)

# ----------------------------
# 🧠 RISK SCORE
# ----------------------------
df["risk_score"] = df["fraud_probability"]

df["risk_level"] = df["risk_score"].apply(
    lambda x: "🔴 Alto" if x > 0.7 else "🟡 Médio" if x > 0.3 else "🟢 Baixo"
)

# ----------------------------
# 🚨 ALERT FUNCTION
# ----------------------------
def fraud_alert(df_local):
    high = df_local[df_local["risk_score"] > 0.8]

    if len(high) > 0:
        st.error(f"🚨 ALERTA CRÍTICO: {len(high)} transações de alto risco detectadas")
    elif df_local["risk_score"].mean() > 0.6:
        st.warning("⚠️ Comportamento suspeito detectado")
    else:
        st.success("🟢 Sistema operando normalmente")

# ----------------------------
# 🎛️ SIDEBAR FILTERS
# ----------------------------
st.sidebar.header("🔎 Filtros")

risk_filter = st.sidebar.selectbox(
    "Nível de risco",
    ["Todos", "🔴 Alto", "🟡 Médio", "🟢 Baixo"]
)

show_fraud_only = st.sidebar.checkbox("Somente fraudes detectadas")

category_filter = st.sidebar.multiselect(
    "Categoria",
    options=df["merchant_category"].unique(),
    default=df["merchant_category"].unique()
)

min_val, max_val = st.sidebar.slider(
    "Valor da transação",
    float(df["amount"].min()),
    float(df["amount"].max()),
    (float(df["amount"].min()), float(df["amount"].max()))
)

# ----------------------------
# 🔍 FILTERING
# ----------------------------
filtered_df = df.copy()

filtered_df = filtered_df[
    (filtered_df["amount"] >= min_val) &
    (filtered_df["amount"] <= max_val)
]

filtered_df = filtered_df[
    filtered_df["merchant_category"].isin(category_filter)
]

if show_fraud_only:
    filtered_df = filtered_df[filtered_df["predicted_fraud"] == 1]

if risk_filter != "Todos":
    filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]

# ----------------------------
# 🚨 ALERT SYSTEM
# ----------------------------
fraud_alert(filtered_df)

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
# 📊 RISK DISTRIBUTION
# ----------------------------
st.subheader("📊 Distribuição de Risco")

fig1 = px.histogram(
    filtered_df,
    x="risk_score",
    nbins=30,
    color_discrete_sequence=["#7C3AED"]
)

st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# 💰 VALUES
# ----------------------------
st.subheader("💰 Distribuição de Valores")

fig2 = px.box(
    filtered_df,
    y="amount",
    color="is_fraud"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# 📍 CATEGORY FRAUD
# ----------------------------
st.subheader("📍 Fraude por Categoria")

cat_df = filtered_df.groupby("merchant_category")["is_fraud"].mean().reset_index()

fig3 = px.bar(
    cat_df,
    x="merchant_category",
    y="is_fraud",
    color="is_fraud",
    color_continuous_scale="reds"
)

st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# 🚨 TOP USERS
# ----------------------------
st.subheader("🚨 Usuários Mais Suspeitos")

user_risk = filtered_df.groupby("user_id")["risk_score"].mean().reset_index()

fig4 = px.bar(
    user_risk.sort_values("risk_score", ascending=False).head(10),
    x="user_id",
    y="risk_score",
    color="risk_score"
)

st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# 👤 USER ANALYSIS
# ----------------------------
st.subheader("👤 Análise por Usuário")

user = st.selectbox("Selecionar usuário", filtered_df["user_id"].unique())

user_df = filtered_df[filtered_df["user_id"] == user]

col1, col2, col3 = st.columns(3)

col1.metric("Transações", len(user_df))
col2.metric("Fraudes", int(user_df["predicted_fraud"].sum()))
col3.metric("Risco médio", round(user_df["risk_score"].mean(), 3))

st.dataframe(user_df, use_container_width=True)

# ----------------------------
# 📋 TABLE
# ----------------------------
st.subheader("📋 Todas as transações")

st.dataframe(
    filtered_df.sort_values("risk_score", ascending=False),
    use_container_width=True
)

# ----------------------------
# 🧠 SHAP (SAFE MODE)
# ----------------------------
st.subheader("🧠 Explicação da Fraude (IA)")

try:
    import shap
    import matplotlib.pyplot as plt

    model_path = os.path.join(BASE_DIR, "data", "model.pkl")

    if os.path.exists(model_path):

        model = pickle.load(open(model_path, "rb"))
        explainer = shap.TreeExplainer(model)

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

        # ----------------------------
        # pega classe fraude
        # ----------------------------
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        base = float(explainer.expected_value[1]) if isinstance(explainer.expected_value, list) else float(explainer.expected_value)

        # ----------------------------
        # 🎯 transforma em texto estilo banco
        # ----------------------------
        impact = pd.Series(sv, index=features).sort_values()

        top_risk = impact.tail(3)
        top_safe = impact.head(2)

        st.markdown("### 🧠 Explicação estilo banco:")

        st.write("💳 Essa transação foi analisada automaticamente pela IA.")

        st.write("🚨 Principais fatores de risco:")

        for i, v in top_risk.items():
            st.write(f"- {i} contribuiu para AUMENTAR risco")

        st.write("🟢 Fatores que reduziram risco:")

        for i, v in top_safe.items():
            st.write(f"- {i} ajudou a reduzir suspeita")

        st.info(f"📊 Score base do modelo: {round(base, 4)}")

    else:
        st.info("Modelo não encontrado para explicação")

except Exception as e:
    st.warning("🧠 Explicação de IA indisponível no momento (modo seguro ativo)")