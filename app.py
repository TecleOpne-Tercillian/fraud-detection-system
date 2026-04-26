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
model = pickle.load(open("data/model.pkl", "rb"))
explainer = pickle.load(open("data/shap_explainer.pkl", "rb"))

# ----------------------------
# 🧠 RISK SCORE
# ----------------------------
df["risk_score"] = df["fraud_probability"]

df["risk_level"] = df["risk_score"].apply(
    lambda x: "🔴 Alto" if x > 0.7 else "🟡 Médio" if x > 0.3 else "🟢 Baixo"
)

# ----------------------------
# 🚨 ALERTA STYLE
# ----------------------------
def fraud_alert(score):
    if score > 0.8:
        st.error("🚨 ALERTA CRÍTICO: Transação com alto risco de fraude detectada!")
    elif score > 0.6:
        st.warning("⚠️ Transação suspeita detectada")

# ----------------------------
# 🎛️ SIDEBAR FILTROS
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

if show_fraud_only:
    filtered_df = filtered_df[filtered_df["predicted_fraud"] == 1]

if risk_filter != "Todos":
    filtered_df = filtered_df[filtered_df["risk_level"] == risk_filter]

high_risk = filtered_df[filtered_df["risk_score"] > 0.8]

if len(high_risk) > 0:
    st.error(f"🚨 ALERTA: {len(high_risk)} transações críticas detectadas")
elif filtered_df["risk_score"].mean() > 0.6:
    st.warning("⚠️ Atenção: comportamento acima do padrão detectado")
else:
    st.success("🟢 Sistema operando dentro do padrão normal")

# ----------------------------
# 📊 KPIs
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Transações", len(filtered_df))

col2.metric(
    "Fraudes reais",
    int(filtered_df["is_fraud"].sum())
)

col3.metric(
    "Fraudes detectadas",
    int(filtered_df["predicted_fraud"].sum())
)

col4.metric(
    "Risco médio",
    round(filtered_df["risk_score"].mean(), 3)
)

st.divider()

# ----------------------------
# 📊 GRÁFICO PROFISSIONAL DE RISCO (NOVO)
# ----------------------------
st.subheader("📊 Distribuição Profissional de Risco")

risk = filtered_df["risk_score"].dropna()

hist = np.histogram(risk, bins=30)

x = hist[1][:-1]
y = hist[0]

fig1 = go.Figure()

# distribuição
fig1.add_trace(go.Bar(
    x=x,
    y=y,
    name="Distribuição",
    marker_color="#636EFA",
    opacity=0.7
))

# curva tendência
fig1.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="lines",
    name="Tendência de risco",
    line=dict(color="red", width=3)
))

# threshold fraude
fig1.add_vline(
    x=0.8,
    line_width=3,
    line_dash="dash",
    line_color="red",
    annotation_text="Limite de Fraude (0.8)"
)

fig1.update_layout(
    template="plotly_dark",
    xaxis_title="Risk Score",
    yaxis_title="Quantidade de Transações",
    bargap=0.1
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

fig3 = px.bar(
    cat_df,
    x="merchant_category",
    y="is_fraud",
    color="is_fraud",
    color_continuous_scale="reds"
)

st.plotly_chart(fig3, use_container_width=True)

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

fig4 = px.bar(user_risk.head(10), x="user_id", y="risk_score", color="risk_score")

st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# 🚨 ALERTAS
# ----------------------------
st.subheader("🚨 Alertas de Alta Risco")

alerts = filtered_df[filtered_df["risk_score"] > 0.8]

if len(alerts) > 0:
    fraud_alert(alerts["risk_score"].max())

st.dataframe(
    alerts.sort_values("risk_score", ascending=False),
    use_container_width=True
)

# ----------------------------
# 🔴 SIMULAÇÃO TEMPO REAL
# ----------------------------
st.subheader("🔴 Simulação de Transações em Tempo Real")

def generate_fake_transaction():
    return {
        "user_id": random.randint(1, 300),
        "amount": round(random.uniform(10, 5000), 2),
        "merchant_category": random.choice(df["merchant_category"].unique()),
        "risk_score": random.random()
    }

if st.button("▶ Iniciar simulação"):
    placeholder = st.empty()

    for i in range(10):
        tx = generate_fake_transaction()

        temp_df = pd.DataFrame([tx])

        placeholder.dataframe(temp_df)

        fraud_alert(tx["risk_score"])

        time.sleep(1)

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

# ----------------------------
# 📋 TABELA FINAL
# ----------------------------
st.subheader("📋 Todas as transações")

st.dataframe(
    filtered_df.sort_values("risk_score", ascending=False),
    use_container_width=True
)

st.subheader("🧠 Explicabilidade da Fraude (SHAP)")

# selecionar uma transação
sample = filtered_df.sample(1)

st.write("Transação analisada:")
st.dataframe(sample)

# preparar dados
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

# explicação SHAP
shap_values = explainer.shap_values(X_sample)

st.text("Impacto das variáveis na decisão:")

st.pyplot(shap.summary_plot(shap_values, X_sample, show=False))