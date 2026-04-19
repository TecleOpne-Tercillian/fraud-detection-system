
---



```md id="r1"
# 💳 Fraud Detection System (Machine Learning Project)

Projeto de detecção de fraudes em transações financeiras usando Machine Learning, engenharia de features e dashboard interativo com Streamlit.

---

## 🚀 Demonstração

🔗 https://fraud-detection-system-tecleopne.streamlit.app/

---

## 📌 Objetivo

Criar um sistema capaz de:
- Detectar transações suspeitas automaticamente
- Analisar comportamento de usuários
- Gerar score de risco
- Visualizar resultados em dashboard interativo

---

## 🧠 Tecnologias

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly

---

## 📊 Pipeline

1. Coleta de dados
2. Engenharia de features:
   - tempo entre transações
   - média de gasto por usuário
   - desvio de comportamento
   - frequência de uso
3. Treinamento do modelo (Random Forest)
4. Avaliação (precision, recall, f1-score)
5. Dashboard com Streamlit

---

## 🤖 Modelo

Modelo utilizado: **Random Forest Classifier**

Features:
- amount
- lat / long
- merchant_category
- device_id
- time_diff
- user_avg_amount
- amount_vs_avg
- user_tx_count

---

## 📊 Dashboard

O sistema inclui:

- Distribuição de fraudes
- Valores das transações
- Fraude por categoria
- Score de risco
- Análise por usuário
- Filtros interativos

---

## 📁 Estrutura

```

fraud-detection-system/
│
├── app.py
├── src/
│   └── ml_model.py
├── data/
│   ├── transactions.csv
│   └── transactions_with_predictions.csv
├── notebooks/
└── requirements.txt

````

---

## 📦 Como executar

### Instalar dependências
```bash
pip install -r requirements.txt
````

### Treinar modelo

```bash id="x1"
python src/ml_model.py
```

### Rodar dashboard

```bash id="x2"
streamlit run app.py
```

---

## 💡 Aprendizados

* Machine Learning aplicado a fraude
* Feature engineering comportamental
* Modelos supervisionados
* Visualização de dados com Streamlit
* Construção de pipeline completo

---

## 👨‍💻 Autor

Projeto desenvolvido para portfólio de Data Science / Machine Learning.

````

---



---



---



---


````
