


```md
# 💳 Fraud Detection System (Machine Learning Project)

Projeto completo de detecção de fraudes em transações financeiras utilizando Machine Learning, engenharia de features e dashboard interativo com Streamlit.

---

## 🚀 Demonstração

🔗 Acesse o sistema online:  
*(adicione aqui o link do Streamlit Cloud depois do deploy)*

---

## 📌 Objetivo

Construir um sistema capaz de:
- Detectar transações suspeitas automaticamente
- Analisar padrões de comportamento de usuários
- Gerar score de risco para cada transação
- Visualizar resultados em dashboard interativo

---

## 🧠 Tecnologias Utilizadas

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly

---

## 📊 Pipeline do Projeto

1. Coleta de dados de transações
2. Engenharia de features comportamentais
   - tempo entre transações
   - média de gasto por usuário
   - desvio de comportamento
   - frequência de uso
3. Treinamento de modelo de Machine Learning
   - Random Forest Classifier
4. Avaliação do modelo
   - precision
   - recall
   - f1-score
5. Dashboard interativo com Streamlit

---

## 🤖 Modelo de Machine Learning

O modelo utilizado foi um **Random Forest Classifier**, treinado para prever a probabilidade de fraude com base em comportamento do usuário.

Features utilizadas:
- amount
- localização (lat/long)
- categoria do comerciante
- device_id
- tempo entre transações
- padrão de gasto do usuário

---

## 📊 Dashboard

O sistema inclui um dashboard interativo com:

- Distribuição de fraudes
- Análise de valores
- Taxa de fraude por categoria
- Score de risco
- Análise individual por usuário
- Filtros interativos

---

## 🧠 Principais Insights

- Transações com valores muito acima da média têm maior risco
- Alta frequência em curto tempo indica comportamento suspeito
- Padrão de gasto do usuário é um forte indicador de fraude

---

## 📁 Estrutura do Projeto

```

fraud-detection-system/
│
├── app.py
├── src/
│   └── ml_model.py
├── data/
│   └── transactions.csv
│   └── transactions_with_predictions.csv
├── notebooks/
└── requirements.txt

````

---

## 📦 Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
````

### 2. Treinar modelo

```bash
python src/ml_model.py
```

### 3. Rodar dashboard

```bash
streamlit run app.py
```

---

## 🌐 Deploy

O projeto pode ser publicado usando **Streamlit Cloud**.

---

## 💡 Aprendizados

* Engenharia de features em dados reais
* Modelos de detecção de fraude
* Visualização de dados com Streamlit
* Construção de pipeline completo de ML

---

## 👨‍💻 Autor

Projeto desenvolvido para fins de aprendizado em Machine Learning e Data Science.

```





```
