```md
💳 Fraud Detection System (Machine Learning Project)

Projeto de detecção de fraudes em transações financeiras utilizando Machine Learning, engenharia de features e dashboard interativo com Streamlit.



🚀 Demonstração

🔗 https://fraud-detection-system-tecleopne.streamlit.app/



📌 Objetivo

Criar um sistema capaz de:
- Detectar transações suspeitas automaticamente
- Analisar comportamento de usuários
- Gerar score de risco para cada transação
- Visualizar resultados em dashboard interativo



🧠 Tecnologias Utilizadas

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- Plotly  



📊 Pipeline do Projeto

1. Coleta de dados de transações
2. Engenharia de features comportamentais:
   - tempo entre transações
   - média de gasto por usuário
   - desvio de comportamento
   - frequência de uso
3. Treinamento de modelo de Machine Learning
4. Avaliação do modelo (precision, recall, f1-score)
5. Dashboard interativo com Streamlit



🤖 Modelo de Machine Learning

Modelo utilizado: Random Forest Classifier

Features utilizadas:
- amount
- lat
- long
- merchant_category
- device_id
- time_diff
- user_avg_amount
- amount_vs_avg
- user_tx_count



📊 Funcionalidades do Dashboard

- Distribuição de fraudes
- Análise de valores das transações
- Taxa de fraude por categoria
- Score de risco
- Análise individual por usuário
- Filtros interativos



📁 Estrutura do Projeto

```

fraud-detection-system/
├── app.py
├── requirements.txt
├── data/
│   ├── transactions.csv
│   └── transactions_with_predictions.csv
├── src/
│   └── ml_model.py
└── notebooks/

````


📦 Como Executar

Instalar dependências
```bash
pip install -r requirements.txt
````

Treinar modelo

```bash
python src/ml_model.py
```

Rodar dashboard

```bash
streamlit run app.py
```



💡 Aprendizados

* Machine Learning aplicado à detecção de fraudes
* Engenharia de features comportamentais
* Modelos supervisionados
* Visualização de dados com Streamlit
* Construção de pipeline completo de dados



👨‍💻 Autor

Projeto desenvolvido para portfólio de Data Science e Machine Learning

```
```
