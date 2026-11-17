# 💳 Detecção de Fraudes em Cartão de Crédito com Machine Learning e SHAP

Projeto de ciência de dados e aprendizado de máquina desenvolvido para identificar transações fraudulentas em um cenário de desbalanceamento extremo, percorrendo todo o ciclo de vida: exploração, engenharia de atributos, modelagem comparativa (**Logistic Regression**, **Random Forest** e **XGBoost**), ajuste de limiar de decisão, otimização de hiperparâmetros e explicabilidade com **SHAP**.

---

## 🎯 O Problema e o Paradoxo da Acurácia

A base utilizada contém transações reais realizadas por titulares de cartões de crédito europeus, onde:
- **`Time`:** segundos decorridos desde a primeira transação da base.
- **`V1` a `V28`:** componentes principais resultantes de transformação **PCA** para anonimização dos dados sensíveis.
- **`Amount`:** valor financeiro da transação.
- **`Class`:** variável alvo (`0` = transação legítima, `1` = fraude).

### Por que o desbalanceamento muda a avaliação?
Apenas **0,173%** das transações são fraudes contra **99,83%** de operações legítimas (uma razão de desbalanceamento de **578:1**). Nesse contexto, um modelo ingênuo que classifique todas as transações como normais alcança **99,83% de acurácia**, mas falha em 100% do objetivo de negócio.

Por isso, a acurácia foi descartada como critério de decisão e o foco foi direcionado para a **classe minoritária (`Class = 1`)** e para a **Curva Precision-Recall**:
- **Recall (Sensibilidade):** proporção de fraudes reais que o modelo conseguiu capturar (crucial para evitar prejuízo financeiro direto).
- **Precision (Precisão):** proporção de alertas gerados que eram de fato fraudes (importante para não bloquear cartões de clientes legítimos).
- **F1-score:** média harmônica entre Precisão e Recall.

---

## 🛠️ Preparação dos Dados

1. **Carregamento Direto via URL:** Os dados são lidos diretamente do repositório remoto via `pandas.read_csv()`, mantendo o repositório leve e dentro dos limites do GitHub.
2. **Transformação Logarítmica (`amount_log`):** Aplicação de `np.log1p(df['Amount'])` para reduzir a assimetria (*skewness*) dos valores monetários.
3. **Padronização (`Amount_scaled`):** Escalonamento da variável `Amount` com `StandardScaler` para garantir convergência adequada em modelos lineares.
4. **Divisão Estratificada (`stratify=y`):** Separação de **70% treino** e **30% teste** preservando exatamente a proporção original de 0,173% de fraudes em ambos os conjuntos.

---

## 📊 Comparação entre os Modelos (Classe 1 — Fraude)

| Modelo / Configuração | Precision | Recall | F1-score | Destaque |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** (Baseline) | `0.83` | `0.64` | `0.72` | Ponto de partida simples (`AUC ROC = 0.924`). |
| **Logistic Regression** (`threshold = 0.3`) | `0.80` | `0.67` | `0.73` | Redução do limiar capturou mais fraudes reais. |
| **Random Forest** (`class_weight='balanced'`) | `0.84` | `0.76` | `0.79` | Salto expressivo no Recall ao penalizar erros na classe rara. |
| **XGBoost** (`scale_pos_weight=10`) | **`0.94`** | **`0.78`** | **`0.85`** | Melhor equilíbrio geral entre precisão e detecção de fraudes. |

---

## 🎚️ Limiar de Decisão e Explicabilidade (SHAP)

- **Ajuste de Limiar (`0.5` → `0.3`):** Na Regressão Logística, reduzir o ponto de corte de probabilidade de `0.50` para `0.30` elevou o **Recall de 64% para 67%**, demonstrando como calibrar a sensibilidade do modelo conforme o apetite a risco da operação financeira.
- **Otimização (`GridSearchCV`):** Busca em grade otimizada diretamente pela métrica `scoring="recall"` com validação cruzada (`cv=3`), identificando `{'max_depth': 7, 'n_estimators': 100}` como melhor combinação estrutural para o XGBoost.
- **Explicabilidade com SHAP:** Além da importância global das variáveis do XGBoost, a aplicação do `shap.Explainer` permitiu auditar o impacto marginal de cada atributo nas predições, evidenciando que componentes como **`V14`**, **`V4`**, **`V12`**, **`V10`** e **`V17`** exercem o maior peso para caracterizar padrões fraudulentos.

---

## 💡 Evoluções Aplicadas em Relação ao Escopo Base

- **Diagnóstico Automatizado de Desbalanceamento:** Cálculo explícito do *Imbalance Ratio* (`578:1`) logo na etapa exploratória para fundamentar a escolha das métricas.
- **Análise Comparativa de Curvas Precision-Recall:** Sobreposição das curvas de **Logistic Regression**, **Random Forest** e **XGBoost** com cálculo de *Average Precision (AP)* no mesmo gráfico para comparação visual direta.
- **Avaliação Pós-GridSearch:** Instanciação e validação do `best_estimator_` no conjunto de teste para comprovar a performance fora da amostra.

---

## 🗂️ Estrutura do Repositório

```text
deteccao-fraudes-cartao-credito/
├── Teste_de_fraude_CC.ipynb    # Notebook completo com todas as saídas e gráficos salvos
└── README.md                   # Documentação técnica e análise dos resultados
```

---

## 🚀 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/Cmarchi2003/deteccao-fraudes-cartao-credito.git
   cd deteccao-fraudes-cartao-credito
   ```
2. Instale as dependências principais:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib shap jupyter
   ```
3. Abra o arquivo `Teste_de_fraude_CC.ipynb` no **Google Colab** ou **Jupyter Notebook** e execute todas as células.
