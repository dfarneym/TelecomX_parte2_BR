## 📈 Análise Preditiva de Evasão de Clientes (Churn) - TelecomX parte 2

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-orange?style=flat&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn)
![Imblearn](https://img.shields.io/badge/Imbalanced--learn-green?style=flat&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=flat&logo=jupyter)

### 📄 Propósito da Análise

Este projeto é a continuação de uma análise de dados sobre a evasão de clientes (Churn) da TelecomX. O objetivo principal desta fase é ir além da análise exploratória e desenvolver **modelos de Machine Learning** capazes de prever quais clientes têm maior probabilidade de cancelar seus serviços. A antecipação da evasão permite à empresa criar estratégias de retenção mais eficazes e direcionadas, protegendo a receita e fortalecendo a base de clientes.

O projeto foi desenvolvido como parte de um curso de Challenge Telecom X: análise de evasão de clientes - Parte 2 da Alura em parceria com a Oracle Next Education.

### 📁 Estrutura e Organização do Projeto

O repositório está organizado da seguinte forma:
````
TelecomX_BR/
├── telecomx_parte2_BR.ipynb   # O notebook principal com o pipeline de Machine Learning.
├── TelecomX_data_tratados.csv  # O conjunto de dados tratado, utilizado como entrada.
└── readme.md                   # Este arquivo README.
````
### 📋 Processo de Preparação dos Dados

A preparação dos dados é a base para a modelagem. O processo foi dividido nas seguintes etapas:

* **Classificação de Variáveis:** As variáveis foram classificadas em numéricas (como `Meses_Permanencia`, `Cobranca_Mensal`, `Cobranca_Total`) e categóricas (como `Tipo_Contrato`, `Metodo_Pagamento`, `Genero`).
* **Tratamento e Codificação:**
    * A variável alvo, `Cancelou`, foi convertida para o formato numérico binário (`1` para "Sim" e `0` para "Não").
    * As colunas categóricas foram transformadas usando **One-Hot Encoding** (`pd.get_dummies`), criando novas colunas binárias para cada categoria.
* **Normalização dos Dados:** As variáveis numéricas foram padronizadas usando **`StandardScaler`**. Essa etapa é crucial para modelos como a Regressão Logística, que são sensíveis à escala das features, garantindo que todas as variáveis contribuam de forma justa para a previsão.
* **Divisão e Balanceamento:**
    * Os dados foram divididos em conjuntos de treino (75%) e teste (25%) usando `train_test_split`.
    * Para lidar com o desequilíbrio de classes (onde a evasão é a classe minoritária), apliquei a técnica de **Oversampling com SMOTE** nos dados de treino.

## 🧠 Justificativas para as escolhas de modelagem

* **Regressão Logística:** Escolhi este modelo como ponto de partida (baseline) por sua simplicidade e interpretabilidade. A análise de seus coeficientes nos ajuda a entender a direção e a força da relação de cada variável com a evasão. A normalização dos dados foi essencial para este modelo.
* **Random Forest:** Optei por este modelo por sua robustez e capacidade de lidar com relações não lineares complexas nos dados. Por ser um modelo baseado em árvores, ele não exige a normalização das features. A sua análise de "importância das variáveis" (Feature Importance) é intuitiva e poderosa.

## 📊 Exemplos de Gráficos e Insights da Análise Exploratória

A análise exploratória (EDA) permitiu identificar padrões e tendências importantes na evasão de clientes. Os gráficos a seguir ilustram os principais achados.

### 1. Proporção de Cancelamento por Tipo de Contrato

Este gráfico de barras mostra claramente a relação entre o tipo de contrato e a taxa de evasão. Clientes com contratos mensais têm uma probabilidade muito maior de cancelar o serviço.

![Proporção de Cancelamento por Tipo de Contrato](Gráficos/Proporção%20de%20Cancelamento%20por%20Tipo%20de%20Contrato.png)

### 2. Relação entre Meses de Permanência e Cancelamento

O histograma abaixo demonstra que a maior parte da evasão ocorre nos primeiros meses de serviço. Conforme a permanência aumenta, a lealdade do cliente se fortalece, e a taxa de cancelamento diminui.

![Relação entre Meses de Permanência e Cancelamento](Gráficos/Relação%20entre%20Meses%20de%20Permanência%20e%20Cancelamento.png)

### 3. Relação entre Cobrança Mensal e Cancelamento

O boxplot a seguir compara a distribuição da cobrança mensal entre clientes que cancelam e os que permanecem. Podemos observar que os clientes com cobranças mais altas têm maior propensão ao churn.

![Relação entre Cobrança Mensal e Cancelamento](Gráficos/Relação%20entre%20Cobrança%20Mensal%20e%20Cancelamento.png)


## 🚀 Como Executar o Notebook

Para rodar a análise e replicar os modelos, siga as instruções:

1.  **Instale as Bibliotecas Necessárias:**
    ```bash
    pip install pandas scikit-learn imbalanced-learn jupyter
    ```
2.  **Clone o Repositório:**
    ```bash
    git clone [https://github.com/dfarneym/TelecomX_parte2_BR](https://github.com/dfarneym/TelecomX_parte2_BR)
    cd TelecomX_parte2_BR
    
    ```
    
3.  **Abra e Execute o Notebook:**
    Abra o arquivo `telecomx_parte2_BR.ipynb` em um ambiente Jupyter e execute as células sequencialmente.
