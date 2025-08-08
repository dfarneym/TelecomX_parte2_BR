## 📈 Projeto de Machine Learning: Análise Preditiva de Evasão de Clientes (Churn) - TelecomX (Parte 2)

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

## 📊 Exemplos de Gráficos e Insights (Análise Exploratória)

Antes da modelagem, a análise exploratória de dados (EDA) revelou padrões cruciais:

* **Gráfico 1 - Distribuição Geral de Cancelamentos**: Vemos que a base de dados é desbalanceada. Cerca de 26,6% dos clientes cancelaram, enquanto 73,4% permaneceram. Isso nos mostra que a maioria dos clientes é leal, mas há uma parcela significativa que precisa de atenção:

![Distribuição Geral de Cancelamentos](Gráficos/Distribuição%20Geral%20de%20Cancelamentos.png)

* **Gráfico 2 - Proporção de Cancelamento por Tipo de Contrato**:

![Distribuição Geral de Cancelamentos](Gráficos/Proporção%20de%20Cancelamento%20por%20Tipo%20de%20Contrato.png)

- Contrato Mensal: A barra vermelha (Cancelou = Sim) é enorme, mostrando que mais de 42% dos clientes com este tipo de contrato cancelam o serviço.
- Contrato de Um Ano: A taxa de cancelamento cai drasticamente, ficando em torno de 11%.
- Contrato de Dois Anos: É o mais seguro para a empresa, com uma taxa de cancelamento baixíssima, de apenas ~3%.

* **Gráfico 3 - Proporção de Cancelamento por Tipo de Contrato**:

![Relação entre Cobrança Mensal e Cancelamento](Gráficos/Relação%20entre%20Cobrança%20Mensal%20e%20Cancelamento.png)

- **Cobrança Mensal:** A análise visual com o `boxplot` mostra que os clientes que cancelam tendem a ter uma mediana de cobrança mensal mais alta, em torno de **R$80**. Isso contrasta com os clientes que permanecem, cuja mediana de cobrança é de aproximadamente **R$65**. É importante notar que clientes com cobranças mensais muito baixas (abaixo de R$30) raramente cancelam.

* **Gráfico 4 - Relação entre Meses de Permanência e Cancelamento**:

![Relação entre Meses de Permanência e Cancelamento](Gráficos/Relação%20entre%20Meses%20de%20Permanência%20e%20Cancelamento.png)

- Meses de Permanência (Gráfico 4): O histograma revela outro padrão crucial. A maioria dos cancelamentos ocorre nos primeiros meses do serviço. Clientes que superam a marca de um ano tendem a se tornar muito mais leais. A lealdade aumenta com o tempo de permanência.

* **Tempo de Permanência:** A maior parte dos cancelamentos acontece nos primeiros meses de serviço, sugerindo que o período inicial é o mais crítico para a retenção.

* **Gráfico 5 -  Proporção de Cancelamento por Fatores Demográficos**:

![ Proporção de Cancelamento por Fatores Demográficos ](Gráficos/Proporção%20de%20Cancelamento%20por%20Fatores%20Demográficos.png)

- Gênero: A taxa de cancelamento é praticamente idêntica para os gêneros Feminino (26,9%) e Masculino (26,2%). Isso nos mostra que o gênero não é um fator relevante para prever o cancelamento.

- Cliente Idoso: Aqui a diferença é gritante. Clientes idosos (Sim) cancelam numa proporção muito maior (41,7%) do que os clientes não idosos (Não), que têm uma taxa de apenas 23,6%. Ser idoso é um forte indicador de risco de cancelamento.

- Possui Cônjuge: Clientes sem cônjuge (Não) têm uma taxa de cancelamento maior (33,0%) em comparação com aqueles que possuem um parceiro (Sim), cuja taxa é de 19,7%.

- Possui Dependentes: O padrão é semelhante ao do cônjuge. Clientes sem dependentes (Não) cancelam muito mais (31,3%) do que aqueles que possuem dependentes (Sim), que têm uma taxa de apenas 15,5%.

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
