# 🍷 Wine Portfolio Intelligence

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)

🌐 **Acesso ao Dashboard em Produção:** [anderson-wine-portfolio-intelligence.streamlit.app](https://anderson-wine-portfolio-intelligence.streamlit.app)

Um dashboard analítico e prescritivo avançado projetado para a exploração de dados do setor vitivinícola. O sistema combina análise descritiva de inventário físico com um simulador preditivo "Padrão-Ouro", atuando como um assistente de suporte à decisão enológica.

---

## 🏛️ Contexto Acadêmico
Este sistema compõe o escopo do curso de Ciência de Dados da **Fatec Cotia**, sob a orientação do Prof. **Rômulo Francisco de Souza Maia**.

**Autoria e Desenvolvimento:** 
* Anderson Capelini Andrade

---

## 🧠 Arquitetura de Dois Motores (Dual-Engine AI)

O grande diferencial deste projeto é a separação estrita de responsabilidades (Separation of Concerns) entre os dados do mundo real (estoque) e o gabarito científico. A aplicação é orquestrada por dois motores independentes:

1. **Motor Analítico de Inventário (K-Means):** Roda de forma "cega" sobre o arquivo físico `vinhos_x.csv`. Aplica clusterização para organizar o estoque atual, gerando os Z-Scores da assinatura química, agrupando as estirpes por perfis de Prolina e calculando similaridade via Distância Euclidiana.
2. **Motor Prescritivo Padrão-Ouro (SVM):** Atua como o "Juiz Biológico". É treinado exclusivamente com o dataset oficial `load_wine` da literatura mundial. Ele não possui contato com o estoque físico, garantindo que as simulações e o sistema *What-If* sejam sempre comparados à perfeição química, e não aos vieses do inventário local.

---

## ⚙️ Estrutura do Projeto

O código está modularizado para garantir escalabilidade e manutenção segura:

* `app.py`: **Orquestrador Central**. Gerencia a interface, transições seguras do `st.session_state` e os callbacks de reatividade (Fonte Única de Verdade).
* `ml_kmeans.py`: **Módulo Analítico**. Responsável por toda a clusterização, cálculos de métricas (Cotovelo/Silhueta) e categorização das regras de negócio do portfólio.
* `ml_svm.py`: **Módulo Científico**. Responsável pelo treinamento do modelo classificador e extração dos centróides perfeitos (Gabarito Scikit-Learn) para orientar o motor prescritivo.
* `ui_components.py` & `style.css`: **Frontend**. Isolam a injeção de estilos da marca e a renderização de componentes HTML (como os KPIs).

---

## 🚀 Funcionalidades Principais

* **📊 Dashboard Executivo:** Visão reativa do portfólio com KPIs comparativos, distribuição demográfica das estirpes e um gráfico de Radar de Perfil Z-Score. O Scatter Plot é interativo: clicar num vinho muda todo o contexto da aplicação.
* **🔮 Simulador Preditivo & What-If:** O usuário pode ajustar 13 características químicas. O SVM prevê a uva, o K-Means encontra o vinho físico mais similar no estoque, e o "Motor Prescritivo" sugere os 3 principais ajustes químicos necessários (ex: *Aumentar Prolina em +150*) para atingir o padrão-ouro da literatura.
* **📐 Auditoria K-Means:** Validação matemática transparente exibindo o Método do Cotovelo (Inércia) e o Score de Silhueta para atestar a qualidade do agrupamento em K=3.

---

## 💻 Instalação e Execução (Ambiente Local - Recomendado Conda)

Recomenda-se a utilização do gerenciador de pacotes **Conda** (via Miniconda ou Anaconda) para garantir que as dependências binárias (especialmente do `scikit-learn`) sejam gerenciadas de forma nativa e isolada.

### Opção 1: Restaurar Ambiente Inteiro (Backup Conda)
Rode este comando se você deseja recriar o ambiente exatamente como ele foi exportado pelo desenvolvedor:

```bash
# Clone o repositório
git clone [https://github.com/A-Capelini/wine-portfolio-intelligence.git](https://github.com/A-Capelini/wine-portfolio-intelligence.git)
cd wine-portfolio-intelligence

# Cria o ambiente a partir do arquivo YAML completo
conda env create -f environment.yml

# Ative o ambiente
conda activate Dash_Vinho_Streamlit

# Execute a aplicação
streamlit run app.py