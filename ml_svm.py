"""
ml_svm.py — Motor Científico / Padrão-Ouro
Responsabilidade ÚNICA: Treinar o SVM e extrair o gabarito ideal com o dataset
oficial do Scikit-Learn (load_wine). Este módulo NÃO conhece o vinhos_x.csv.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_wine
import streamlit as st


@st.cache_resource
def treinar_especialista_svm():
    """
    Treina o SVM com o gabarito real do Scikit-Learn (Padrão Ouro).
    Retorna: (model, scaler_svm)
    """
    wine_data = load_wine()
    df_train = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

    scaler_svm = StandardScaler()
    X_train_scaled = scaler_svm.fit_transform(df_train)

    # Mapeamento fixo das classes do dataset oficial para os nomes de uva do projeto
    mapa_uvas = {0: 'Barolo', 1: 'Grignolino', 2: 'Barbera'}
    y_train = [mapa_uvas[t] for t in wine_data.target]

    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler_svm


@st.cache_data
def obter_gabarito_scikit():
    """
    Extrai as médias perfeitas (Centróides Padrão-Ouro) da literatura mundial.
    Esses valores guiarão a análise prescritiva (What-If).
    """
    wine_data = load_wine()
    df_train = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    
    # Mapeia os alvos para os nomes das uvas
    mapa_uvas = {0: 'Barolo', 1: 'Grignolino', 2: 'Barbera'}
    df_train['Cluster'] = [mapa_uvas[t] for t in wine_data.target]
    
    # Renomeia as colunas do Scikit-Learn para alinhar com a variável FEATURES do projeto
    dicionario_renomeacao = {
        'alcohol': 'Alcohol',
        'malic_acid': 'Malic',
        'ash': 'Ash',
        'alcalinity_of_ash': 'Alcalinity',
        'magnesium': 'Magnesium',
        'total_phenols': 'Phenols',
        'flavanoids': 'Flavanoids',
        'nonflavanoid_phenols': 'Nonflavanoids',
        'proanthocyanins': 'Proanthocyanins',
        'color_intensity': 'Color',
        'hue': 'Hue',
        'od280/od315_of_diluted_wines': 'Dilution',
        'proline': 'Proline'
    }
    df_train.rename(columns=dicionario_renomeacao, inplace=True)
    
    # Retorna apenas as médias calculadas por tipo de uva
    return df_train.groupby('Cluster').mean()