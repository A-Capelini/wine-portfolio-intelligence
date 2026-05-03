"""
ml_kmeans.py — Motor do Portfólio / Analítico
Responsabilidade ÚNICA: Carregar vinhos_x.csv, aplicar K-Means, enriquecer
o DataFrame com clusters, categorias e Z-Scores. Fornecer funções utilitárias
de similaridade e auditoria matemática do agrupamento.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


FEATURES = [
    'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
    'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
    'Dilution', 'Proline'
]


@st.cache_data
def carregar_portfolio(caminho_csv='vinhos_x.csv'):
    """
    Carrega e enriquece o portfólio de vinhos.
    Usa a coluna 'Vino' do CSV como ID_Vinho (não recria por índice).
    Retorna: (df_enriquecido, scaler_km)
    """
    try:
        df = pd.read_csv(caminho_csv)
    except FileNotFoundError:
        return pd.DataFrame(), None

    # Garante consistência entre o ID exibido e o dado real
    df['ID_Vinho'] = df['Vino'].astype(str)

    # ── Padronização e Z-Scores ─────────────────────────────────────────────
    scaler_km = StandardScaler()
    X_scaled = scaler_km.fit_transform(df[FEATURES])

    colunas_z = [c + '_z' for c in FEATURES]
    df[colunas_z] = X_scaled

    # ── K-Means (K=3) ──────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster_Num'] = kmeans.fit_predict(X_scaled)

    # Mapeamento por média de Prolina: maior Prolina → Barolo
    medias_prolina = df.groupby('Cluster_Num')['Proline'].mean().sort_values()
    mapa_uvas = {
        medias_prolina.index[2]: 'Barolo',
        medias_prolina.index[1]: 'Barbera',
        medias_prolina.index[0]: 'Grignolino'
    }
    df['Cluster'] = df['Cluster_Num'].map(mapa_uvas)

    # ── Categorias de Negócio ───────────────────────────────────────────────
    df['Categoria_Alcool'] = pd.qcut(
        df['Alcohol'], q=3, labels=['Baixo', 'Médio', 'Alto']
    )
    df['Categoria_Cor'] = pd.qcut(
        df['Color'], q=3, labels=['Baixa', 'Média', 'Alta']
    )

    df['Classificacao_Mercado'] = df.apply(_classificar_mercado, axis=1)

    return df, scaler_km


def _classificar_mercado(row):
    """Regras de negócio para classificação de mercado."""
    if row['Proline'] > 1000 and row['Categoria_Alcool'] == 'Alto':
        return 'Premium - Estruturado'
    if row['Categoria_Cor'] == 'Alta' and row['Flavanoids'] > 2.0:
        return 'Tinto Intenso'
    return 'Clássico Padrão'


def encontrar_similar(df, vetor_usuario, scaler_km):
    """
    Encontra o vinho mais próximo no portfólio via distância euclidiana.
    Retorna: (id_vinho_str, similaridade_pct, is_exato)
    """
    X_portfolio = scaler_km.transform(df[FEATURES])
    vetor_scaled = scaler_km.transform(vetor_usuario)
    distancias = np.linalg.norm(X_portfolio - vetor_scaled, axis=1)
    idx = np.argmin(distancias)
    similaridade = max(0.0, 100.0 - (distancias[idx] * 8.0))
    is_exato = distancias[idx] < 1e-4
    return df.iloc[idx]['ID_Vinho'], similaridade, is_exato


@st.cache_data
def obter_metricas_cotovelo():
    """
    Calcula Inércia e Score de Silhueta para K de 1 a 10.
    Retorna: DataFrame com colunas ['K', 'Inercia', 'Silhueta']
    """
    df, _ = carregar_portfolio()
    if df.empty:
        return pd.DataFrame(columns=['K', 'Inercia', 'Silhueta'])

    X_sc = StandardScaler().fit_transform(df[FEATURES])
    inercia, silhueta = [], []

    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc)
        inercia.append(km.inertia_)
        silhueta.append(silhouette_score(X_sc, labels) if k > 1 else 0)

    return pd.DataFrame({'K': range(1, 11), 'Inercia': inercia, 'Silhueta': silhueta})