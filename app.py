"""
app.py — Orquestrador Central — Wine Portfolio Intelligence
Responsabilidades:
  - Instanciar os dois motores de IA (SVM e K-Means)
  - Gerenciar EXCLUSIVAMENTE o st.session_state
  - Renderizar sidebar, abas, KPIs, gráficos e simulador
  - Capturar eventos de interação e propagar estado
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# ── Importamos a nova função obter_gabarito_scikit ──
from ml_svm import treinar_especialista_svm, obter_gabarito_scikit
from ml_kmeans import carregar_portfolio, encontrar_similar, obter_metricas_cotovelo, FEATURES
from ui_components import load_css, kpi_card

# ── 1. Configuração da Página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine Portfolio Intelligence",
    page_icon="🍷",
    layout="wide"
)
load_css("style.css")

# ── 2. Inicialização dos Motores de IA ───────────────────────────────────────
svm_model, scaler_svm = treinar_especialista_svm()
df, scaler_km = carregar_portfolio()

if df.empty:
    st.error("❌ Erro: Arquivo vinhos_x.csv não encontrado no diretório da aplicação.")
    st.stop()

# ── 3. CONTROLADOR CENTRAL DE ESTADO (Dispatcher) ────────────────────────────
def _change_active_wine(new_id):
    """
    Fonte Única de Verdade (Single Source of Truth).
    Sincroniza TUDO: o ID ativo, o Selectbox visual e os Sliders.
    """
    st.session_state['active_wine_id'] = new_id
    st.session_state['_selectbox_wine_id'] = new_id
    
    if new_id != "Todos":
        st.session_state['modo_simulador'] = 'id_vinho'
        vinho_alvo = df[df['ID_Vinho'] == new_id].iloc[0]
        for f in FEATURES:
            st.session_state[f"slider_{f}"] = float(vinho_alvo[f])
    else:
        st.session_state['modo_simulador'] = 'livre'

# ── 4. INICIALIZAÇÃO DO ESTADO ───────────────────────────────────────────────
if 'active_wine_id' not in st.session_state:
    st.session_state['active_wine_id'] = "Todos"

if '_selectbox_wine_id' not in st.session_state:
    st.session_state['_selectbox_wine_id'] = "Todos"

if 'modo_simulador' not in st.session_state:
    st.session_state['modo_simulador'] = 'livre'

# Processamento de Ação Pendente (Correção para o Scatter Plot)
if 'pending_wine_id' in st.session_state:
    _change_active_wine(st.session_state['pending_wine_id'])
    del st.session_state['pending_wine_id']

# Checkboxes de filtro
for uva in df['Cluster'].unique():
    if f"chk_uva_{uva}" not in st.session_state:
        st.session_state[f"chk_uva_{uva}"] = True

for merc in df['Classificacao_Mercado'].unique():
    if f"chk_merc_{merc}" not in st.session_state:
        st.session_state[f"chk_merc_{merc}"] = True

for alc in ['Baixo', 'Médio', 'Alto']:
    if f"chk_alc_{alc}" not in st.session_state:
        st.session_state[f"chk_alc_{alc}"] = True

# Sliders
for f in FEATURES:
    if f"slider_{f}" not in st.session_state:
        st.session_state[f"slider_{f}"] = float(df[f].mean())


# ── 5. CALLBACKS DE EVENTOS ──────────────────────────────────────────────────
def _on_selectbox_change():
    novo_id = st.session_state['_selectbox_wine_id']
    _change_active_wine(novo_id)

def _reset_todos_filtros():
    _change_active_wine("Todos")
    for uva in df['Cluster'].unique():
        st.session_state[f"chk_uva_{uva}"] = True
    for merc in df['Classificacao_Mercado'].unique():
        st.session_state[f"chk_merc_{merc}"] = True
    for alc in ['Baixo', 'Médio', 'Alto']:
        st.session_state[f"chk_alc_{alc}"] = True
    for f in FEATURES:
        st.session_state[f"slider_{f}"] = float(df[f].mean())


# ── 6. DETECÇÃO DECLARATIVA DE DIVERGÊNCIA (Simulação Livre) ─────────────────
if st.session_state['modo_simulador'] == 'id_vinho' and st.session_state['active_wine_id'] != "Todos":
    v_data = df[df['ID_Vinho'] == st.session_state['active_wine_id']].iloc[0]
    divergiu = any(
        abs(st.session_state[f"slider_{f}"] - float(v_data[f])) > 1e-6
        for f in FEATURES
    )
    if divergiu:
        st.session_state['modo_simulador'] = 'livre'


# ── 7. SIDEBAR ───────────────────────────────────────────────────────────────
id_atual = st.session_state['active_wine_id']

with st.sidebar:
    st.markdown("## 🍷 Vintage Intelligence")

    cor_box = "#4E0707" if id_atual != "Todos" else "#7D8C7C"
    label_foco = f"ID: {id_atual}" if id_atual != "Todos" else "Visão Global"
    st.markdown(f"""
        <div style="background-color: {cor_box}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; border-bottom: 4px solid #D4AF37;">
            <p style="color: #FFFFFF; margin: 0; font-size: 11px; text-transform: uppercase; font-weight: bold;">Vinho em Foco</p>
            <p style="color: #FFFFFF; margin: 5px 0 0 0; font-size: 28px; font-weight: 800; font-family: sans-serif;">{label_foco}</p>
        </div>
    """, unsafe_allow_html=True)

    lista_ids = ["Todos"] + df['ID_Vinho'].tolist()
    
    st.selectbox("Selecionar Vinho por ID:", lista_ids, key='_selectbox_wine_id', on_change=_on_selectbox_change)

    st.markdown("---")
    st.markdown("**Estirpe (Classe Real):**")
    for uva in sorted(df['Cluster'].unique()):
        st.checkbox(uva, key=f"chk_uva_{uva}")

    st.markdown("<br>**Teor Alcoólico:**", unsafe_allow_html=True)
    for alc in ['Baixo', 'Médio', 'Alto']:
        st.checkbox(alc, key=f"chk_alc_{alc}")

    st.markdown("<br>**Classificação de Mercado:**", unsafe_allow_html=True)
    for merc in sorted(df['Classificacao_Mercado'].unique()):
        st.checkbox(merc, key=f"chk_merc_{merc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔄 Resetar Todos os Filtros", on_click=_reset_todos_filtros, use_container_width=True)


# ── 8. LÓGICA DE FILTRAGEM SEGURA ────────────────────────────────────────────
uvas_ativas = [u for u in df['Cluster'].unique() if st.session_state.get(f"chk_uva_{u}", True)]
mercados_ativos = [m for m in df['Classificacao_Mercado'].unique() if st.session_state.get(f"chk_merc_{m}", True)]
alcool_ativo = [a for a in ['Baixo', 'Médio', 'Alto'] if st.session_state.get(f"chk_alc_{a}", True)]

if id_atual != "Todos":
    df_filt = df[df['ID_Vinho'] == id_atual]
else:
    df_filt = df[
        df['Cluster'].isin(uvas_ativas) &
        df['Classificacao_Mercado'].isin(mercados_ativos) &
        df['Categoria_Alcool'].isin(alcool_ativo)
    ]

avg_alc_global = df['Alcohol'].mean()
avg_proline_global = df['Proline'].mean()

MAPA_CORES = {'Barolo': '#2196F3', 'Grignolino': '#ED7D31', 'Barbera': '#0B1E73'}


# ── 9. TÍTULO E ABAS ─────────────────────────────────────────────────────────
st.title("🍷 Wine Portfolio Intelligence")

aba_exec, aba_dados, aba_sim, aba_auditoria = st.tabs([
    "📊 Dashboard", "🗃️ Base de Dados", "🔮 Simulador SVM", "📐 Auditoria K-Means"
])

# ════════════════════════════════════════════════════════════════════════════
# ABA 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with aba_exec:
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(kpi_card("Amostras", len(df_filt), "🍇"), unsafe_allow_html=True)

    with c2:
        val_alc = df_filt['Alcohol'].mean() if not df_filt.empty else 0.0
        delta_alc = val_alc - avg_alc_global if not df_filt.empty else 0.0
        st.markdown(kpi_card("Álcool Médio", f"{val_alc:.2f}%", "🍷", delta_alc), unsafe_allow_html=True)

    with c3:
        val_pro = df_filt['Proline'].mean() if not df_filt.empty else 0.0
        delta_pro = val_pro - avg_proline_global if not df_filt.empty else 0.0
        st.markdown(kpi_card("Média Prolina", f"{val_pro:.0f}", "🧪", delta_pro), unsafe_allow_html=True)

    with c4:
        if not df_filt.empty and df_filt['Cluster'].nunique() == 1:
            estirpe_val = df_filt['Cluster'].iloc[0]
        elif not df_filt.empty:
            estirpe_val = f"{df_filt['Cluster'].nunique()} estirpes"
        else:
            estirpe_val = "N/D"
        st.markdown(kpi_card("Estirpe", estirpe_val, "🏷"), unsafe_allow_html=True)

    with c5:
        if not df_filt.empty and df_filt['Classificacao_Mercado'].nunique() == 1:
            mercado_val = df_filt['Classificacao_Mercado'].iloc[0]
        elif not df_filt.empty:
            mercado_val = "Diversos"
        else:
            mercado_val = "N/D"
        st.markdown(kpi_card("Mercado", mercado_val, "💰"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([6, 4])

    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.subheader("Mapa Químico: Prolina vs Álcool")

        fig_scat = px.scatter(
            df_filt, x='Alcohol', y='Proline', color='Cluster',
            color_discrete_map=MAPA_CORES, hover_data=['ID_Vinho'],
            template='plotly_white', labels={'Alcohol': 'Álcool (%)', 'Proline': 'Prolina'}
        )

        if id_atual != "Todos":
            fig_scat.update_traces(marker=dict(size=22, line=dict(width=3, color='white')))

        event_scat = st.plotly_chart(fig_scat, use_container_width=True, on_select="rerun", key="scatter_principal")

        if event_scat and hasattr(event_scat, 'selection') and event_scat.selection and event_scat.selection.get("points"):
            ponto = event_scat.selection["points"][0]
            customdata = ponto.get("customdata", [])
            if customdata and customdata[0] is not None:
                id_clicado = str(customdata[0])
                if id_clicado != id_atual and id_clicado in df['ID_Vinho'].values:
                    st.session_state['pending_wine_id'] = id_clicado
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.subheader("Distribuição por Estirpe")
        if not df_filt.empty:
            contagem = df_filt['Cluster'].value_counts().reset_index()
            contagem.columns = ['Cluster', 'count']
            fig_bar = px.bar(contagem, y='Cluster', x='count', color='Cluster', orientation='h', color_discrete_map=MAPA_CORES, template='plotly_white')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Nenhum vinho corresponde aos filtros.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader("🕸️ Assinatura Química — Perfil Z-Score")
    if not df_filt.empty:
        colunas_z = [f + '_z' for f in FEATURES]
        labels_radar = FEATURES

        if id_atual != "Todos" and len(df_filt) == 1:
            valores_vinho = df_filt[colunas_z].iloc[0].tolist()
            valores_media = [0.0] * len(FEATURES)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=valores_vinho + [valores_vinho[0]], theta=labels_radar + [labels_radar[0]], fill='toself', name=f'Vinho {id_atual}', line_color='#4E0707'))
            fig_radar.add_trace(go.Scatterpolar(r=valores_media + [valores_media[0]], theta=labels_radar + [labels_radar[0]], fill='toself', name='Média Global', line_color='#7D8C7C', opacity=0.4))
        else:
            fig_radar = go.Figure()
            for estirpe, cor in MAPA_CORES.items():
                df_estirpe = df_filt[df_filt['Cluster'] == estirpe]
                if not df_estirpe.empty:
                    valores = df_estirpe[colunas_z].mean().tolist()
                    fig_radar.add_trace(go.Scatterpolar(r=valores + [valores[0]], theta=labels_radar + [labels_radar[0]], fill='toself', name=estirpe, line_color=cor, opacity=0.7))

        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-3, 3])), template='plotly_white', legend=dict(orientation="h", y=-0.15), height=420)
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Nenhum dado para exibir.")
    st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# ABA 2 E 4 — DADOS E AUDITORIA
# ════════════════════════════════════════════════════════════════════════════
with aba_dados:
    st.subheader("🗃️ Portfólio Filtrado")
    colunas_exibir = ['ID_Vinho', 'Cluster', 'Classificacao_Mercado', 'Categoria_Alcool'] + FEATURES
    st.dataframe(df_filt[colunas_exibir], hide_index=True, use_container_width=True)

with aba_auditoria:
    st.subheader("📐 Auditoria do Agrupamento (K-Means)")
    res_aud = obter_metricas_cotovelo()
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        fig_cot = px.line(res_aud, x='K', y='Inercia', markers=True, template='plotly_white', title="Cotovelo")
        fig_cot.add_vline(x=3, line_dash="dash", line_color="#4E0707")
        st.plotly_chart(fig_cot, use_container_width=True)
    with c_a2:
        fig_sil = px.bar(res_aud[res_aud['K'] > 1], x='K', y='Silhueta', template='plotly_white', color='Silhueta', color_continuous_scale=['#F0EAD6', '#4E0707'], title="Silhueta")
        fig_sil.add_vline(x=3, line_dash="dash", line_color="#4E0707")
        st.plotly_chart(fig_sil, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# ABA 3 — SIMULADOR SVM (COM MOTOR PRESCRITIVO)
# ════════════════════════════════════════════════════════════════════════════
with aba_sim:
    st.subheader("🔮 Simulador de Inteligência Artificial (SVM)")

    if st.session_state['modo_simulador'] == 'id_vinho' and id_atual != "Todos":
        st.success(f"✅ Sliders carregados com os dados reais do Vinho **ID {id_atual}**. Mova qualquer slider para entrar em Simulação Livre.")
    else:
        st.info("🎛️ Modo Simulação Livre — ajuste os sliders para análise preditiva.")

    cols = st.columns(3)
    for i, f in enumerate(FEATURES):
        with cols[i % 3]:
            st.slider(
                label=f, min_value=float(df[f].min()), max_value=float(df[f].max()),
                value=float(st.session_state[f"slider_{f}"]), key=f"slider_{f}"
            )

    st.markdown("---")

    vetor_entrada = np.array([[st.session_state[f"slider_{f}"] for f in FEATURES]])
    vetor_scaled = scaler_svm.transform(vetor_entrada)

    pred_estirpe = svm_model.predict(vetor_scaled)[0]
    prob_confianca = np.max(svm_model.predict_proba(vetor_scaled)) * 100
    cor_pred = MAPA_CORES.get(pred_estirpe, '#333333')

    col_pred, col_conf = st.columns(2)
    with col_pred:
        st.markdown(f"### Estirpe Prevista: <span style='color:{cor_pred}; font-size:28px;'>{pred_estirpe}</span>", unsafe_allow_html=True)
    with col_conf:
        st.metric("Grau de Confiança do Modelo", f"{prob_confianca:.1f}%")

    id_sugerido, sim_val, is_exato = encontrar_similar(df, vetor_entrada, scaler_km)

    if is_exato and st.session_state['modo_simulador'] == 'id_vinho':
        st.success(f"📍 Analisando dados reais do Vinho ID **{id_atual}** — correspondência exata no catálogo.")
    else:
        st.info(f"🔍 Simulação Livre — Vinho mais similar no catálogo: **ID {id_sugerido}** ({sim_val:.1f}% de similaridade)")
        st.button(f"📂 Carregar Vinho ID {id_sugerido} no Painel Inteiro", key="btn_carregar_sugerido", on_click=_change_active_wine, args=(id_sugerido,))

    # ── NOVO: PAINEL PRESCRITIVO (WHAT-IF) ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Motor Prescritivo (What-If)")
    st.markdown("Descubra os ajustes químicos necessários para que este vinho atinja a perfeição do **Padrão-Ouro** mundial.")

    # Selectbox sugerindo automaticamente a uva que o modelo previu
    estirpe_alvo = st.selectbox(
        "Qual estirpe clássica você deseja atingir?",
        ['Barolo', 'Barbera', 'Grignolino'],
        index=['Barolo', 'Barbera', 'Grignolino'].index(pred_estirpe)
    )

    gabarito = obter_gabarito_scikit()
    valores_ideais = gabarito.loc[estirpe_alvo]

    # Cálculos das Diferenças (Deltas)
    deltas = []
    for f in FEATURES:
        v_atual = st.session_state[f"slider_{f}"]
        v_ideal = valores_ideais[f]
        diferenca = v_ideal - v_atual
        
        # Evitar divisão por zero e usar diferença percentual para alinhar escalas
        pct_diff = abs(diferenca) / v_ideal if v_ideal != 0 else 0

        deltas.append({
            'feature': f,
            'atual': v_atual,
            'ideal': v_ideal,
            'diferenca': diferenca,
            'pct_diff': pct_diff
        })

    # Pega os 3 atributos com a maior discrepância percentual
    deltas.sort(key=lambda x: x['pct_diff'], reverse=True)
    top_3 = deltas[:3]

    st.markdown(f"**Top 3 Intervenções Recomendadas para um(a) {estirpe_alvo}:**")

    cols_presc = st.columns(3)
    for i, item in enumerate(top_3):
        with cols_presc[i]:
            if item['pct_diff'] <= 0.05:  # Se a diferença for menor que 5%, está ótimo
                st.success(f"✅ **{item['feature']}**\n\nJá está no padrão ideal! ({item['atual']:.2f})")
            else:
                acao = "Aumentar" if item['diferenca'] > 0 else "Reduzir"
                cor_seta = "🔺" if item['diferenca'] > 0 else "🔻"
                
                # HTML Customizado para os alertas prescritivos
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; border-left: 5px solid {'#dc3545' if item['diferenca'] < 0 else '#28a745'}; padding: 15px; border-radius: 5px;">
                        <p style="margin: 0; font-size: 16px; color: #4E0707;"><strong>{cor_seta} {acao} {item['feature']}</strong></p>
                        <p style="margin: 10px 0 0 0; font-size: 13px; color: #666;">Valor Atual: {item['atual']:.2f}</p>
                        <p style="margin: 5px 0 0 0; font-size: 14px; font-weight: bold; color: #333;">Alvo Ideal: {item['ideal']:.2f} <span style="color: {'#dc3545' if item['diferenca'] < 0 else '#28a745'};">({item['diferenca']:+.2f})</span></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

st.caption("Wine Portfolio Intelligence")