"""
ui_components.py — Componentes de Interface Reutilizáveis
Responsabilidade ÚNICA: Renderização de HTML/CSS para KPIs e carregamento
de estilos. Sem lógica de negócio ou estado.
"""

import streamlit as st


def load_css(file_name):
    """Injeta o arquivo CSS no contexto do Streamlit."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS é cosmético; não deve travar a aplicação


def kpi_card(title, value, icon, delta=None):
    """
    Gera HTML de um cartão KPI em UMA ÚNICA LINHA — sem quebras.
    Múltiplas linhas fazem o Streamlit dividir o conteúdo em blocos
    separados, aplicando unsafe_allow_html apenas ao primeiro deles
    e exibindo o HTML bruto nos demais.
    """
    delta_html = ""
    if delta is not None:
        color = "#28a745" if delta >= 0 else "#dc3545"
        sign = "+" if delta >= 0 else ""
        delta_html = (
            f'<p style="margin:5px 0 0 0;font-size:14px;color:{color};font-weight:bold;">'
            f'{sign}{delta:.2f} vs M&#233;dia Geral</p>'
        )

    # HTML compactado em uma única linha — crítico para unsafe_allow_html funcionar
    return (
        f'<div style="background-color:#FFFFFF;padding:20px;border-radius:12px;'
        f'border-bottom:5px solid #4E0707;box-shadow:0 4px 15px rgba(0,0,0,0.05);'
        f'text-align:center;min-height:160px;display:flex;flex-direction:column;'
        f'justify-content:center;align-items:center;">'
        f'<p style="color:#7D8C7C;font-size:12px;text-transform:uppercase;'
        f'margin-bottom:5px;font-weight:800;letter-spacing:1px;">{title}</p>'
        f'<h2 style="color:#4E0707;margin:0;font-size:32px;'
        f'font-family:\'Playfair Display\',serif;">{value}</h2>'
        f'{delta_html}'
        f'<span style="font-size:24px;margin-top:8px;">{icon}</span>'
        f'</div>'
    )