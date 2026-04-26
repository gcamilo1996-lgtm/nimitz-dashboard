"""
dashboard.py — SPX Nimitz | PnL Attribution Monitor
Bloomberg-style Dash app. Deploy on Railway.
"""

import os, sys, traceback
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from atribuicao_pnl import (
    obter_retornos_mercado, obter_cotas_fundos,
    calcular_retorno_fundo, atribuir_pnl, resumo_betas,
    FATORES, FUNDOS, DATA_INICIO, DATA_FIM,
)

# ── Paleta Bloomberg ─────────────────────────────────────────────────────────
BG        = "#0b0e14"
SURFACE   = "#131720"
SURFACE2  = "#1a1f2e"
BORDER    = "#252d3d"
ORANGE    = "#e8830c"
GREEN     = "#00c27a"
RED       = "#e8404a"
TEXT      = "#d4dbe8"
MUTED     = "#5a6478"
MONO      = "'IBM Plex Mono', 'Courier New', monospace"
SANS      = "'IBM Plex Sans', 'Arial', sans-serif"

FATOR_COLORS = {
    "Ibovespa": "#4a9eff",
    "S&P 500":  "#00c27a",
    "USD/BRL":  "#e8830c",
    "Ouro":     "#f5cc00",
    "Petróleo": "#9b6dff",
    "Alpha":    "#5a6478",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def fmt_brl(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "+" if v >= 0 else "-"
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"{sign}R${abs_v/1e9:.{decimals}f}B"
    if abs_v >= 1e6:
        return f"{sign}R${abs_v/1e6:.{decimals}f}M"
    if abs_v >= 1e3:
        return f"{sign}R${abs_v/1e3:.0f}K"
    return f"{sign}R${abs_v:.0f}"

def fmt_pct(v, decimals=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"

def color_val(v):
    try:
        if v is None or np.isnan(v):
            return MUTED
    except (TypeError, ValueError):
        return MUTED
    return GREEN if v >= 0 else RED

# ── App ──────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="NIMITZ | PnL Monitor",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server   # expõe o Flask server para o Railway

# ── CSS inline (sem arquivo externo) ─────────────────────────────────────────

GLOBAL_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    background: {BG};
    color: {TEXT};
    font-family: {SANS};
    font-size: 13px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}

.topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    height: 48px;
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
    position: sticky;
    top: 0;
    z-index: 100;
}}

.topbar-left {{
    display: flex;
    align-items: center;
    gap: 24px;
}}

.logo {{
    font-family: {MONO};
    font-size: 14px;
    font-weight: 600;
    color: {ORANGE};
    letter-spacing: 3px;
    text-transform: uppercase;
}}

.divider-v {{
    width: 1px;
    height: 20px;
    background: {BORDER};
}}

.fund-name {{
    font-family: {SANS};
    font-size: 12px;
    font-weight: 500;
    color: {TEXT};
    letter-spacing: .5px;
    text-transform: uppercase;
}}

.topbar-right {{
    display: flex;
    align-items: center;
    gap: 16px;
}}

.last-update {{
    font-family: {MONO};
    font-size: 11px;
    color: {MUTED};
}}

.last-update span {{
    color: {ORANGE};
    margin-left: 6px;
}}

.btn-refresh {{
    font-family: {MONO};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {BG};
    background: {ORANGE};
    border: none;
    padding: 6px 16px;
    cursor: pointer;
    transition: opacity .15s;
}}
.btn-refresh:hover {{ opacity: .85; }}
.btn-refresh:disabled {{ opacity: .4; cursor: wait; }}

.main {{
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}}

.metrics-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
}}

.metric-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
}}

.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: {ORANGE};
    opacity: .5;
}}

.metric-label {{
    font-family: {SANS};
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 8px;
}}

.metric-value {{
    font-family: {MONO};
    font-size: 20px;
    font-weight: 600;
    line-height: 1.1;
}}

.metric-sub {{
    font-family: {MONO};
    font-size: 10px;
    color: {MUTED};
    margin-top: 4px;
}}

.charts-row {{
    display: grid;
    grid-template-columns: 1.6fr 1fr;
    gap: 8px;
}}

.chart-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    padding: 14px;
}}

.chart-card-full {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    padding: 14px;
}}

.card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid {BORDER};
}}

.card-title {{
    font-family: {SANS};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {MUTED};
}}

.legend-row {{
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-family: {MONO};
    font-size: 10px;
    color: {MUTED};
}}

.legend-dot {{
    width: 8px;
    height: 8px;
    border-radius: 1px;
    flex-shrink: 0;
}}

.betas-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
}}

.beta-card {{
    background: {SURFACE2};
    border: 1px solid {BORDER};
    padding: 12px;
}}

.beta-fator {{
    font-family: {SANS};
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: {MUTED};
    margin-bottom: 6px;
}}

.beta-val {{
    font-family: {MONO};
    font-size: 22px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
}}

.beta-stats {{
    font-family: {MONO};
    font-size: 10px;
    color: {MUTED};
    line-height: 1.6;
}}

.sig-star {{
    color: {ORANGE};
    margin-left: 2px;
}}

.error-msg {{
    font-family: {MONO};
    font-size: 12px;
    color: {RED};
    background: {SURFACE};
    border: 1px solid {RED};
    padding: 16px 20px;
    margin: 20px;
}}

.loading-overlay {{
    position: fixed;
    inset: 0;
    background: rgba(11,14,20,.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 999;
    font-family: {MONO};
    font-size: 13px;
    color: {ORANGE};
    letter-spacing: 2px;
}}
"""

# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div([
    html.Style(GLOBAL_CSS),

    # Top bar
    html.Div([
        html.Div([
            html.Div("BLOOMBERG", className="logo"),
            html.Div(className="divider-v"),
            html.Div("SPX NIMITZ FEEDER · PnL ATTRIBUTION", className="fund-name"),
        ], className="topbar-left"),

        html.Div([
            html.Div([
                "ÚLTIMA CVM:",
                html.Span("—", id="last-cvm-date"),
            ], className="last-update"),
            html.Button("↻ ATUALIZAR", id="btn-refresh", className="btn-refresh", n_clicks=0),
        ], className="topbar-right"),
    ], className="topbar"),

    # Loading
    dcc.Loading(
        id="loading",
        type="circle",
        color=ORANGE,
        children=html.Div(id="page-content", className="main"),
    ),

], style={"minHeight": "100vh", "background": BG})


# ── Callback principal ────────────────────────────────────────────────────────

@app.callback(
    Output("page-content", "children"),
    Output("last-cvm-date", "children"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=False,
)
def refresh_dashboard(n_clicks):
    try:
        ret_mercado  = obter_retornos_mercado(FATORES, DATA_INICIO, DATA_FIM)
        df_cotas     = obter_cotas_fundos(FUNDOS)
        df_ret       = calcular_retorno_fundo(df_cotas)
        df_atrib     = atribuir_pnl(df_ret, ret_mercado)
        df_betas     = resumo_betas(df_ret, ret_mercado)

        df = df_atrib.dropna(subset=["retorno_fundo_%"]).copy()
        df["data"] = pd.to_datetime(df["data"])
        df = df.sort_values("data")

        ultima_data = df["data"].max().strftime("%d/%m/%Y")

        # Acumulado
        df["pnl_acum"] = df["pnl_brl"].cumsum()

        fatores = list(FATORES.keys())
        pnl_total     = df["pnl_brl"].sum()
        retorno_mtd   = (1 + df["retorno_fundo_%"] / 100).prod() - 1
        r2_val        = df_betas["r2"].iloc[0] if not df_betas.empty else None
        beta_ibov     = df_betas.loc[df_betas["fator"] == "Ibovespa", "beta"].values
        beta_ibov     = beta_ibov[0] if len(beta_ibov) else None

        # ── Métricas ──────────────────────────────────────────────────────
        pnl_hoje      = df.iloc[-1]["pnl_brl"]
        ret_hoje      = df.iloc[-1]["retorno_fundo_%"]

        metrics = html.Div([
            _metric("PnL MTD", fmt_brl(pnl_total), color_val(pnl_total),
                    f"{len(df)} pregões"),
            _metric("PnL HOJE", fmt_brl(pnl_hoje), color_val(pnl_hoje),
                    df.iloc[-1]["data"].strftime("%d/%m")),
            _metric("RETORNO MTD", fmt_pct(retorno_mtd * 100, 2), color_val(retorno_mtd),
                    f"retorno acum."),
            _metric("R² MODELO", f"{r2_val*100:.2f}%" if r2_val else "—", ORANGE,
                    "explicado pelos fatores"),
            _metric("β IBOVESPA", f"{beta_ibov:.3f}" if beta_ibov else "—",
                    color_val(beta_ibov), "coef. estimado"),
        ], className="metrics-row")

        # ── Gráfico 1: Barras empilhadas de PnL ──────────────────────────
        fig_bar = go.Figure()
        for fator in fatores + ["Alpha"]:
            key = "pnl_" + fator.lower().replace(" ", "_").replace("&", "").replace("/", "") + "_brl"
            fig_bar.add_trace(go.Bar(
                name=fator,
                x=df["data"].dt.strftime("%d/%m"),
                y=df[key] / 1e6,
                marker_color=FATOR_COLORS[fator],
                marker_line_width=0,
                hovertemplate=f"<b>{fator}</b><br>%{{x}}<br>R$%{{y:.2f}}M<extra></extra>",
            ))
        fig_bar.add_trace(go.Scatter(
            name="PnL Total",
            x=df["data"].dt.strftime("%d/%m"),
            y=df["pnl_brl"] / 1e6,
            mode="markers",
            marker=dict(symbol="diamond", size=7, color=TEXT,
                        line=dict(width=1, color=SURFACE)),
            hovertemplate="<b>Total</b><br>%{x}<br>R$%{y:.2f}M<extra></extra>",
        ))

        _apply_layout(fig_bar, barmode="relative", height=260)
        fig_bar.update_yaxes(title_text="R$ MM", tickformat=".1f",
                             ticksuffix="M", zeroline=True,
                             zerolinecolor=BORDER, zerolinewidth=1)

        # ── Gráfico 2: PnL acumulado ──────────────────────────────────────
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df["data"].dt.strftime("%d/%m"),
            y=df["pnl_acum"] / 1e6,
            mode="lines+markers",
            line=dict(color=ORANGE, width=2),
            marker=dict(size=4, color=ORANGE),
            fill="tozeroy",
            fillcolor=f"rgba(232,131,12,0.07)",
            hovertemplate="<b>Acumulado</b><br>%{x}<br>R$%{y:.2f}M<extra></extra>",
        ))
        _apply_layout(fig_cum, barmode=None, height=260)
        fig_cum.update_yaxes(tickformat=".1f", ticksuffix="M",
                             zeroline=True, zerolinecolor=BORDER, zerolinewidth=1)

        legend_items = html.Div([
            html.Div([
                html.Div(style={"background": FATOR_COLORS[f], "width": "8px",
                                "height": "8px", "borderRadius": "1px"}),
                html.Span(f),
            ], className="legend-item")
            for f in list(FATORES.keys()) + ["Alpha"]
        ], className="legend-row")

        charts = html.Div([
            html.Div([
                html.Div([
                    html.Span("PnL DIÁRIO POR FATOR (R$ MM)", className="card-title"),
                    legend_items,
                ], className="card-header"),
                dcc.Graph(figure=fig_bar, config={"displayModeBar": False},
                          style={"height": "260px"}),
            ], className="chart-card"),

            html.Div([
                html.Div([
                    html.Span("PnL ACUMULADO MTD", className="card-title"),
                ], className="card-header"),
                dcc.Graph(figure=fig_cum, config={"displayModeBar": False},
                          style={"height": "260px"}),
            ], className="chart-card"),
        ], className="charts-row")

        # ── Gráfico 3: Retorno diário do fundo ───────────────────────────
        colors_ret = [GREEN if v >= 0 else RED for v in df["retorno_fundo_%"]]
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Bar(
            x=df["data"].dt.strftime("%d/%m"),
            y=df["retorno_fundo_%"],
            marker_color=colors_ret,
            marker_line_width=0,
            hovertemplate="<b>Retorno</b><br>%{x}<br>%{y:.3f}%<extra></extra>",
        ))
        _apply_layout(fig_ret, barmode=None, height=140)
        fig_ret.update_yaxes(ticksuffix="%", zeroline=True,
                             zerolinecolor=BORDER, zerolinewidth=1)

        chart_ret = html.Div([
            html.Div([
                html.Span("RETORNO DIÁRIO DA COTA (%)", className="card-title"),
            ], className="card-header"),
            dcc.Graph(figure=fig_ret, config={"displayModeBar": False},
                      style={"height": "140px"}),
        ], className="chart-card-full")

        # ── Betas ─────────────────────────────────────────────────────────
        beta_cards = []
        for fator in fatores:
            row = df_betas[df_betas["fator"] == fator]
            if row.empty:
                continue
            b      = row.iloc[0]
            pv     = b["p_value"]
            stars  = "★★★" if pv < 0.001 else "★★" if pv < 0.01 else "★" if pv < 0.05 else "n.s."
            sig_color = ORANGE if pv < 0.05 else MUTED
            beta_cards.append(
                html.Div([
                    html.Div(fator, className="beta-fator"),
                    html.Div(f"{b['beta']:+.4f}",
                             className="beta-val",
                             style={"color": color_val(b["beta"])}),
                    html.Div([
                        f"t = {b['t_stat']:+.2f}",
                        html.Br(),
                        f"p = {pv:.4f} ",
                        html.Span(stars, style={"color": sig_color}),
                    ], className="beta-stats"),
                ], className="beta-card")
            )

        r2_pct = df_betas["r2"].iloc[0] * 100 if not df_betas.empty else 0

        betas_section = html.Div([
            html.Div([
                html.Span("BETAS OLS — PERÍODO COMPLETO", className="card-title"),
                html.Span(f"R² = {r2_pct:.2f}%",
                          style={"fontFamily": MONO, "fontSize": "11px", "color": ORANGE}),
            ], className="card-header"),
            html.Div(beta_cards, className="betas-grid"),
        ], className="chart-card-full")

        return [metrics, charts, chart_ret, betas_section], ultima_data

    except Exception as e:
        traceback.print_exc()
        error_block = html.Div(
            f"ERRO AO CARREGAR DADOS: {str(e)}",
            className="error-msg"
        )
        return [error_block], "ERRO"


# ── Helpers de layout Plotly ──────────────────────────────────────────────────

def _metric(label, value, value_color, sub):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value", style={"color": value_color}),
        html.Div(sub, className="metric-sub"),
    ], className="metric-card")


def _apply_layout(fig, barmode=None, height=260):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, size=10, color=MUTED),
        margin=dict(l=40, r=8, t=8, b=30),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=SURFACE2,
            bordercolor=BORDER,
            font=dict(family=MONO, size=11, color=TEXT),
        ),
        xaxis=dict(
            showgrid=False,
            showline=False,
            tickcolor=MUTED,
            ticklen=4,
            tickfont=dict(size=10),
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=BORDER,
            gridwidth=0.5,
            showline=False,
            tickcolor=MUTED,
            ticklen=4,
            tickfont=dict(size=10),
        ),
    )
    if barmode:
        fig.update_layout(barmode=barmode)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
