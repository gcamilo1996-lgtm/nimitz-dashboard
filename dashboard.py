"""
dashboard.py — PnL Attribution Monitor v2
Bloomberg-style Dash app. Deploy on Railway.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import yfinance as yf
from flask import Response
import dash
from dash import dcc, html, Input, Output, callback_context, ALL
import plotly.graph_objects as go

from atribuicao_pnl import (
    obter_retornos_mercado, obter_cotas_fundos, obter_cdi,
    calcular_retorno_fundo, atribuir_pnl, resumo_betas,
    FATORES, FUNDOS, DATA_INICIO, DATA_FIM, RIDGE_ALPHA, JANELA_ROLLING,
)

# ── Paleta Bloomberg ──────────────────────────────────────────────────────────
BG       = "#0b0e14"
SURFACE  = "#131720"
SURFACE2 = "#1a1f2e"
BORDER   = "#252d3d"
ORANGE   = "#e8830c"
GREEN    = "#00c27a"
RED      = "#e8404a"
YELLOW   = "#f5cc00"
TEXT     = "#d4dbe8"
MUTED    = "#5a6478"
MONO     = "'IBM Plex Mono', 'Courier New', monospace"
SANS     = "'IBM Plex Sans', Arial, sans-serif"

FATOR_COLORS = {
    "Ibovespa":     "#4a9eff",
    "S&P 500":      "#00c27a",
    "Nasdaq":       "#00e5a0",
    "Euro Stoxx":   "#3db8f5",
    "Hang Seng":    "#1a6fa8",
    "Treasury 10Y": "#f5cc00",
    "Treasury 3M":  "#f5e070",
    "Treasury 30Y": "#c9a800",
    "IMA-B 5+":     "#ffe066",
    "IRF-M":        "#b8a000",
    "USD/BRL":      "#e8830c",
    "EUR/BRL":      "#f0a050",
    "JPY/BRL":      "#c46000",
    "DXY":          "#ff6b35",
    "Ouro":         "#ffd700",
    "Petróleo":     "#9b6dff",
    "Prata":        "#c0c0c0",
    "Cobre":        "#b87333",
    "Açúcar":       "#ff9eb5",
    "VIX":          "#e8404a",
    "Bitcoin":      "#f7931a",
    "Alpha":        "#5a6478",
}

FUND_COLORS = [
    "#4a9eff", "#00c27a", "#f5cc00", "#e8830c", "#9b6dff",
    "#e8404a", "#00e5a0", "#f0a050", "#ffd700", "#c0c0c0", "#ff9eb5",
]

_cache: dict = {"df_atrib": None, "df_betas_ytd": None, "df_betas_mtd": None, "cdi": None}


# ── CSS ───────────────────────────────────────────────────────────────────────
def build_css():
    return (
        "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
        "&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');\n"
        "*, *::before, *::after { box-sizing:border-box; margin:0; padding:0; }\n"
        "body { background:" + BG + "; color:" + TEXT + "; font-family:" + SANS + ";"
        " font-size:13px; line-height:1.5; -webkit-font-smoothing:antialiased; }\n"
        "::-webkit-scrollbar { width:4px; height:4px; }\n"
        "::-webkit-scrollbar-track { background:" + BG + "; }\n"
        "::-webkit-scrollbar-thumb { background:" + BORDER + "; border-radius:2px; }\n"

        # Topbar
        ".topbar { display:flex; align-items:center; justify-content:space-between;"
        " padding:0 20px; height:48px; background:" + SURFACE + "; border-bottom:1px solid " + BORDER + ";"
        " position:sticky; top:0; z-index:100; }\n"
        ".topbar-left { display:flex; align-items:center; gap:16px; }\n"
        ".logo { font-family:" + MONO + "; font-size:14px; font-weight:600; color:" + ORANGE + ";"
        " letter-spacing:3px; text-transform:uppercase; white-space:nowrap; }\n"
        ".divider-v { width:1px; height:20px; background:" + BORDER + "; flex-shrink:0; }\n"
        ".regime-badge { display:flex; align-items:center; gap:6px;"
        " font-family:" + MONO + "; font-size:11px; font-weight:600;"
        " letter-spacing:1px; text-transform:uppercase; color:" + TEXT + "; }\n"
        ".regime-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }\n"
        ".topbar-right { display:flex; align-items:center; gap:16px; }\n"
        ".last-update { font-family:" + MONO + "; font-size:11px; color:" + MUTED + "; white-space:nowrap; }\n"
        ".last-update span { color:" + ORANGE + "; margin-left:6px; }\n"
        ".btn-refresh { font-family:" + MONO + "; font-size:11px; font-weight:600;"
        " letter-spacing:1.5px; text-transform:uppercase; color:" + BG + "; background:" + ORANGE + ";"
        " border:none; padding:6px 16px; cursor:pointer; transition:opacity .15s; white-space:nowrap; }\n"
        ".btn-refresh:hover { opacity:.85; }\n"

        # Main layout
        ".main { padding:16px 20px; display:flex; flex-direction:column; gap:12px; }\n"
        ".card { background:" + SURFACE + "; border:1px solid " + BORDER + "; padding:14px; }\n"
        ".card-header { display:flex; align-items:center; justify-content:space-between;"
        " margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid " + BORDER + "; }\n"
        ".card-title { font-family:" + SANS + "; font-size:10px; font-weight:600;"
        " letter-spacing:1.5px; text-transform:uppercase; color:" + MUTED + "; }\n"
        ".heatmap-wrap { overflow-x:auto; }\n"

        # Fund list
        ".fund-row { border-bottom:1px solid " + BORDER + "; }\n"
        ".fund-row:last-child { border-bottom:none; }\n"
        ".fund-row-header { display:flex; align-items:center; gap:12px; padding:12px 4px;"
        " cursor:pointer; transition:background .1s; user-select:none; }\n"
        ".fund-row-header:hover { background:rgba(255,255,255,0.02); }\n"
        ".fund-arrow { font-family:" + MONO + "; font-size:10px; color:" + MUTED + ";"
        " width:14px; flex-shrink:0; }\n"
        ".fund-color-bar { width:3px; height:14px; border-radius:1px; flex-shrink:0; }\n"
        ".fund-name { font-family:" + MONO + "; font-size:12px; font-weight:600;"
        " color:" + TEXT + "; flex:1; min-width:160px; white-space:nowrap;"
        " overflow:hidden; text-overflow:ellipsis; }\n"
        ".fund-stat-wrap { display:flex; flex-direction:column; align-items:flex-end;"
        " min-width:64px; }\n"
        ".fund-stat-label { font-family:" + SANS + "; font-size:9px; text-transform:uppercase;"
        " letter-spacing:.8px; color:" + MUTED + "; }\n"
        ".fund-stat { font-family:" + MONO + "; font-size:11px; text-align:right;"
        " white-space:nowrap; }\n"

        # Fund detail (accordion content)
        ".fund-detail { padding:4px 0 16px 26px; }\n"
        ".detail-metrics { display:grid; grid-template-columns:repeat(4,1fr);"
        " gap:8px; margin-bottom:12px; }\n"
        ".detail-metric { background:" + SURFACE2 + "; border:1px solid " + BORDER + ";"
        " padding:10px 12px; }\n"
        ".detail-metric-label { font-family:" + SANS + "; font-size:9px; text-transform:uppercase;"
        " letter-spacing:1px; color:" + MUTED + "; margin-bottom:4px; }\n"
        ".detail-metric-value { font-family:" + MONO + "; font-size:18px; font-weight:600; }\n"

        # Period toggle buttons
        ".period-toggle { display:flex; gap:4px; }\n"
        ".period-btn { font-family:" + MONO + "; font-size:10px; font-weight:600;"
        " letter-spacing:1px; text-transform:uppercase; padding:4px 12px;"
        " border:1px solid " + BORDER + "; background:transparent; color:" + MUTED + ";"
        " cursor:pointer; transition:all .15s; }\n"
        ".period-btn:hover { border-color:" + ORANGE + "; color:" + TEXT + "; }\n"
        ".period-btn.active { background:" + ORANGE + "; color:" + BG + ";"
        " border-color:" + ORANGE + "; }\n"

        ".error-msg { font-family:" + MONO + "; font-size:12px; color:" + RED + ";"
        " background:" + SURFACE + "; border:1px solid " + RED + ";"
        " padding:16px 20px; margin:20px; white-space:pre-wrap; }\n"
    )


# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="PnL Monitor | Attribution",
    update_title=None,
    external_stylesheets=["/nimitz.css"],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


@server.route("/nimitz.css")
def serve_css():
    return Response(build_css(), mimetype="text/css")


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div("BLOOMBERG", className="logo"),
            html.Div(className="divider-v"),
            html.Div([
                html.Div(id="regime-dot", className="regime-dot"),
                html.Div(id="regime-label"),
            ], className="regime-badge"),
        ], className="topbar-left"),

        html.Div([
            html.Div([
                "ÚLTIMA CVM:",
                html.Span("—", id="last-cvm-date"),
            ], className="last-update"),
            html.Button("↻ ATUALIZAR", id="btn-refresh", className="btn-refresh", n_clicks=0),
        ], className="topbar-right"),
    ], className="topbar"),

    dcc.Loading(type="circle", color=ORANGE,
                children=html.Div(id="page-content", className="main")),

], style={"minHeight": "100vh", "background": BG})


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_brl(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    sign = "+" if v >= 0 else "-"
    a = abs(v)
    if a >= 1e9:
        return f"{sign}R${a/1e9:.{decimals}f}B"
    if a >= 1e6:
        return f"{sign}R${a/1e6:.{decimals}f}M"
    return f"{sign}R${a/1e3:.0f}K"

def fmt_pct(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{'+' if v >= 0 else ''}{v:.{decimals}f}%"

def color_val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return MUTED
    return GREEN if v >= 0 else RED

def _apply_layout(fig, barmode=None, height=260):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, size=10, color=MUTED),
        margin=dict(l=40, r=8, t=8, b=30),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER,
                        font=dict(family=MONO, size=11, color=TEXT)),
        xaxis=dict(showgrid=False, showline=False, tickcolor=MUTED, ticklen=4,
                   tickfont=dict(size=10), automargin=True),
        yaxis=dict(showgrid=True, gridcolor=BORDER, gridwidth=0.5,
                   showline=False, tickcolor=MUTED, ticklen=4, tickfont=dict(size=10)),
    )
    if barmode:
        fig.update_layout(barmode=barmode)


def _get_vix_regime() -> tuple[str, str]:
    """Returns (label, color) based on latest VIX level."""
    try:
        raw = yf.download("^VIX", period="5d", auto_adjust=True, progress=False)
        level = float(raw["Close"].dropna().iloc[-1])
        if level > 25:
            return f"RISK-OFF  VIX {level:.1f}", RED
        elif level > 20:
            return f"CAUTIOUS  VIX {level:.1f}", YELLOW
        else:
            return f"RISK-ON  VIX {level:.1f}", GREEN
    except Exception:
        return "VIX —", MUTED


def _build_heatmap(df_betas: pd.DataFrame) -> go.Figure:
    fatores = list(FATORES.keys())
    fundos  = sorted(df_betas["fundo"].unique())

    pivot = (
        df_betas[df_betas["fator"].isin(fatores)]
        .pivot(index="fundo", columns="fator", values="beta")
        .reindex(index=fundos, columns=fatores)
    )
    z = pivot.values
    abs_max = float(np.nanmax(np.abs(z))) if not np.all(np.isnan(z)) else 1.0

    fig = go.Figure(go.Heatmap(
        z=z, x=fatores, y=fundos,
        colorscale=[
            [0.0,  "#c0392b"],
            [0.45, "#1a1f2e"],
            [0.5,  "#252d3d"],
            [0.55, "#1a1f2e"],
            [1.0,  "#00c27a"],
        ],
        zmid=0, zmin=-abs_max, zmax=abs_max,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>β = %{z:.4f}<extra></extra>",
        colorbar=dict(
            thickness=10, len=0.8, tickfont=dict(family=MONO, size=9, color=MUTED),
            outlinecolor=BORDER, outlinewidth=1,
        ),
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        height=max(200, len(fundos) * 36 + 80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=MONO, size=9, color=MUTED),
        margin=dict(l=200, r=80, t=10, b=110),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), side="bottom"),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def _build_perf_chart(df_atrib: pd.DataFrame, cdi: pd.Series | None) -> go.Figure:
    fig = go.Figure()
    fundos = sorted(df_atrib["fundo"].unique())

    for i, fundo in enumerate(fundos):
        df_f = (
            df_atrib[df_atrib["fundo"] == fundo]
            .dropna(subset=["retorno_fundo_%"])
            .copy()
        )
        if df_f.empty:
            continue
        df_f["data"] = pd.to_datetime(df_f["data"])
        df_f = df_f.sort_values("data")
        cum = ((1 + df_f["retorno_fundo_%"] / 100).cumprod() - 1) * 100
        fig.add_trace(go.Scatter(
            x=df_f["data"].dt.strftime("%d/%m"),
            y=cum,
            mode="lines",
            name=fundo,
            line=dict(color=FUND_COLORS[i % len(FUND_COLORS)], width=1.5),
            hovertemplate=f"<b>{fundo}</b><br>%{{x}}<br>%{{y:+.2f}}%<extra></extra>",
        ))

    if cdi is not None and not cdi.empty:
        cdi_s = cdi.sort_index()
        cdi_cum = ((1 + cdi_s / 100).cumprod() - 1) * 100
        fig.add_trace(go.Scatter(
            x=[pd.to_datetime(d).strftime("%d/%m") for d in cdi_cum.index],
            y=cdi_cum,
            mode="lines",
            name="CDI",
            line=dict(color=TEXT, width=2, dash="dot"),
            hovertemplate="<b>CDI</b><br>%{x}<br>%{y:+.3f}%<extra></extra>",
        ))

    _apply_layout(fig, height=300)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            font=dict(family=MONO, size=9, color=MUTED),
            bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
    )
    fig.update_yaxes(ticksuffix="%", zeroline=True,
                     zerolinecolor=BORDER, zerolinewidth=1)
    return fig


def _build_fund_detail(fundo: str, df_f: pd.DataFrame, df_betas_f: pd.DataFrame) -> html.Div:
    fatores = list(FATORES.keys())

    pnl_total   = df_f["pnl_brl"].sum()
    retorno_mtd = float((1 + df_f["retorno_fundo_%"] / 100).prod() - 1) * 100
    alpha_mtd   = float((1 + df_f["alpha_%"].fillna(0) / 100).prod() - 1) * 100
    r2_val      = float(df_betas_f["r2"].iloc[0]) * 100 if not df_betas_f.empty else None

    metrics = html.Div([
        html.Div([
            html.Div("PNL MTD",        className="detail-metric-label"),
            html.Div(fmt_brl(pnl_total), className="detail-metric-value",
                     style={"color": color_val(pnl_total)}),
        ], className="detail-metric"),
        html.Div([
            html.Div("RETORNO MTD",       className="detail-metric-label"),
            html.Div(fmt_pct(retorno_mtd), className="detail-metric-value",
                     style={"color": color_val(retorno_mtd)}),
        ], className="detail-metric"),
        html.Div([
            html.Div("ALPHA MTD",         className="detail-metric-label"),
            html.Div(fmt_pct(alpha_mtd),  className="detail-metric-value",
                     style={"color": color_val(alpha_mtd)}),
        ], className="detail-metric"),
        html.Div([
            html.Div("R² MODELO", className="detail-metric-label"),
            html.Div(f"{r2_val:.1f}%" if r2_val is not None else "—",
                     className="detail-metric-value", style={"color": ORANGE}),
        ], className="detail-metric"),
    ], className="detail-metrics")

    # Stacked bar — factor PnL attribution
    fig = go.Figure()
    for fator in fatores + ["Alpha"]:
        key = f"pnl_{fator}_brl" if fator != "Alpha" else "pnl_alpha_brl"
        if key not in df_f.columns:
            continue
        fig.add_trace(go.Bar(
            name=fator,
            x=df_f["data"].dt.strftime("%d/%m"),
            y=df_f[key] / 1e6,
            marker_color=FATOR_COLORS.get(fator, MUTED),
            marker_line_width=0,
            hovertemplate=f"<b>{fator}</b><br>%{{x}}<br>R$%{{y:.2f}}M<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        name="Total",
        x=df_f["data"].dt.strftime("%d/%m"),
        y=df_f["pnl_brl"] / 1e6,
        mode="markers",
        marker=dict(symbol="diamond", size=6, color=TEXT,
                    line=dict(width=1, color=SURFACE)),
        hovertemplate="<b>Total</b><br>%{x}<br>R$%{y:.2f}M<extra></extra>",
    ))
    _apply_layout(fig, barmode="relative", height=220)
    fig.update_yaxes(tickformat=".1f", ticksuffix="M",
                     zeroline=True, zerolinecolor=BORDER, zerolinewidth=1)

    return html.Div([
        metrics,
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"height": "220px"}),
    ], className="fund-detail")


def _build_fund_list(df_atrib: pd.DataFrame, df_betas: pd.DataFrame) -> list:
    fundos = sorted(df_atrib["fundo"].unique())
    rows   = []

    for i, fundo in enumerate(fundos):
        df_f = (
            df_atrib[df_atrib["fundo"] == fundo]
            .dropna(subset=["retorno_fundo_%"])
            .copy()
        )
        if df_f.empty:
            continue

        df_f["data"] = pd.to_datetime(df_f["data"])
        df_f = df_f.sort_values("data").reset_index(drop=True)
        df_betas_f = df_betas[df_betas["fundo"] == fundo]

        retorno_mtd = float((1 + df_f["retorno_fundo_%"] / 100).prod() - 1) * 100
        alpha_mtd   = float((1 + df_f["alpha_%"].fillna(0) / 100).prod() - 1) * 100
        pnl_mtd     = df_f["pnl_brl"].sum()
        r2_val      = float(df_betas_f["r2"].iloc[0]) * 100 if not df_betas_f.empty else None
        _ibov       = df_betas_f[df_betas_f["fator"] == "Ibovespa"]["beta"]
        beta_ibov   = float(_ibov.iloc[0]) if not _ibov.empty else None
        color       = FUND_COLORS[i % len(FUND_COLORS)]

        rows.append(html.Div([
            # ── Header (clicável) ────────────────────────────────────────
            html.Div([
                html.Div("▶", id={"type": "fund-arrow", "index": fundo},
                         className="fund-arrow"),
                html.Div(className="fund-color-bar",
                         style={"background": color}),
                html.Div(fundo.upper(), className="fund-name"),

                html.Div([
                    html.Div("MTD",          className="fund-stat-label"),
                    html.Div(fmt_pct(retorno_mtd), className="fund-stat",
                             style={"color": color_val(retorno_mtd)}),
                ], className="fund-stat-wrap"),

                html.Div([
                    html.Div("ALPHA",        className="fund-stat-label"),
                    html.Div(fmt_pct(alpha_mtd), className="fund-stat",
                             style={"color": color_val(alpha_mtd)}),
                ], className="fund-stat-wrap"),

                html.Div([
                    html.Div("PNL",          className="fund-stat-label"),
                    html.Div(fmt_brl(pnl_mtd), className="fund-stat",
                             style={"color": color_val(pnl_mtd)}),
                ], className="fund-stat-wrap"),

                html.Div([
                    html.Div("β IBOV", className="fund-stat-label"),
                    html.Div(f"{beta_ibov:+.3f}" if beta_ibov is not None else "—",
                             className="fund-stat",
                             style={"color": color_val(beta_ibov)}),
                ], className="fund-stat-wrap"),

                html.Div([
                    html.Div("R²",   className="fund-stat-label"),
                    html.Div(f"{r2_val:.0f}%" if r2_val is not None else "—",
                             className="fund-stat", style={"color": ORANGE}),
                ], className="fund-stat-wrap"),

            ], id={"type": "fund-btn", "index": fundo},
               className="fund-row-header", n_clicks=0),

            # ── Detalhe (oculto por padrão) ──────────────────────────────
            html.Div(
                _build_fund_detail(fundo, df_f, df_betas_f),
                id={"type": "fund-detail", "index": fundo},
                style={"display": "none"},
            ),
        ], className="fund-row"))

    return rows


# ── Callback principal — busca dados e monta layout ──────────────────────────

@app.callback(
    Output("page-content",  "children"),
    Output("last-cvm-date", "children"),
    Output("regime-dot",    "style"),
    Output("regime-label",  "children"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=False,
)
def refresh_dashboard(n_clicks):
    ctx       = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    should_fetch = _cache["df_atrib"] is None or "btn-refresh" in triggered

    try:
        if should_fetch:
            ret_mercado        = obter_retornos_mercado(FATORES, DATA_INICIO, DATA_FIM)
            df_cotas           = obter_cotas_fundos(FUNDOS)
            df_ret             = calcular_retorno_fundo(df_cotas)
            _cache["df_atrib"]    = atribuir_pnl(df_ret, ret_mercado, JANELA_ROLLING, RIDGE_ALPHA)
            _cache["df_betas_ytd"] = resumo_betas(df_ret, ret_mercado)

            # Betas MTD — usa Ridge pois o mês corrente tem poucos dias vs 21 fatores
            from datetime import date as _date
            mtd_start = _date.today().replace(day=1)
            df_ret_mtd      = df_ret[df_ret["data"] >= mtd_start]
            ret_mercado_mtd = ret_mercado[ret_mercado.index >= mtd_start]
            try:
                _cache["df_betas_mtd"] = resumo_betas(df_ret_mtd, ret_mercado_mtd,
                                                       ridge_alpha=RIDGE_ALPHA)
            except Exception:
                _cache["df_betas_mtd"] = None

            try:
                _cache["cdi"] = obter_cdi(DATA_INICIO, DATA_FIM)
            except Exception:
                _cache["cdi"] = None

        df_atrib    = _cache["df_atrib"]
        df_betas    = _cache["df_betas_ytd"]
        cdi         = _cache["cdi"]

        if df_atrib is None or df_atrib.empty:
            return [html.Div("Sem dados disponíveis.", className="error-msg")], "—", {}, "—"

        # Regime
        regime_text, regime_color = _get_vix_regime()
        dot_style = {"width": "8px", "height": "8px", "borderRadius": "50%",
                     "background": regime_color, "flexShrink": "0"}

        # Última data CVM
        ultima_data = pd.to_datetime(str(df_atrib["data"].max())).strftime("%d/%m/%Y")

        # ── Seção 1: Heatmap betas ────────────────────────────────────────
        heatmap_section = html.Div([
            html.Div([
                html.Span("POSICIONAMENTO POR FATOR — BETAS", className="card-title"),
                html.Div([
                    html.Button("MTD", id="btn-period-mtd", n_clicks=0,
                                className="period-btn"),
                    html.Button("YTD", id="btn-period-ytd", n_clicks=0,
                                className="period-btn active"),
                ], className="period-toggle"),
            ], className="card-header"),
            html.Div(
                dcc.Graph(id="heatmap-graph",
                          figure=_build_heatmap(df_betas),
                          config={"displayModeBar": False}),
                className="heatmap-wrap",
            ),
        ], className="card")

        # ── Seção 2: Performance vs CDI ───────────────────────────────────
        perf_section = html.Div([
            html.Div([
                html.Span("PERFORMANCE ACUMULADA MTD vs CDI", className="card-title"),
                html.Span("retorno bruto acumulado · CDI = linha pontilhada",
                          style={"fontFamily": MONO, "fontSize": "9px", "color": MUTED}),
            ], className="card-header"),
            dcc.Graph(figure=_build_perf_chart(df_atrib, cdi),
                      config={"displayModeBar": False},
                      style={"height": "300px"}),
        ], className="card")

        # ── Seção 3: Lista de fundos (accordion) ──────────────────────────
        fund_section = html.Div([
            html.Div([
                html.Span("FUNDOS — CLIQUE PARA DETALHAR", className="card-title"),
                html.Span("MTD  ·  alpha  ·  PnL  ·  β Ibovespa  ·  R²",
                          style={"fontFamily": MONO, "fontSize": "9px", "color": MUTED}),
            ], className="card-header"),
            html.Div(_build_fund_list(df_atrib, _cache["df_betas_ytd"]), id="fund-list"),
        ], className="card")

        return [heatmap_section, perf_section, fund_section], ultima_data, dot_style, regime_text

    except Exception as e:
        import traceback
        return (
            [html.Div(f"ERRO: {str(e)}\n\n{traceback.format_exc()}", className="error-msg")],
            "ERRO", {}, "ERRO",
        )


# ── Heatmap period toggle ────────────────────────────────────────────────────

@app.callback(
    Output("heatmap-graph",   "figure"),
    Output("btn-period-mtd",  "className"),
    Output("btn-period-ytd",  "className"),
    Input("btn-period-mtd",   "n_clicks"),
    Input("btn-period-ytd",   "n_clicks"),
    prevent_initial_call=True,
)
def update_heatmap_period(n_mtd, n_ytd):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else "btn-period-ytd"
    period = "MTD" if "mtd" in triggered else "YTD"

    df_betas = _cache.get(f"df_betas_{period.lower()}")
    if df_betas is None or df_betas.empty:
        raise dash.exceptions.PreventUpdate

    mtd_cls = "period-btn active" if period == "MTD" else "period-btn"
    ytd_cls = "period-btn active" if period == "YTD" else "period-btn"
    return _build_heatmap(df_betas), mtd_cls, ytd_cls


# ── Accordion toggle ──────────────────────────────────────────────────────────

@app.callback(
    Output({"type": "fund-detail", "index": ALL}, "style"),
    Output({"type": "fund-arrow",  "index": ALL}, "children"),
    Input({"type":  "fund-btn",    "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def toggle_fund(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    clicked_fund = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])["index"]
    clicked_val  = ctx.triggered[0]["value"] or 0
    is_opening   = (clicked_val % 2) == 1   # ímpar = abrir, par = fechar

    styles, arrows = [], []
    for inp in ctx.inputs_list[0]:
        fund = inp["id"]["index"]
        if fund == clicked_fund:
            styles.append({"display": "block" if is_opening else "none"})
            arrows.append("▼" if is_opening else "▶")
        else:
            styles.append({"display": "none"})
            arrows.append("▶")

    return styles, arrows


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
