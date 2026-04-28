"""
atribuicao_pnl.py
-----------------
Junta dados de mercado (Yahoo Finance) com cotas de fundos (CVM) e
decompõe o PnL diário do fundo em contribuições por fator de mercado
via regressão OLS (betas rolling ou full-period).

Uso:
    python atribuicao_pnl.py

Saída:
    atribuicao_pnl.csv   — tabela completa de atribuição
    betas_ols.csv        — betas estimados (e erros padrão)
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import io
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf
from dateutil.relativedelta import relativedelta

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURAÇÃO CENTRAL                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Período de análise — YTD (1º de janeiro do ano corrente)
_hoje       = date.today()
DATA_INICIO = _hoje.replace(month=1, day=1).strftime("%Y-%m-%d")  # 1º jan do ano corrente
DATA_FIM    = (_hoje + timedelta(days=1)).strftime("%Y-%m-%d")     # +1 dia (end exclusivo no yfinance)

# Fundos monitorados — importados de cotas_fundos.py (única fonte de verdade)
from cotas_fundos import FUNDOS  # noqa: E402

# Fatores de mercado  →  {nome_amigável: ticker Yahoo Finance}
FATORES = {
    # ── Renda Variável ──────────────────────────────────────────────
    "Ibovespa":     "^BVSP",
    "S&P 500":      "^GSPC",
    "Nasdaq":       "^NDX",
    "Euro Stoxx":   "^STOXX50E",
    "Hang Seng":    "^HSI",
    # ── Juros ────────────────────────────────────────────────────────
    "Treasury 10Y": "^TNX",
    "Treasury 3M":  "^IRX",
    "Treasury 30Y": "^TYX",
    "IMA-B 5+":     "B5P211.SA",
    "IRF-M":        "IRFM11.SA",
    # ── Moedas ───────────────────────────────────────────────────────
    "USD/BRL":      "BRL=X",
    "EUR/BRL":      "EURBRL=X",
    "JPY/BRL":      "JPYBRL=X",
    "DXY":          "DX-Y.NYB",
    # ── Commodities ──────────────────────────────────────────────────
    "Ouro":         "GC=F",
    "Petróleo":     "CL=F",
    "Prata":        "SI=F",
    "Cobre":        "HG=F",
    "Açúcar":       "SB=F",
    # ── Volatilidade / Risk-On ───────────────────────────────────────
    "VIX":          "^VIX",
    "Bitcoin":      "BTC-USD",
}

# Janela rolling para betas (None = usa o período inteiro)
JANELA_ROLLING = None   # ex: 21 para betas móveis de 21 dias

# Ridge regularization — evita multicolinearidade com muitos fatores
# None = OLS puro | float (ex: 1.0) = Ridge
RIDGE_ALPHA = 1.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     MÓDULO 1 — DADOS DE MERCADO                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def obter_retornos_mercado(
    fatores: dict[str, str],
    data_inicio: str,
    data_fim: str,
) -> pd.DataFrame:
    """
    Baixa preços de fechamento via Yahoo Finance e calcula retornos diários (%).

    Retorna
    -------
    DataFrame  [date index, colunas = nomes amigáveis dos fatores]
    """
    tickers = list(fatores.values())
    nome_para_ticker = {v: k for k, v in fatores.items()}

    print(f"[mercado] Baixando {len(tickers)} tickers: {data_inicio} → {data_fim}")
    raw = yf.download(
        tickers,
        start=data_inicio,
        end=data_fim,
        auto_adjust=True,
        progress=False,
    )

    # Suporte a download de ticker único (yfinance retorna estrutura diferente)
    if isinstance(raw.columns, pd.MultiIndex):
        fechamento = raw["Close"].copy()
    else:
        fechamento = raw[["Close"]].copy()
        fechamento.columns = tickers

    fechamento.index = pd.to_datetime(fechamento.index).date
    fechamento.rename(columns=nome_para_ticker, inplace=True)

    retornos = fechamento.pct_change() * 100
    retornos.index.name = "data"

    print(f"[mercado] {len(retornos)} linhas carregadas.\n")
    return retornos


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     MÓDULO 2 — COTAS CVM                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _baixar_informe_cvm(ano_mes: str) -> pd.DataFrame:
    """Baixa e descompacta o informe diário da CVM para um dado ano-mês (YYYYMM)."""
    url = (
        f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
        f"inf_diario_fi_{ano_mes}.zip"
    )
    print(f"[cvm] Baixando: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(
                f, sep=";", dtype={"CNPJ_FUNDO": str, "CNPJ_FUNDO_CLASSE": str},
                encoding="latin-1",
            )


def obter_cotas_fundos(fundos: dict[str, str]) -> pd.DataFrame:
    """
    Baixa cotas diárias da CVM para os CNPJs informados.
    Tenta os últimos 3 meses e combina tudo — garante que fundos que ainda não
    publicaram o mês corrente apareçam com dados do mês anterior.

    Retorna
    -------
    DataFrame com colunas: data, fundo, cnpj, vl_quota, patrimonio_liq, num_cotistas
    """
    hoje = date.today()

    # Calcula quantos meses cobrir a partir de DATA_INICIO
    inicio = date.fromisoformat(DATA_INICIO)
    meses_cobrir = (hoje.year - inicio.year) * 12 + (hoje.month - inicio.month) + 1

    dfs = []
    for delta in range(meses_cobrir):
        mes_ref = hoje - relativedelta(months=delta)
        ano_mes = mes_ref.strftime("%Y%m")
        try:
            dfs.append(_baixar_informe_cvm(ano_mes))
            print(f"[cvm] Carregado: {mes_ref.strftime('%B/%Y')}")
        except Exception:
            print(f"[cvm] {ano_mes} indisponível, pulando...")

    if not dfs:
        raise RuntimeError("Não foi possível baixar nenhum informe diário da CVM.")

    df_raw = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"[cvm] Total combinado: {len(df_raw)} registros\n")

    # Suporte ao rename de coluna da CVM em 2025
    cnpj_col = (
        "CNPJ_FUNDO_CLASSE" if "CNPJ_FUNDO_CLASSE" in df_raw.columns else "CNPJ_FUNDO"
    )
    df_raw[cnpj_col] = df_raw[cnpj_col].str.strip()

    cnpjs = list(fundos.values())
    df_f = df_raw[df_raw[cnpj_col].isin(cnpjs)].copy()

    if df_f.empty:
        raise ValueError(
            "Nenhum CNPJ encontrado no arquivo CVM. Verifique o dicionário FUNDOS."
        )

    cnpj_para_nome = {v: k for k, v in fundos.items()}
    df_f["fundo"] = df_f[cnpj_col].map(cnpj_para_nome)
    df_f["data"]  = pd.to_datetime(df_f["DT_COMPTC"]).dt.date

    df_out = (
        df_f.rename(columns={
            cnpj_col:        "cnpj",
            "VL_QUOTA":      "vl_quota",
            "VL_PATRIM_LIQ": "patrimonio_liq",
            "NR_COTST":      "num_cotistas",
        })
        [["data", "fundo", "cnpj", "vl_quota", "patrimonio_liq", "num_cotistas"]]
        .sort_values(["fundo", "data"])
        .reset_index(drop=True)
    )

    print(f"[cvm] {len(df_out)} registros encontrados.\n")
    return df_out


def obter_cdi(data_inicio: str, data_fim: str) -> pd.Series:
    """
    Busca CDI diário via API do Banco Central (série 12).
    Retorna Series com index=date, values=retorno diário em %.
    """
    d_ini = pd.to_datetime(data_inicio).strftime("%d/%m/%Y")
    d_fim = min(pd.to_datetime(data_fim).date(), date.today()).strftime("%d/%m/%Y")
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        f"?formato=json&dataInicial={d_ini}&dataFinal={d_fim}"
    )
    print(f"[cdi] Buscando: {d_ini} → {d_fim}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    s = pd.Series(
        {pd.to_datetime(r["data"], dayfirst=True).date(): float(r["valor"])
         for r in resp.json()},
        name="CDI",
    )
    s.index.name = "data"
    print(f"[cdi] {len(s)} dias carregados.\n")
    return s


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     MÓDULO 3 — ATRIBUIÇÃO DE PnL                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calcular_retorno_fundo(df_cotas: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o retorno diário da cota (%) e o PnL em R$ por dia.

    PnL_BRL = retorno_dia (decimal) × patrimônio_liq do dia anterior
    """
    df = df_cotas.copy().sort_values(["fundo", "data"])

    df["retorno_fundo_%"] = (
        df.groupby("fundo")["vl_quota"]
        .pct_change() * 100
    )
    df["patrimonio_anterior"] = (
        df.groupby("fundo")["patrimonio_liq"].shift(1)
    )
    df["pnl_brl"] = (
        df["retorno_fundo_%"] / 100 * df["patrimonio_anterior"]
    )
    return df


def estimar_betas(
    retorno_fundo: pd.Series,
    retornos_fatores: pd.DataFrame,
    ridge_alpha: float | None = None,
) -> tuple:
    """
    Estima betas via OLS ou Ridge (quando ridge_alpha > 0).

    Ridge é recomendado com muitos fatores correlacionados para evitar
    multicolinearidade. ridge_alpha controla a força da regularização.

    Retorna (params: Series, modelo)
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    dados = retornos_fatores.join(retorno_fundo.rename("fundo"), how="inner").dropna()
    X_raw = dados[retornos_fatores.columns]
    y = dados["fundo"]

    if ridge_alpha is not None:
        # Ridge: normaliza X, estima, desnormaliza betas
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(X_scaled, y)
        # Desnormaliza betas para escala original
        betas_raw = model.coef_ / scaler.scale_
        intercept = model.intercept_ - (betas_raw * scaler.mean_).sum()
        params = pd.Series(
            [intercept] + list(betas_raw),
            index=["const"] + list(retornos_fatores.columns),
        )
        # Calcula R² para diagnóstico
        y_pred = model.predict(X_scaled)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        # Objeto mock para compatibilidade com resumo_betas
        class _MockModel:
            def __init__(self):
                self.params   = params
                self.rsquared = r2
                self.bse      = pd.Series(np.nan, index=params.index)
                self.tvalues  = pd.Series(np.nan, index=params.index)
                self.pvalues  = pd.Series(np.nan, index=params.index)
        return params, _MockModel()
    else:
        # OLS puro
        X = sm.add_constant(X_raw)
        modelo = sm.OLS(y, X).fit()
        return modelo.params, modelo


# Alias para compatibilidade retroativa
def estimar_betas_ols(retorno_fundo, retornos_fatores):
    return estimar_betas(retorno_fundo, retornos_fatores, ridge_alpha=None)


def atribuir_pnl(
    df_cotas_retorno: pd.DataFrame,
    retornos_mercado: pd.DataFrame,
    janela_rolling: int | None = None,
    ridge_alpha: float | None = None,
) -> pd.DataFrame:
    """
    Junta cotas e mercado e decompõe o PnL diário por fator.

    Para cada dia d:
        contrib_fator_d = beta_fator × retorno_fator_d
        alpha_d         = retorno_fundo_d − Σ(contrib_fator_d) − intercepto

    Parâmetros
    ----------
    janela_rolling : se None, usa betas do período inteiro para todos os dias.
                     Se inteiro (ex: 21), recalcula betas com janela deslizante.

    Retorna
    -------
    DataFrame com colunas:
        data, fundo, retorno_fundo_%, pnl_brl,
        [contrib_<fator> para cada fator],
        alpha_%, pnl_explicado_brl, pnl_alpha_brl
    """
    resultados = []

    for nome_fundo, grupo in df_cotas_retorno.groupby("fundo"):
        grupo = grupo.set_index("data").sort_index()

        # Alinha datas entre fundo e mercado
        # left join: mantém todas as datas do fundo;
        # fatores sem dado naquele dia ficam 0 (sem contribuição)
        merged = grupo[["retorno_fundo_%", "pnl_brl", "patrimonio_anterior"]].join(
            retornos_mercado, how="left"
        ).dropna(subset=["retorno_fundo_%"])
        # Preenche fatores ausentes com 0 (não contribuem naquele dia)
        for col in retornos_mercado.columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
            else:
                merged[col] = 0.0

        fatores_cols = list(retornos_mercado.columns)

        if janela_rolling is None:
            # ── Betas do período inteiro ──────────────────────────────────
            betas, modelo = estimar_betas(
                merged["retorno_fundo_%"], merged[fatores_cols],
                ridge_alpha=ridge_alpha,
            )
            betas_df = pd.DataFrame(
                [betas.values] * len(merged),
                index=merged.index,
                columns=betas.index,
            )
        else:
            # ── Betas rolling ─────────────────────────────────────────────
            betas_list = []
            for i in range(len(merged)):
                if i < janela_rolling:
                    betas_list.append(pd.Series(np.nan, index=["const"] + fatores_cols))
                    continue
                janela = merged.iloc[i - janela_rolling : i]
                b, _ = estimar_betas(
                    janela["retorno_fundo_%"], janela[fatores_cols],
                    ridge_alpha=ridge_alpha,
                )
                betas_list.append(b)
            betas_df = pd.DataFrame(betas_list, index=merged.index)

        # ── Calcula contribuições diárias ─────────────────────────────────
        contrib_cols = []
        for fator in fatores_cols:
            col = f"contrib_{fator}_%"
            contrib_cols.append(col)
            merged[col] = betas_df[fator] * merged[fator]

        merged["soma_contribs_%"] = merged[contrib_cols].sum(axis=1)
        merged["alpha_%"] = (
            merged["retorno_fundo_%"]
            - betas_df.get("const", 0)
            - merged["soma_contribs_%"]
        )

        # ── PnL em R$ por componente ──────────────────────────────────────
        for fator in fatores_cols:
            merged[f"pnl_{fator}_brl"] = (
                merged[f"contrib_{fator}_%"] / 100 * merged["patrimonio_anterior"]
            )
        merged["pnl_alpha_brl"]     = merged["alpha_%"] / 100 * merged["patrimonio_anterior"]
        merged["pnl_explicado_brl"] = merged[[f"pnl_{f}_brl" for f in fatores_cols]].sum(axis=1)

        merged.insert(0, "fundo", nome_fundo)
        resultados.append(merged.reset_index())

    return pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame()


def resumo_betas(
    df_cotas_retorno: pd.DataFrame,
    retornos_mercado: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retorna tabela de betas OLS (full-period) com erro padrão e t-stat.
    Sempre usa OLS puro para preservar inferência estatística (p-values, t-stats).
    """
    rows = []
    for nome_fundo, grupo in df_cotas_retorno.groupby("fundo"):
        grupo = grupo.set_index("data").sort_index()
        fatores_cols = list(retornos_mercado.columns)
        merged = grupo[["retorno_fundo_%"]].join(retornos_mercado, how="inner").dropna()
        _, modelo = estimar_betas_ols(merged["retorno_fundo_%"], merged[fatores_cols])

        for param in modelo.params.index:
            rows.append({
                "fundo":   nome_fundo,
                "fator":   param,
                "beta":    modelo.params[param],
                "std_err": modelo.bse[param],
                "t_stat":  modelo.tvalues[param],
                "p_value": modelo.pvalues[param],
                "r2":      modelo.rsquared,
            })
    return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            MAIN                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)

    # 1. Coleta de dados
    retornos_mercado = obter_retornos_mercado(FATORES, DATA_INICIO, DATA_FIM)
    df_cotas         = obter_cotas_fundos(FUNDOS)

    # 2. Retorno diário da cota + PnL em R$
    df_cotas_ret = calcular_retorno_fundo(df_cotas)

    # 3. Atribuição de PnL (Ridge para estabilidade com 21 fatores)
    df_atrib = atribuir_pnl(df_cotas_ret, retornos_mercado, JANELA_ROLLING, RIDGE_ALPHA)

    # 4. Tabela de betas (OLS para inferência estatística)
    df_betas = resumo_betas(df_cotas_ret, retornos_mercado)

    # 5. Exibe resultados
    fatores_cols = list(FATORES.keys())
    colunas_exibir = (
        ["data", "fundo", "retorno_fundo_%", "pnl_brl"]
        + [f"contrib_{f}_%" for f in fatores_cols]
        + ["alpha_%", "pnl_explicado_brl", "pnl_alpha_brl"]
    )
    print("=" * 80)
    print("ATRIBUIÇÃO DE PnL DIÁRIO")
    print("=" * 80)
    print(df_atrib[colunas_exibir].to_string(index=False))

    print("\n" + "=" * 80)
    print("BETAS OLS (período completo)")
    print("=" * 80)
    pd.set_option("display.float_format", "{:.6f}".format)
    print(df_betas.to_string(index=False))

    # 6. Salva CSVs
    df_atrib.to_csv("atribuicao_pnl.csv", index=False)
    df_betas.to_csv("betas_ols.csv", index=False)
    print("\nArquivos salvos: atribuicao_pnl.csv | betas_ols.csv")


if __name__ == "__main__":
    main()
