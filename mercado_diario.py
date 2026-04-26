import yfinance as yf
import pandas as pd
from datetime import date

# ── Datas de referência (dias úteis do Nimitz em abril/2026) ─────────────────
# O script detecta automaticamente o intervalo a partir das cotas
# mas você pode sobrescrever manualmente se quiser:
DATA_INICIO = "2026-04-01"
DATA_FIM    = "2026-04-23"

# ── Ativos de mercado (tickers Yahoo Finance) ────────────────────────────────
ATIVOS = {
    "Ibovespa":  "^BVSP",
    "S&P 500":   "^GSPC",
    "USD/BRL":   "BRL=X",
    "Ouro":      "GC=F",
    "Petróleo":  "CL=F",
}

# ── Baixa dados de todos os ativos de uma vez ────────────────────────────────
print(f"Baixando dados de mercado: {DATA_INICIO} → {DATA_FIM}\n")

tickers = list(ATIVOS.values())
raw = yf.download(
    tickers,
    start=DATA_INICIO,
    end=DATA_FIM,
    auto_adjust=True,
    progress=False,
)

# Pega só o fechamento
fechamento = raw["Close"].copy()
fechamento.index = pd.to_datetime(fechamento.index).date

# Renomeia colunas para nomes legíveis
nome_para_ticker = {v: k for k, v in ATIVOS.items()}
fechamento.rename(columns=nome_para_ticker, inplace=True)

# ── Calcula retorno diário (%) ────────────────────────────────────────────────
retorno = fechamento.pct_change() * 100

# ── Exibe resultados ─────────────────────────────────────────────────────────
print("=== Fechamento diário ===")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)
print(fechamento.to_string())

print("\n=== Retorno diário (%) ===")
pd.set_option("display.float_format", "{:.2f}".format)
print(retorno.to_string())

# ── Salva CSV para usar depois ────────────────────────────────────────────────
fechamento.to_csv("mercado_fechamento.csv")
retorno.to_csv("mercado_retorno.csv")
print("\nArquivos salvos: mercado_fechamento.csv | mercado_retorno.csv")
