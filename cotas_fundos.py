import requests
import zipfile
import io
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd

# ── Fundos que você quer acompanhar ──────────────────────────────────────────
FUNDOS = {
    "SPX Nimitz Feeder": "12.831.360/0001-14",
    # Adicione mais fundos aqui:
    # "Nome do Fundo": "XX.XXX.XXX/0001-XX",
}

# ── Tenta mês atual, se ainda não disponível cai pro mês anterior ────────────
def baixar_informe(ano_mes: str) -> pd.DataFrame:
    url = f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_{ano_mes}.zip"
    print(f"Baixando: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open(z.namelist()[0]) as f:
            return pd.read_csv(f, sep=";", dtype={"CNPJ_FUNDO": str}, encoding="latin-1")

hoje = date.today()
for delta in [0, 1]:
    mes_ref = hoje - relativedelta(months=delta)
    ano_mes = mes_ref.strftime("%Y%m")
    try:
        df = baixar_informe(ano_mes)
        print(f"Mês carregado: {mes_ref.strftime('%B/%Y')}\n")
        break
    except Exception as e:
        print(f"Mês {ano_mes} indisponível, tentando anterior...")
else:
    raise RuntimeError("Não foi possível baixar nenhum informe diário da CVM.")

# ── Normaliza e filtra ────────────────────────────────────────────────────────
# CVM renamed the column in 2025; support both old and new names
cnpj_col = "CNPJ_FUNDO_CLASSE" if "CNPJ_FUNDO_CLASSE" in df.columns else "CNPJ_FUNDO"
df[cnpj_col] = df[cnpj_col].str.strip()
cnpjs = list(FUNDOS.values())
df_filtrado = df[df[cnpj_col].isin(cnpjs)].copy()

if df_filtrado.empty:
    print("Nenhum dos CNPJs foi encontrado no arquivo. Verifique os CNPJs em FUNDOS.")
else:
    cnpj_para_nome = {v: k for k, v in FUNDOS.items()}
    df_filtrado["NOME_FUNDO"] = df_filtrado[cnpj_col].map(cnpj_para_nome)

    colunas = ["DT_COMPTC", "NOME_FUNDO", cnpj_col, "VL_QUOTA", "VL_PATRIM_LIQ", "NR_COTST"]
    df_resultado = (
        df_filtrado[colunas]
        .sort_values(["NOME_FUNDO", "DT_COMPTC"])
        .reset_index(drop=True)
    )

    print(f"Registros encontrados: {len(df_resultado)}\n")
    pd.set_option("display.float_format", "{:.6f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(df_resultado.to_string(index=False))
