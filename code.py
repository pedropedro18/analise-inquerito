import re
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Análise de Inquérito", layout="wide")
st.title("📊 Análise de respostas de inquérito")

STOPWORDS = set(
    "a o as os um uma uns umas de do da dos das em no na nos nas por para com "
    "sem e ou que se não nao mais muito muita foi é são ser ter tem como mas "
    "ao aos à às me te lhe eu tu ele ela nós vos eles elas já ja só so pelo pela".split()
)


@st.cache_data
def carregar(ficheiro):
    nome = ficheiro.name.lower()
    if nome.endswith(".csv"):
        try:
            return pd.read_csv(ficheiro, sep=None, engine="python", encoding="utf-8")
        except UnicodeDecodeError:
            ficheiro.seek(0)
            return pd.read_csv(ficheiro, sep=None, engine="python", encoding="latin-1")
    return pd.read_excel(ficheiro)


def tipo_coluna(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "vazia"
    if pd.api.types.is_numeric_dtype(s):
        return "numérica" if s.nunique() > 10 else "escala/categoria"
    if s.astype(str).str.contains(r"[;,]").mean() > 0.3 and s.nunique() < len(s) * 0.5:
        return "escolha múltipla"
    if s.nunique() <= 15 or s.nunique() / len(s) < 0.2:
        return "categórica"
    return "texto livre"


def tabela_freq(s: pd.Series) -> pd.DataFrame:
    c = s.value_counts(dropna=True)
    return pd.DataFrame({"Respostas": c, "%": (c / c.sum() * 100).round(1)})


def analisar_coluna(nome, s, tipo):
    st.subheader(nome)
    st.caption(f"Tipo detetado: {tipo} · {s.notna().sum()} respostas · {s.isna().sum()} em branco")

    if tipo == "numérica":
        st.dataframe(s.describe().round(2).to_frame("Valor"))
        contagens, limites = np.histogram(s.dropna(), bins=10)
        hist = pd.DataFrame(
            {"Frequência": contagens},
            index=[f"{limites[i]:.1f}–{limites[i+1]:.1f}" for i in range(len(contagens))],
        )
        st.bar_chart(hist)

    elif tipo in ("escala/categoria", "categórica"):
        freq = tabela_freq(s)
        c1, c2 = st.columns([1, 2])
        c1.dataframe(freq)
        c2.bar_chart(freq["Respostas"])
        if tipo == "escala/categoria":
            st.metric("Média", round(float(s.mean()), 2))

    elif tipo == "escolha múltipla":
        itens = s.dropna().astype(str).str.split(r"\s*[;,]\s*").explode().str.strip()
        freq = tabela_freq(itens)
        c1, c2 = st.columns([1, 2])
        c1.dataframe(freq)
        c2.bar_chart(freq["Respostas"])

    elif tipo == "texto livre":
        palavras = []
        for txt in s.dropna().astype(str):
            palavras += [
                p for p in re.findall(r"\w+", txt.lower())
                if p not in STOPWORDS and len(p) > 2
            ]
        top = Counter(palavras).most_common(15)
        if top:
            df = pd.DataFrame(top, columns=["Palavra", "Ocorrências"]).set_index("Palavra")
            c1, c2 = st.columns([1, 2])
            c1.dataframe(df)
            c2.bar_chart(df)
        with st.expander("Ver todas as respostas"):
            st.write(s.dropna().tolist())


ficheiro = st.file_uploader("Carrega as respostas (CSV ou Excel)", type=["csv", "xlsx", "xls"])

if ficheiro is None:
    st.info("Carrega um ficheiro exportado do Google Forms, Microsoft Forms, etc.")
    st.stop()

df = carregar(ficheiro)
st.success(f"{len(df)} respostas · {len(df.columns)} perguntas")

# Filtros opcionais
with st.sidebar:
    st.header("Filtros")
    colunas_filtro = st.multiselect("Filtrar por pergunta", df.columns)
    for col in colunas_filtro:
        opcoes = df[col].dropna().unique().tolist()
        escolha = st.multiselect(col, opcoes, default=opcoes, key=f"f_{col}")
        df = df[df[col].isin(escolha) | df[col].isna()]
    st.caption(f"{len(df)} respostas após filtros")

aba1, aba2, aba3 = st.tabs(["Resumo por pergunta", "Cruzar duas perguntas", "Dados"])

with aba1:
    for col in df.columns:
        analisar_coluna(col, df[col], tipo_coluna(df[col]))
        st.divider()

with aba2:
    c1, c2 = st.columns(2)
    a = c1.selectbox("Pergunta A (linhas)", df.columns, key="a")
    b = c2.selectbox("Pergunta B (colunas)", df.columns, index=min(1, len(df.columns) - 1), key="b")
    if a != b:
        cruz = pd.crosstab(df[a], df[b])
        st.dataframe(cruz)
        st.caption("Percentagem por linha")
        st.dataframe((pd.crosstab(df[a], df[b], normalize="index") * 100).round(1))
        st.bar_chart(cruz)
    else:
        st.warning("Escolhe duas perguntas diferentes.")

with aba3:
    st.dataframe(df, use_container_width=True)
    buf = BytesIO()
    df.to_excel(buf, index=False)
    st.download_button("⬇️ Descarregar dados filtrados (Excel)", buf.getvalue(), "respostas_filtradas.xlsx")