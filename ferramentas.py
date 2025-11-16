# ferramentas.py

import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool

# ---------------------------------------------------------
# 1) LLM único (serve tanto para as ferramentas quanto para o agente)
# ---------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY não encontrada. Configure no .env ou em secrets do Streamlit.")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # modelo menor e mais rápido
    max_tokens=512,
    temperature=0,
)

# Parser de saída em texto puro
parser = StrOutputParser()


# ---------------------------------------------------------
# 2) Função auxiliar para criar cadeias LLM → texto
# ---------------------------------------------------------
def criar_cadeia_resposta(template: str):
    prompt = PromptTemplate(
        template=template,
        input_variables=["pergunta", "conteudo"]
    )
    return prompt | llm | parser


# ---------------------------------------------------------
# 3) Fábrica de ferramentas, com DF em closure
# ---------------------------------------------------------
def criar_ferramentas(df: pd.DataFrame):
    """
    Cria as ferramentas ligadas ao DataFrame df e as retorna como uma lista de Tool.
    """

    # Para reduzir tokens, usamos apenas uma amostra (5 linhas x 10 colunas)
    df_sample = df.iloc[:5, : min(df.shape[1], 10)]
    df_head = df_sample.to_markdown(index=False)

    info_basica = f"""
    Linhas: {df.shape[0]}
    Colunas: {df.shape[1]}

    Tipos de dados:
    {df.dtypes.to_string()}
    """

    # =========================
    # 3.1 – Ferramenta: Informações gerais
    # =========================
    cadeia_info = criar_cadeia_resposta(
        """
Você é um analista de dados encarregado de gerar um RELATÓRIO GERAL
sobre uma base de dados, a partir da seguinte pergunta do usuário:

Pergunta: {pergunta}

A seguir estão informações sobre a base de dados:

AMOSTRA (head):
=============
{conteudo}

Responda em português, de forma organizada, com seções como:
- Visão geral da base
- Estrutura das colunas
- Possíveis problemas nos dados
- Sugestões de próximas análises

Não invente colunas ou informações que não existam.
"""
    )

    def _informacoes_dataframe(pergunta: str) -> str:
        conteudo = f"{info_basica}\n\nHead do DataFrame:\n{df_head}"
        return cadeia_info.invoke({"pergunta": pergunta, "conteudo": conteudo})

    ferramenta_informacoes_dataframe = Tool(
        name="Informações DataFrame",
        func=_informacoes_dataframe,
        description=(
            "Gera um relatório com informações gerais sobre o DataFrame, "
            "como número de linhas/colunas, tipos de dados e possíveis problemas."
        ),
    )

    # =========================
    # 3.2 – Ferramenta: Resumo estatístico
    # =========================
    try:
        estatisticas_descritivas = df.describe(include="number").transpose().to_string()
    except Exception as e:
        estatisticas_descritivas = f"Não foi possível calcular describe(): {e}"

    cadeia_estatisticas = criar_cadeia_resposta(
        """
Você é um analista de dados encarregado de interpretar ESTATÍSTICAS DESCRITIVAS
de uma base de dados, a partir da seguinte pergunta do usuário:

Pergunta: {pergunta}

A seguir estão as estatísticas descritivas (apenas colunas numéricas):

=============
{conteudo}
=============

Explique, em português e de forma didática:
- A distribuição das variáveis (média, mediana, desvio padrão etc.)
- Possíveis outliers
- Insights relevantes
- Sugestões de próximos passos de análise

Não invente medidas que não estejam na tabela.
"""
    )

    def _resumo_estatistico(pergunta: str) -> str:
        conteudo = estatisticas_descritivas
        return cadeia_estatisticas.invoke({"pergunta": pergunta, "conteudo": conteudo})

    ferramenta_resumo_estatistico = Tool(
        name="Resumo Estatístico",
        func=_resumo_estatistico,
        description=(
            "Gera um relatório interpretando as estatísticas descritivas "
            "das colunas numéricas do DataFrame."
        ),
    )

    # =========================
    # 3.3 – Ferramenta: Geração de gráfico (versão simples)
    # =========================
    def _gerar_grafico(pergunta: str) -> str:
        """
        Implementação simplificada:
        - Espera frases do tipo: 'Gráfico de barras de tempo_entrega por clima'
        - Você pode ir refinando depois conforme a necessidade.
        """
        # Exemplo didático: se a pergunta contém 'tempo_entrega' e 'clima'
        if "tempo_entrega" in pergunta and "clima" in pergunta and \
           "barra" in pergunta.lower():
            fig, ax = plt.subplots()
            df.groupby("clima")["tempo_entrega"].mean().plot(kind="bar", ax=ax)
            ax.set_title("Média de tempo de entrega por clima")
            ax.set_ylabel("Tempo de entrega (médio)")
            st.pyplot(fig)
            return "Gráfico gerado: média de tempo_entrega por clima."

        # Caso contrário, apenas informa que não entendeu
        return (
            "Ainda não consigo interpretar essa solicitação de gráfico automaticamente. "
            "Tente algo como: 'Gráfico de barras de tempo_entrega por clima'."
        )

    ferramenta_gerar_grafico = Tool(
        name="Gerar Gráfico",
        func=_gerar_grafico,
        description=(
            "Gera gráficos simples com base na pergunta do usuário. "
            "Exemplo suportado: 'Gráfico de barras de tempo_entrega por clima'."
        ),
    )

    # =========================
    # 3.4 – Ferramenta: Execução de código Python seguro
    # =========================
    ferramenta_codigos_python = PythonAstREPLTool(
        name="Códigos Python",
        description=(
            "Execute código Python sobre o DataFrame df já carregado. "
            "Use esta ferramenta apenas quando precisar de cálculos específicos. "
            "Não envie perguntas em linguagem natural; envie diretamente o código."
        ),
        locals={"df": df},
        handle_tool_error=True,
    )

    return [
        ferramenta_informacoes_dataframe,
        ferramenta_resumo_estatistico,
        ferramenta_gerar_grafico,
        ferramenta_codigos_python,
    ]