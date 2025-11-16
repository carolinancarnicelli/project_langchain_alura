# ferramentas.py

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool

parser = StrOutputParser()


def criar_cadeia_resposta(template: str, llm):
    """Cria uma cadeia simples: PromptTemplate -> LLM -> StrOutputParser."""
    prompt = PromptTemplate(
        template=template,
        input_variables=["pergunta", "conteudo"],
    )
    return prompt | llm | parser


def criar_ferramentas(df: pd.DataFrame, llm):
    """
    Cria as ferramentas ligadas ao DataFrame df e ao LLM informado.
    Retorna uma lista de Tool (para ser usada no agente e nas ações rápidas).
    """

    # Amostra pequena para evitar prompt gigante
    df_sample = df.iloc[:5, : min(df.shape[1], 10)]
    df_head = df_sample.to_markdown(index=False)

    info_basica = f"""
    Linhas: {df.shape[0]}
    Colunas: {df.shape[1]}

    Tipos de dados:
    {df.dtypes.to_string()}
    """

    # ======================================================
    # 1) Ferramenta: Informações gerais do DataFrame
    # ======================================================
    template_info = """
Você é um analista de dados encarregado de gerar um RELATÓRIO GERAL
sobre uma base de dados, a partir da seguinte pergunta do usuário:

Pergunta: {pergunta}

A seguir estão informações sobre a base de dados:

INFORMAÇÕES BÁSICAS:
====================
{conteudo}

Responda em português, de forma organizada, com seções como:
- Visão geral da base
- Estrutura das colunas
- Possíveis problemas nos dados
- Sugestões de próximas análises

Não invente colunas ou informações que não existam.
"""
    cadeia_info = criar_cadeia_resposta(template_info, llm)

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

    # ======================================================
    # 2) Ferramenta: Resumo estatístico
    # ======================================================
    try:
        estatisticas_descritivas = df.describe(include="number").transpose().to_string()
    except Exception as e:
        estatisticas_descritivas = f"Não foi possível calcular describe(): {e}"

    template_est = """
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
    cadeia_estatisticas = criar_cadeia_resposta(template_est, llm)

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

    # ======================================================
    # 3) Ferramenta: Gerar gráfico
    # ======================================================
    def _gerar_grafico(pergunta: str) -> str:
        """
        Implementação simples e específica para o curso:

        - Se a pergunta mencionar 'tempo_entrega' e 'clima',
          gera um gráfico de barras da média de tempo_entrega por clima,
          ordenando do menor valor para o maior.

        Caso contrário, retorna uma mensagem de orientação.
        """

        if "tempo_entrega" in pergunta and "clima" in pergunta:
            if "tempo_entrega" not in df.columns or "clima" not in df.columns:
                return (
                    "Não encontrei as colunas 'tempo_entrega' e 'clima' no DataFrame. "
                    "Verifique os nomes das colunas."
                )

            try:
                medias = (
                    df.groupby("clima")["tempo_entrega"]
                    .mean()
                    .sort_values(ascending=True)
                )

                fig, ax = plt.subplots()
                medias.plot(kind="bar", ax=ax)
                ax.set_title("Média de tempo_entrega por clima")
                ax.set_ylabel("tempo_entrega (médio)")
                ax.set_xlabel("clima")
                st.pyplot(fig)

                return (
                    "Gráfico gerado: média de tempo_entrega por clima, "
                    "ordenada do menor valor para o maior."
                )
            except Exception as e:
                return f"Não consegui gerar o gráfico por causa do erro: {e}"

        return (
            "Ainda não sei montar esse tipo de gráfico automaticamente. "
            "Tente algo como: 'Crie um gráfico da média de tempo_entrega por clima.'"
        )

    ferramenta_gerar_grafico = Tool(
        name="Gerar Gráfico",
        func=_gerar_grafico,
        description=(
            "Gera gráficos simples com base na pergunta do usuário. "
            "Exemplo suportado: 'Gráfico da média de tempo_entrega por clima'."
        ),
    )

    # ======================================================
    # 4) Ferramenta: Execução de códigos Python
    # ======================================================
    ferramenta_codigos_python = PythonAstREPLTool(
        name="Códigos Python",
        description=(
            "Execute código Python sobre o DataFrame df já carregado. "
            "Use esta ferramenta para cálculos específicos, por exemplo: "
            "df.groupby('clima')['tempo_entrega'].mean(). "
            "Não envie perguntas em linguagem natural; envie diretamente o código."
        ),
        locals={"df": df, "pd": pd},
        handle_tool_error=True,
    )

    return [
        ferramenta_informacoes_dataframe,
        ferramenta_resumo_estatistico,
        ferramenta_gerar_grafico,
        ferramenta_codigos_python,
    ]
