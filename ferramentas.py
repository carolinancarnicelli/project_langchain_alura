import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain.agents import Tool
from langchain_experimental.tools import PythonAstREPLTool


# Obtenção da chave de api
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configurações do LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# Relatório informações
@tool
def informacoes_dataframe(pergunta: str, df: pd.DataFrame) -> str:
    """Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
        incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
        nulos e duplicados para dar um panomara geral sobre o arquivo."""

    # Coleta de informações
    shape = df.shape
    columns = df.dtypes
    nulos = df.isnull().sum()
    nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
    duplicados = df.duplicated().sum()

   # Prompt de resposta 

    template_resposta = PromptTemplate( 

        template=""" 
        Você é um analista de dados encarregado de apresentar um resumo informativo sobre um DataFrame 
        a partir de uma {pergunta} feita pelo usuário. 

        A seguir, você encontrará as informações gerais da base de dados: 

        ================= INFORMAÇÕES DO DATAFRAME ================= 

        Dimensões: {shape}
 
        Colunas e tipos de dados: {columns} 

        Valores nulos por coluna: {nulos} 

        Strings 'nan' (qualquer capitalização) por coluna: {nans_str} 

        Linhas duplicadas: {duplicados} 

        ============================================================ 

        Com base nessas informações, escreva um resumo claro e organizado contendo: 

        1. Um título: ## Relatório de informações gerais sobre o dataset 
        2. A dimensão total do DataFrame; 
        3. A descrição de cada coluna (incluindo nome, tipo de dado e o que aquela coluna é) 
        4. As colunas que contêm dados nulos, com a respectiva quantidade.  
        5. As colunas que contêm strings 'nan', com a respectiva quantidade. 
        6. E a existência (ou não) de dados duplicados. 
        7. Escreva um parágrafo sobre análises que podem ser feitas com 
        esses dados. 
        8. Escreva um parágrafo sobre tratamentos que podem ser feitos nos dados. 
        """, 
        input_variables=["pergunta","shape", "columns", "nulos", "nans_str", "duplicados"] ) 

    cadeia = template_resposta | llm | StrOutputParser()

    resposta = cadeia.invoke({
              "pergunta": pergunta,
              "shape": shape,
              "columns": columns,
              "nulos": nulos,
              "nans_str": nans_str,
              "duplicados": duplicados
        })

    return resposta

# Relatório estatístico
@tool
def resumo_estatistico(pergunta: str, df: pd.DataFrame) -> str:
    """
    Gera um relatório textual com base em estatísticas descritivas
    das colunas numéricas do DataFrame.
    """

    # 1) Colunas que já são numéricas
    df_num = df.select_dtypes(include="number")

    # 2) Tentar converter automaticamente colunas numéricas que estejam como texto
    possiveis_numericas = {}

    for col in df.columns:
        if col in df_num.columns:
            continue

        serie_original = df[col]

        serie = (
            serie_original.astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        conv = pd.to_numeric(serie, errors="coerce")

        # entra se tiver ao menos 1 valor numérico
        if conv.notna().sum() > 0:
            possiveis_numericas[col] = conv

    if possiveis_numericas:
        df_convertidas = pd.DataFrame(possiveis_numericas)
        if not df_num.empty:
            df_num = pd.concat([df_num, df_convertidas], axis=1)
        else:
            df_num = df_convertidas

    # 3) Se ainda não tiver colunas numéricas, retorna mensagem amigável
    if df_num.empty:
        return (
            "Não foi possível gerar estatísticas descritivas numéricas, "
            "porque nenhuma coluna foi identificada como numérica neste arquivo. "
            "Verifique se as colunas de quantidade/valor estão em formato numérico "
            "ou se vêm como texto com vírgula, ponto etc."
        )

    # 4) Gera o describe com segurança
    try:
        estatisticas_descritivas = df_num.describe().transpose().to_string()
    except ValueError as e:
        return (
            "Ocorreu um erro ao gerar as estatísticas descritivas numéricas "
            f"(detalhe técnico: {e}). Isso geralmente acontece quando não há "
            "dados numéricos suficientes após o tratamento."
        )

    # 5) Prompt de resposta (igual à aula)
    template_resposta = PromptTemplate(
        template="""
        Você é um analista de dados encarregado de interpretar resultados estatísticos de uma base de dados
        a partir de uma {pergunta} feita pelo usuário.

        A seguir, você encontrará as estatísticas descritivas da base de dados:

        ================= ESTATÍSTICAS DESCRITIVAS =================

        {resumo}

        ============================================================

        Com base nesses dados, elabore um resumo explicativo com linguagem clara, acessível e fluida, destacando
        os principais pontos dos resultados. Inclua:

        1. Um título: ## Relatório de estatísticas descritivas
        2. Uma visão geral das estatísticas das colunas numéricas
        3. Um paráfrago sobre cada uma das colunas, comentando informações sobre seus valores.
        4. Identificação de possíveis outliers com base nos valores mínimo e máximo
        5. Recomendações de próximos passos na análise com base nos padrões identificados
        """,
        input_variables=["pergunta", "resumo"],
    )

    cadeia = template_resposta | llm | StrOutputParser()
    resposta = cadeia.invoke(
        {"pergunta": pergunta, "resumo": estatisticas_descritivas}
    )

    return resposta


# Ferramenta genérica para criação de gráficos
@tool
def gerar_grafico(pergunta: str, df: pd.DataFrame) -> str:
    """
    Gera gráficos a partir do DataFrame `df` com base em uma pergunta em linguagem natural.
    O LLM é usado apenas para escolher colunas/medida/tipo de gráfico.
    O agrupamento e o plot são feitos 100% em Python, usando o df completo.
    """

    # 1) Monta string com nomes de colunas
    colunas_lista = list(df.columns)
    colunas_str = "\n".join(f"- {c}" for c in colunas_lista)

    # 2) Prompt para o modelo devolver apenas JSON com a configuração do gráfico
    prompt_cfg = PromptTemplate(
        template="""
        Você recebe uma pergunta do usuário sobre um gráfico que deve ser feito
        a partir de um DataFrame pandas chamado `df`.

        O DataFrame possui as seguintes colunas:
        {colunas}

        Pergunta do usuário:
        "{pergunta}"

        Sua tarefa é escolher:
        - qual coluna será usada no eixo X (`x_col`);
        - qual coluna será usada no eixo Y (`y_col`), se houver;
        - qual agregação usar (`agg`): "sum", "mean", "count" ou "none";
        - qual tipo de gráfico é mais adequado (`chart_type`): "bar", "line", "scatter", "hist";
        - opcionalmente, quantas categorias máximas mostrar (`top_n`), por exemplo 20.

        Restrições IMPORTANTES:
        - Escolha apenas nomes de colunas que existam na lista fornecida.
        - Se a pergunta falar "soma de X por Y", use `agg = "sum"`, `y_col = X` e `x_col = Y`.
        - Se a pergunta falar "contagem de registros por X", use `agg = "count"` e deixe `y_col = null`.
        - Se não ficar claro, escolha um padrão razoável (por exemplo, count por uma coluna categórica).

        Responda APENAS com um JSON válido, sem texto extra, no formato:

        {{
          "x_col": "...",
          "y_col": "... ou null",
          "agg": "sum|mean|count|none",
          "chart_type": "bar|line|scatter|hist",
          "top_n": 20
        }}
        """,
        input_variables=["pergunta", "colunas"],
    )

    cadeia_cfg = prompt_cfg | llm | StrOutputParser()
    cfg_str = cadeia_cfg.invoke({"pergunta": pergunta, "colunas": colunas_str})

    try:
        cfg = json.loads(cfg_str)
    except json.JSONDecodeError:
        st.error("Não consegui interpretar a configuração de gráfico retornada pelo modelo.")
        st.text(cfg_str)
        return ""

    x_col = cfg.get("x_col")
    y_col = cfg.get("y_col")
    agg = (cfg.get("agg") or "sum").lower()
    chart_type = (cfg.get("chart_type") or "bar").lower()
    top_n = cfg.get("top_n")

    # 3) Validação básica das colunas
    if x_col not in df.columns:
        st.error(f"Coluna para eixo X não encontrada no DataFrame: {x_col}")
        return ""

    if y_col is not None and y_col not in df.columns:
        st.error(f"Coluna para eixo Y não encontrada no DataFrame: {y_col}")
        return ""

    # 4) Preparar dados agregados
    df_plot = df.copy()

    # Se for soma/média, tentar garantir que y_col seja numérica
    if y_col is not None and agg in ["sum", "mean"]:
        serie = (
            df_plot[y_col]
            .astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        df_plot[y_col] = pd.to_numeric(serie, errors="coerce")

    if agg in ["sum", "mean"] and y_col is not None:
        dados = (
            df_plot.groupby(x_col)[y_col]
            .agg(agg)
            .sort_values(ascending=False)
        )
    elif agg == "count":
        # contagem de linhas por categoria de x_col
        dados = df_plot.groupby(x_col).size().sort_values(ascending=False)
    else:
        # sem agregação: usa a coluna original (ex. hist de uma coluna numérica)
        dados = None

    # Se houver agregação, opcionalmente limita Top N
    if dados is not None and isinstance(top_n, int) and top_n > 0:
        dados = dados.head(top_n)

    # 5) Plotar com seaborn/matplotlib
    plt.figure(figsize=(14, 6))
    sns.set_theme()

    if dados is not None:
        # temos série agregada (index = categorias, values = métrica)
        x_vals = dados.index
        y_vals = dados.values

        if chart_type in ["bar", "barplot"]:
            sns.barplot(x=x_vals, y=y_vals)
        elif chart_type in ["line", "lineplot"]:
            sns.lineplot(x=x_vals, y=y_vals, marker="o")
        else:
            # fallback para barplot
            sns.barplot(x=x_vals, y=y_vals)

        plt.xlabel(x_col)
        if y_col is not None:
            plt.ylabel(f"{agg} de {y_col}")
        else:
            plt.ylabel("Contagem de registros")

    else:
        # caso sem agregação (por ex. hist de uma coluna numérica)
        if y_col is None:
            st.error("Configuração de gráfico inválida: agg='none' mas 'y_col' não foi definida.")
            return ""

        serie = (
            df_plot[y_col]
            .astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        valores = pd.to_numeric(serie, errors="coerce").dropna()

        if chart_type in ["hist", "histplot"]:
            sns.histplot(valores, bins=30, kde=True)
        else:
            sns.histplot(valores, bins=30, kde=True)

        plt.xlabel(y_col)
        plt.ylabel("Frequência")

    plt.title(pergunta, loc="left", pad=20, fontsize=14)
    plt.xticks(rotation=90)
    sns.despine()

    st.pyplot(plt.gcf())
    return ""

# Função para criar ferramentas 
def criar_ferramentas(df):
    ferramenta_informacoes_dataframe = Tool(
        name="Informações Dataframe",
        func=lambda pergunta:informacoes_dataframe.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
        incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados
        nulos e duplicados para dar um panomara geral sobre o arquivo.""",
        return_direct=True) # Para exibir o relatório gerado pela função

    ferramenta_resumo_estatistico = Tool(
        name="Resumo Estatístico",
        func=lambda pergunta:resumo_estatistico.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo e descritivo da base de dados,
        incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.) e/ou múltiplas colunas numéricas.
        Não utilize esta ferramenta para calcular uma única métrica como 'qual é a média de X' ou 'qual a correlação das variáveis'.
        Para isso, use a ferramenta_python.""",
        return_direct=True) # Para exibir o relatório gerado pela função

    ferramenta_gerar_grafico = Tool(
        name="Gerar Gráfico",
        func=lambda pergunta:gerar_grafico.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas (`df`) com base em uma instrução do usuário.
        A instrução pode conter pedidos como: 'Crie um gráfico da média de tempo de entrega por clima','Plote a distribuição do tempo de entrega'"
        ou "Plote a relação entre a classificação dos agentes e o tempo de entrega. Palavras-chave comuns que indicam o uso desta ferramenta incluem:
        'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição', 'represente graficamente', entre outros.""",
        return_direct=True)
    
    ferramenta_codigos_python = Tool(
        name="Códigos Python",
        func=PythonAstREPLTool(locals={"df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar cálculos, consultas ou transformações específicas usando Python diretamente sobre o DataFrame `df`.
        Exemplos de uso incluem: "Qual é a média da coluna X?", "Quais são os valores únicos da coluna Y?", "Qual a correlação entre A e B?". 
        Evite utilizar esta ferramenta para solicitações mais amplas ou descritivas, como informações gerais sobre o dataframe, resumos estatísticos completos ou geração de gráficos — nesses casos, use as ferramentas apropriadas.""")

    return [
        ferramenta_informacoes_dataframe, 
        ferramenta_resumo_estatistico, 
        ferramenta_gerar_grafico,
        ferramenta_codigos_python

    ]






