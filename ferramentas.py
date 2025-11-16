import os
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
    model_name="llama-3.3-70b-versatile",
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
    Gera um resumo estatístico das colunas numéricas do DataFrame.
    Tenta:
    1) Usar colunas já numéricas;
    2) Detectar e converter automaticamente colunas que parecem numéricas.
    """

    # 1) Colunas que já são numéricas
    df_num = df.select_dtypes(include="number")

    # 2) Se não houver colunas numéricas, tentar converter automaticamente
    if df_num.empty:
        possiveis_numericas = {}

        for col in df.columns:
            serie_original = df[col]

            # Converte para string e normaliza:
            # - remove espaços
            # - remove separador de milhar "."
            # - troca vírgula decimal "," por ponto "."
            serie = (
                serie_original.astype(str)
                .str.strip()
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )

            conv = pd.to_numeric(serie, errors="coerce")

            # Se pelo menos 50% dos valores viram número, consideramos a coluna numérica
            if conv.notna().mean() >= 0.5:
                possiveis_numericas[col] = conv

        if possiveis_numericas:
            df_num = pd.DataFrame(possiveis_numericas)

    # 3) Se ainda assim não tiver colunas numéricas, devolve mensagem amigável
    if df_num.empty:
        return (
            "Não foi possível gerar estatísticas descritivas numéricas, "
            "porque nenhuma coluna foi identificada como numérica neste arquivo. "
            "Verifique se as colunas de quantidade/valor estão realmente em formato numérico "
            "ou se vêm como texto com vírgula, ponto, etc."
        )

    # 4) Agora sim, gera o describe com segurança
    estatisticas_descritivas = df_num.describe().transpose().to_string()

    resposta = (
        "Segue o resumo estatístico das colunas numéricas identificadas no dataset:\n\n"
        f"{estatisticas_descritivas}"
    )
    return resposta


# Gerador de gráficos 
@tool
def gerar_grafico(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas (`df`) com base em uma instrução do usuário.
    A instrução pode conter pedidos como: 'Crie um gráfico da média de tempo de entrega por clima','Plote a distribuição do tempo de entrega'"
    ou "Plote a relação entre a classifição dos agentes e o tempo de entrega. Palavras-chave comuns que indicam o uso desta ferramenta incluem:
    'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição', 'represente graficamente', entre outros."""

 # Captura informações sobre o dataframe
    colunas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
    amostra_dados = df.head(3).to_dict(orient='records')

  # Template otimizado para geração de código de gráficos
    template_resposta = PromptTemplate(
            template="""
            Você é um especialista em visualização de dados. Sua tarefa é gerar **apenas o código Python** para plotar um gráfico com base na solicitação do usuário.

            ## Solicitação do usuário:
            "{pergunta}"

            ## Metadados do DataFrame:
            {colunas}

            ## Amostra dos dados (3 primeiras linhas):
            {amostra}

            ## Instruções obrigatórias:
            1. Use as bibliotecas `matplotlib.pyplot` (como `plt`) e `seaborn` (como `sns`).
            2. Defina o tema com `sns.set_theme()`
            3. Certifique-se de que todas as colunas mencionadas na solicitação existem no DataFrame chamado `df`.
            4. Escolha o tipo de gráfico adequado conforme a análise solicitada:
            - **Distribuição de variáveis numéricas**: `histplot`, `kdeplot`, `boxplot` ou `violinplot`
            - **Distribuição de variáveis categóricas**: `countplot` 
            - **Comparação entre categorias**: `barplot`
            - **Relação entre variáveis**: `scatterplot` ou `lineplot`
            - **Séries temporais**: `lineplot`, com o eixo X formatado como datas
            5. Configure o tamanho do gráfico com `figsize=(8, 4)`.
            6. Adicione título e rótulos (`labels`) apropriados aos eixos.
            7. Posicione o título à esquerda com `loc='left'`, deixe o `pad=20` e use `fontsize=14`.
            8. Mantenha os ticks eixo X sem rotação com `plt.xticks(rotation=0)`
            9. Remova as bordas superior e direita do gráfico com `sns.despine()`.
            10. Finalize o código com `plt.show()`.

            Retorne APENAS o código Python, sem nenhum texto adicional ou explicação.

            Código Python:
            """, input_variables=["pergunta", "colunas", "amostra"]
        )

        # Gera o código
    cadeia = template_resposta | llm | StrOutputParser()
    codigo_bruto = cadeia.invoke({
            "pergunta": pergunta,
            "colunas": colunas_info,
            "amostra": amostra_dados
        })

        # Limpa o código gerado
    codigo_limpo = codigo_bruto.replace("```python", "").replace("```", "").strip()

        # Tenta executar o código para validação
    exec_globals = {'df': df, 'plt': plt, 'sns': sns}
    exec_locals = {}
    exec(codigo_limpo, exec_globals, exec_locals)

        # Mostra o gráfico 
    fig = plt.gcf()
    st.pyplot(fig)
        
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

