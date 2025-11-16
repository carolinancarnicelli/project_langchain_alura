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


# >>> CORREÇÃO: carregar variáveis do .env <<<
load_dotenv()

# Opcional: se quiser inspecionar rapidamente
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# print("DEBUG GROQ_API_KEY:", GROQ_API_KEY)  # só para teste local

# >>> CORREÇÃO: deixar ChatGroq usar GROQ_API_KEY do ambiente <<<
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",  # modelo menor e mais barato
    max_tokens=512,
    temperature=0,
    # Se quiser forçar explicitamente:
    # groq_api_key=GROQ_API_KEY,
)

# Relatório informações
@tool

def informacoes_dataframe(pergunta: str, df: pd.DataFrame) -> str:
  """Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o DataFrame,
  incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados nulos
  e duplicados para dar um panorama geral sobre o arquivo."""

# Coleta de informações

  shape = df.shape
  columns = df.dtypes
  nulos = df.isnull().sum()
  nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
  duplicados = df.duplicated().sum()

# Prompt de resposta

  template_resposta = PromptTemplate(
      template = """
        Você é um analista de dados encarregado de apresentar um resumo informativo sobre um DataFrame
        a partir de uma {pergunta} feita pelo usuário.

        A seguir, você encontrará as informações gerais da base de dados:

        ================= INFORMAÇÕES DO DATAFRAME =================

        Dimensões: {shape}

        Colunas e tipos de dados:
        {columns}

        Valores nulos por coluna:
        {nulos}

        Strings 'nan' (qualquer capitalização) por coluna:
        {nans_str}

        Linhas duplicadas: {duplicados}

        ============================================================

        Com base nessas informações, escreva um resumo claro e organizado contendo:
        1. Um título: ## Relatório de informações gerais sobre o dataset,
        2. A dimensão total do DataFrame;
        3. A descrição de cada coluna (incluindo nome, tipo de dado e o que aquela coluna é),
        4. As colunas que contêm dados nulos, com a respectiva quantidade;
        5. As colunas que contêm strings 'nan', com a respectiva quantidade;
        6. E a existência (ou não) de dados duplicados;
        7. Escreva um parágrafo sobre análises que podem ser feitas com
        esses dados;
        8. Escreva um parágrafo sobre tratamentos que podem ser feitos nos dados.""",
      input_variables=["pergunta", "shape", "columns", "nulos", "nans_str", "duplicados"]
  )

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
  """Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo
  e descritivo da base de dados, incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.)."""

# Coleta de estatisticas descritivas

  estatisticas_descritivas = df.describe(include='number').transpose().to_string()

# Prompt de resposta

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

        1. Um título: ## Relatório de estatísticas descritivas;
        2. Uma visão geral das estatísticas das colunas numéricas;
        3. Um parágrafo sobre cada uma das colunas, comentando informações sobre seus valores;
        4. Identificação de possíveis outliers com base nos valores mínimo e máximo;
        5. Recomendações de próximos passos na análise com base nos padrões identificados.
""", input_variables=["pergunta", "resumo"]
  )

  cadeia = template_resposta | llm | StrOutputParser()

  resposta = cadeia.invoke({"pergunta": pergunta, "resumo": estatisticas_descritivas})

  return resposta

# Gerador de gráficos
@tool

def gerar_grafico(pergunta: str, df: pd.DataFrame) -> str:
    """
    Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame pandas (`df`)
    com base em uma instrução do usuário.
    """

    # 1) Informações sobre o dataframe
    colunas_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
    # Usar uma visualização segura da amostra (markdown), não dict literal
    amostra_markdown = df.head(3).to_markdown(index=False)

    # 2) Prompt mais restritivo para o LLM
    template_resposta = PromptTemplate(
        template="""
Você é um analista de dados responsável por gerar código Python para criar um gráfico
a partir de um DataFrame pandas chamado `df`, com base na seguinte solicitação do usuário:

Pergunta do usuário:
{pergunta}

Abaixo estão informações sobre o DataFrame disponível:

================= INFORMAÇÕES DO DATAFRAME =================
Colunas e tipos de dados:
{colunas}

Amostra dos dados (3 primeiras linhas, em formato de tabela):
{amostra}
============================================================

Instruções IMPORTANTES:

1. NÃO crie dados de exemplo. NÃO declare listas ou dicionários como:
   `dados = [{{'ID_pedido': ...}}]` ou algo semelhante.
2. NÃO crie um novo DataFrame. NÃO faça `df = pd.DataFrame(...)`.
   Use APENAS o DataFrame `df` que já existe no ambiente.
3. Use obrigatoriamente as bibliotecas já importadas:
   - `matplotlib.pyplot` como `plt`
   - `seaborn` como `sns`
4. Comece o código configurando um estilo, por exemplo:
   `sns.set_theme()`
5. Escolha o tipo de gráfico adequado à pergunta do usuário. Exemplos:
   - Distribuição de variável numérica: `sns.histplot`, `sns.kdeplot`, `sns.boxplot`
   - Distribuição de variável categórica: `sns.countplot`
   - Comparação entre categorias: `sns.barplot`
   - Relação entre duas variáveis: `sns.scatterplot`
   - Séries temporais: `sns.lineplot` com eixo X em datas
6. Crie a figura e o eixo, por exemplo:
   `plt.figure(figsize=(8, 4))`
7. Configure título e rótulos dos eixos com `plt.title`, `plt.xlabel`, `plt.ylabel`.
8. Posicione o título à esquerda com `loc='left'` e `pad=20`, `fontsize=14`.
9. Mantenha os ticks do eixo X sem rotação (`plt.xticks(rotation=0)`), a menos que
   seja estritamente necessário para visualização.
10. Remova as bordas superior e direita com `sns.despine()`.
11. Finalize SEMPRE com `plt.tight_layout()` e `plt.show()`.

Retorne APENAS o código Python, sem nenhum texto adicional, explicação ou comentário.
Não use aspas triplas, não coloque o código dentro de string, não use ```.

Código Python:
""",
        input_variables=["pergunta", "colunas", "amostra"],
    )

    cadeia = template_resposta | llm | StrOutputParser()

    codigo_bruto = cadeia.invoke(
        {
            "pergunta": pergunta,
            "colunas": colunas_info,
            "amostra": amostra_markdown,
        }
    )

    # Limpeza básica de marcações que o modelo possa ter colocado
    codigo_limpo = (
        codigo_bruto.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    exec_globals = {"df": df, "plt": plt, "sns": sns}
    exec_locals = {}

    try:
        exec(codigo_limpo, exec_globals, exec_locals)
    except SyntaxError as e:
        # Opcional: retornar uma mensagem mais amigável
        raise ValueError(
            f"Código gerado pelo modelo é inválido (SyntaxError).\n\n"
            f"Código gerado:\n{codigo_limpo}\n\nErro: {e}"
        )

    fig = plt.gcf()
    st.pyplot(fig)

    return ""


# Função para criar ferramentas
def criar_ferramentas(df):
    ferramenta_informacoes_dataframe = Tool(
        name="Informações DataFrame",
        func=lambda pergunta:informacoes_dataframe.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar informações gerais sobre o dataframe,
        incluindo número de colunas e linhas, nomes das colunas e seus tipos de dados, contagem de dados nulos e
        duplicados para dar um panorama geral sobre o arquivo.""",
        return_direct=True
    ) # Para exibir o relatório gerado pela função

    ferramenta_resumo_estatistico = Tool(
        name="Resumo Estatístico",
        func=lambda pergunta:resumo_estatistico.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar um resumo estatístico completo
        e descritivo da base de dados, incluindo várias estatísticas (média, desvio padrão, mínimo, máximo etc.).
        Não utilize esta ferramenta para calcular uma única métrica como 'qual é a média de X' ou
        qual a correlação das variáveis'. Nesses casos, utilize a ferramenta_codigos_python.""", return_direct=True
    )

    ferramenta_gerar_grafico = Tool(
        name="Gerar Gráfico",
        func=lambda pergunta:gerar_grafico.run({"pergunta": pergunta, "df": df}),
        description="""Utilize esta ferramenta sempre que o usuário solicitar um gráfico a partir de um DataFrame
        pandas (`df`) com base em uma instrução do usuário.  A instrução pode conter pedidos como: 'Crie um gráfico
        da média de tempo de entrega por clima', 'Plote a distribuição do tempo de entrega'  ou 'Plote a relação
        entre a classificação dos agentes e o tempo de entrega'. Palavras-chave comuns que indicam o uso desta
        ferramenta incluem: 'crie um gráfico', 'plote', 'visualize', 'faça um gráfico de', 'mostre a distribuição',
        'represente graficamente', entre outros.""",  return_direct=True
    )



##    ferramenta_codigos_python = Tool(
##        name="Códigos Python",
##        func=PythonAstREPLTool(locals={"df": df}),
##        description="""Utilize esta ferramenta sempre que o usuário solicitar cálculos, consultas ou transformações
##        específicas usando Python diretamente sobre o DataFrame `df`.
##        Exemplos de uso incluem: "Qual é a média da coluna X?", "Quais são os valores únicos da coluna Y?",
##        "Qual a correlação entre A e B?". Evite utilizar esta ferramenta para solicitações mais amplas ou descritivas,
##        como informações gerais sobre o DataFrame, resumos estatísticos completos ou geração de gráficos — nesses casos,
##        use as ferramentas apropriadas."""
##    )
    
    ferramenta_codigos_python = PythonAstREPLTool(
        name="Códigos Python",
        description=(
            "Use esta ferramenta para executar CÓDIGO PYTHON diretamente sobre o DataFrame `df`.\n"
            "A ENTRADA DEVE SER APENAS CÓDIGO PYTHON VÁLIDO, sem texto explicativo.\n\n"
            "Exemplos de entrada válidos:\n"
            "- \"df['tempo_entrega'].mean()\"\n"
            "- \"df['tempo_entrega'].describe()\"\n"
            "- \"df[['tempo_entrega', 'distancia']].corr()\"\n\n"
            "NÃO envie perguntas em linguagem natural como 'Qual é a média da coluna tempo_entrega?'; "
            "em vez disso, envie diretamente o código necessário."
        ),
        locals={"df": df},
        handle_tool_error=True,   # evita que o erro estoure a cadeia; vira Observation
    )

    
    return [
        ferramenta_informacoes_dataframe,
        ferramenta_resumo_estatistico,
        ferramenta_gerar_grafico,
        ferramenta_codigos_python
    ]
