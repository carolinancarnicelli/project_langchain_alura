import streamlit as st
import pandas as pd
import os
from pandas.errors import ParserError
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from ferramentas import criar_ferramentas
import groq

# Função para detectar a linha de cabeçalho
def detectar_linha_cabecalho(df_raw: pd.DataFrame, max_linhas: int = 50) -> int:
    """
    Detecta a linha de cabeçalho procurando a PRIMEIRA linha
    em que praticamente todas as colunas estejam preenchidas.

    Regra principal:
    - primeira linha (entre as `max_linhas` primeiras) em que
      número de colunas não vazias >= 90% do total.

    Se não achar nenhuma, usa a linha com maior preenchimento
    (>= 60% das colunas). Como último recurso, devolve a
    primeira linha do subset.
    """
    if df_raw.empty:
        return 0

    total_cols = df_raw.shape[1]
    subset = df_raw.head(max_linhas)

    candidatos = []

    for idx, row in subset.iterrows():
        valores = row.astype(str).str.strip()

        # considera vazio: "", "nan", "NaN", "none"
        mask_nao_vazio = ~valores.isin(["", "nan", "NaN", "NONE", "None"])
        qtd_nao_vazios = int(mask_nao_vazio.sum())

        candidatos.append((idx, qtd_nao_vazios))

        # Regra principal: pelo menos 90% das colunas preenchidas
        if qtd_nao_vazios >= int(0.9 * total_cols):
            return idx

    # Fallback 1: linha com maior número de colunas não vazias,
    # desde que tenha pelo menos 60% preenchidas
    melhor_idx, melhor_qtd = max(candidatos, key=lambda x: x[1])
    if melhor_qtd >= int(0.6 * total_cols):
        return melhor_idx

    # Fallback 2: primeira linha do subset
    return subset.index[0]


def carregar_csv_flexivel(arquivo) -> pd.DataFrame:
    """
    Lê o CSV:
    - detecta separador automaticamente (vírgula, ponto-e-vírgula etc.)
    - ignora linhas quebradas
    - detecta linha de cabeçalho que pode NÃO estar na 1ª linha
    """
    # 1) Ler tudo sem cabeçalho
    try:
        df_raw = pd.read_csv(
            arquivo,
            sep=None,             # autodetecta separador
            engine="python",
            on_bad_lines="skip",
            header=None,
        )
    except UnicodeDecodeError:
        # tenta novamente com latin1 (muito comum nesses relatórios)
        arquivo.seek(0)
        df_raw = pd.read_csv(
            arquivo,
            sep=None,
            engine="python",
            on_bad_lines="skip",
            header=None,
            encoding="latin1",
        )

    if df_raw.empty:
        raise ValueError("Arquivo CSV sem dados.")

    # 2) Detectar a linha de cabeçalho
    header_idx = detectar_linha_cabecalho(df_raw)

    # DEBUG opcional: ver qual linha foi detectada
    # st.write("Linha de cabeçalho detectada:", header_idx)
    # st.dataframe(df_raw.head(20))

    # 3) Usar essa linha como cabeçalho e remover as linhas anteriores
    colunas = df_raw.iloc[header_idx].astype(str).str.strip().tolist()
    df = df_raw[(header_idx + 1):].copy()
    df.columns = colunas
    df = df.reset_index(drop=True)

    return df


# Inicia o app
st.set_page_config(page_title="Assistente de análise de dados com IA", layout="centered")
st.title("🦜 Assistente de análise de dados com IA")

# Descrição da ferramenta
st.info("""
Este assistente utiliza um agente, criado com Langchain, para te ajudar a explorar, analisar e visualizar dados de forma interativa.
Basta fazer o upload de um arquivo CSV e você poderá:

- 📄 **Gerar relatórios automáticos**:
    - **Relatório de informações gerais**: apresenta a dimensão do DataFrame, nomes e tipos das colunas, contagem de dados nulos e duplicados, além de sugestões de tratamentos e análises adicionais.
    - **Relatório de estatísticas descritivas**: exibe valores como média, mediana, desvio padrão, mínimo e máximo; identifica possíveis outliers e sugere próximos passos com base nos padrões detectados.

- 🔎 **Fazer perguntas simples sobre os dados**: como "Qual é a média da coluna X?", "Quantos registros existem para cada categoria da coluna Y?".
                
- 📊 **Criar gráficos automaticamente** com base em perguntas em linguagem natural.

Ideal para analistas, cientistas de dados e equipes que buscam agilidade e insights rápidos com apoio de IA.
""")

# Upload do CSV
st.markdown("### 📁 Faça upload do seu arquivo CSV")
arquivo_carregado = st.file_uploader("Faça upload de um arquivo CSV", type="csv")

df = None  # inicializa

if arquivo_carregado is not None:
    try:
        df = carregar_csv_flexivel(arquivo_carregado)
    except ParserError as e:
        st.error("Não foi possível ler o arquivo CSV (erro de parsing).")
        st.text(f"Detalhes técnicos: {e}")
        st.stop()
    except Exception as e:
        st.error("Erro ao carregar o arquivo CSV.")
        st.text(f"Detalhes técnicos: {e}")
        st.stop()

    st.success(f"Arquivo carregado com sucesso! Formato: {df.shape[0]} linhas x {df.shape[1]} colunas")
    st.dataframe(df.head())

    # LLM
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # modelo atual da Groq
    temperature=0
)


    # Ferramentas
    tools = criar_ferramentas(df)

    # Prompt react
    df_head = df.head().to_markdown()

    prompt_react_pt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"df_head": df_head},
        template="""
        Você é um assistente que sempre responde em português.

        Você tem acesso a um dataframe pandas chamado `df`.
        Aqui estão as primeiras linhas do DataFrame, obtidas com `df.head().to_markdown()`:

        {df_head}

        Responda às seguintes perguntas da melhor forma possível.

        Para isso, você tem acesso às seguintes ferramentas:

        {tools}

        Use o seguinte formato:

        Question: a pergunta de entrada que você deve responder  
        Thought: você deve sempre pensar no que fazer  
        Action: a ação a ser tomada, deve ser uma das [{tool_names}]  
        Action Input: a entrada para a ação  
        Observation: o resultado da ação  
        ... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
        Thought: Agora eu sei a resposta final  
        Final Answer: a resposta final para a pergunta de entrada original.
        Quando usar a ferramenta_python: formate sua resposta final de forma clara, em lista, com valores separados por vírgulas e duas casas decimais sempre que apresentar números.

        Comece!

        Question: {input}  
        Thought: {agent_scratchpad}"""
    )

    
    # Agente
    agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_pt)
    orquestrador = AgentExecutor(agent=agente,
                                tools=tools,
                                verbose=True,
                                handle_parsing_errors=True,
                                max_iterations=4)

    # AÇÕES RÁPIDAS
    st.markdown("---")
    st.markdown("## ⚡ Ações rápidas")

    # Relatório de informações gerais
    if st.button("📄 Relatório de informações gerais", key="botao_relatorio_geral"):
    with st.spinner("Gerando relatório 🦜"):
        try:
            resposta = orquestrador.invoke({"input": "Quero um relatório com informações sobre os dados"})
            st.session_state['relatorio_geral'] = resposta["output"]
        except groq.RateLimitError:
            st.error(
                "A API da Groq retornou erro de limite de requisições (Rate Limit). "
                "Tente novamente em alguns instantes."
            )
        except Exception as e:
            st.error("Ocorreu um erro ao gerar o relatório de informações gerais.")
            st.text(str(e))


    # Exibe o relatório com botão de download
    if 'relatorio_geral' in st.session_state:
        with st.expander("Resultado: Relatório de informações gerais"):
            st.markdown(st.session_state['relatorio_geral'])

            st.download_button(
                label="📥 Baixar relatório",
                data=st.session_state['relatorio_geral'],
                file_name="relatorio_informacoes_gerais.md",
                mime="text/markdown"
            )

    # Relatório de estatísticas descritivas
    if st.button("📄 Relatório de estatísticas descritivas", key="botao_relatorio_estatisticas"):
    with st.spinner("Gerando relatório 🦜"):
        try:
            resposta = orquestrador.invoke({"input": "Quero um relatório de estatísticas descritivas"})
            st.session_state['relatorio_estatisticas'] = resposta["output"]
        except groq.RateLimitError:
            st.error(
                "A API da Groq retornou erro de limite de requisições (Rate Limit). "
                "Tente novamente em alguns instantes."
            )
        except Exception as e:
            st.error("Ocorreu um erro ao gerar o relatório de estatísticas descritivas.")
            st.text(str(e))


    # Exibe o relatório salvo com opção de download
    if 'relatorio_estatisticas' in st.session_state:
        with st.expander("Resultado: Relatório de estatísticas descritivas"):
            st.markdown(st.session_state['relatorio_estatisticas'])

            st.download_button(
                label="📥 Baixar relatório",
                data=st.session_state['relatorio_estatisticas'],
                file_name="relatorio_estatisticas_descritivas.md",
                mime="text/markdown"  
            )
   
   # PERGUNTA SOBRE OS DADOS
    st.markdown("---")
st.markdown("## 🔎 Perguntas sobre os dados")
pergunta_sobre_dados = st.text_input("Faça uma pergunta sobre os dados (ex: 'Qual é a média do tempo de entrega?')")

if st.button("Responder pergunta", key="responder_pergunta_dados"):
    if not pergunta_sobre_dados.strip():
        st.warning("Digite uma pergunta antes de clicar em responder.")
    else:
        with st.spinner("Analisando os dados 🦜"):
            try:
                resposta = orquestrador.invoke({"input": pergunta_sobre_dados})
                st.markdown(resposta["output"])
            except groq.RateLimitError:
                st.error(
                    "A API da Groq retornou erro de limite de requisições (Rate Limit). "
                    "Tente novamente em alguns instantes."
                )
            except Exception as e:
                st.error("Ocorreu um erro ao responder sua pergunta sobre os dados.")
                st.text(str(e))



    # GERAÇÃO DE GRÁFICOS
    st.markdown("---")
    st.markdown("## 📊 Criar gráfico com base em uma pergunta")

    pergunta_grafico = st.text_input(
    "Digite o que deseja visualizar (ex.: 'Crie um gráfico da média de tempo de entrega por clima.')"
)

if st.button("Gerar gráfico", key="gerar_grafico"):
    if not pergunta_grafico.strip():
        st.warning("Digite uma instrução antes de gerar o gráfico.")
    else:
        with st.spinner("Gerando o gráfico 🦜"):
            try:
                orquestrador.invoke({"input": pergunta_grafico})
            except groq.RateLimitError:
                st.error(
                    "A API da Groq retornou erro de limite de requisições (Rate Limit). "
                    "Tente novamente em alguns instantes."
                )
            except Exception as e:
                st.error("Ocorreu um erro ao gerar o gráfico.")
                st.text(str(e))












