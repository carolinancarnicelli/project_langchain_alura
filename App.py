import streamlit as st
import pandas as pd
import os

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

from ferramentas import criar_ferramentas


# =====================================================
# Configuração da página
# =====================================================
st.set_page_config(page_title="Assistente de análise de dados com IA", layout="centered")
st.title("🦜 Assistente de análise de dados com IA")

st.info("""
Este assistente utiliza um agente, criado com LangChain, para te ajudar a explorar,
analisar e visualizar dados de forma interativa. Carregue um arquivo CSV, gere relatórios
rápidos ou faça perguntas livres para o agente.
""")


# =====================================================
# Upload do arquivo
# =====================================================
arquivo = st.file_uploader("Envie um arquivo CSV para análise", type=["csv"])

if arquivo is None:
    st.warning("Envie um arquivo CSV para começar a análise.")
    st.stop()

# Tentativa padrão (separador vírgula)
try:
    df = pd.read_csv(arquivo)
except Exception:
    # Fallback comum no Brasil: separador ;
    arquivo.seek(0)
    df = pd.read_csv(arquivo, sep=";", encoding="latin1")

st.success(f"Arquivo carregado com sucesso! Formato: {df.shape[0]} linhas x {df.shape[1]} colunas")

st.markdown("### Prévia dos dados")
st.dataframe(df.head())


# =====================================================
# LLM (mantido aqui como no curso)
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
if not GROQ_API_KEY and "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY não encontrada. Configure no .env ou em secrets do Streamlit.")
    st.stop()

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",  # modelo menor e mais barato
    max_tokens=512,
    temperature=0,
)


# =====================================================
# Ferramentas (ligadas ao df)
# =====================================================
tools = criar_ferramentas(df)

# Dicionário para acesso direto às ferramentas por nome
tools_by_name = {t.name: t for t in tools}


# =====================================================
# Agente ReAct (usado para perguntas livres e gráficos)
# =====================================================

# Para não pesar o prompt, usamos pequena amostra (5 linhas x 10 colunas)
df_sample = df.iloc[:5, : min(df.shape[1], 10)]
df_head = df_sample.to_markdown(index=False)

prompt_react_pt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    partial_variables={"df_head": df_head},
    template="""
Você é um assistente que SEMPRE responde em português, de forma clara e objetiva.

Você tem acesso a um DataFrame pandas chamado `df`.
Aqui estão as primeiras linhas (amostra):

{df_head}

Você tem acesso às seguintes ferramentas:

{tools}

Regras IMPORTANTES:
- Para RELATÓRIO GERAL prefira a ferramenta "Informações DataFrame".
- Para ESTATÍSTICAS DESCRITIVAS prefira a ferramenta "Resumo Estatístico".
- Para GRÁFICOS use a ferramenta "Gerar Gráfico".
- Para CÁLCULOS ESPECÍFICOS em Python use "Códigos Python", enviando código direto.
- NÃO chame a mesma ferramenta mais de uma vez na mesma pergunta.
- NÃO entre em loops chamando a mesma ação repetidamente.
- Responda SEMPRE em português.

Use SEMPRE o seguinte formato:

Thought: (seu raciocínio sobre o que fazer; NÃO inclua código aqui)
Action: (o nome exato de uma das ferramentas: {tool_names})
Action Input: (o input da ferramenta, em texto simples ou código Python quando for o caso)

... (este ciclo se repete até você ter informações suficientes)

Quando tiver a resposta final, use o formato:

Final Answer: (sua resposta final ao usuário, em português, sem mostrar o raciocínio interno)

Lembre-se: seja objetivo, não repita a mesma ferramenta várias vezes, e não invente colunas que não existem.
Pergunta do usuário: {input}
"""
)

agente = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_react_pt,
)

orquestrador = AgentExecutor(
    agent=agente,
    tools=tools,
    verbose=True,                 # log no terminal (não aparece para o usuário)
    handle_parsing_errors=True,   # tenta se recuperar de erros de parsing
    max_iterations=6,             # limite de passos para evitar loops
    early_stopping_method="force" # força resposta final se bater o limite
)


# =====================================================
# ⚡ AÇÕES RÁPIDAS (sem passar pelo agente)
# =====================================================
st.markdown("---")
st.markdown("## ⚡ Ações rápidas")

# Relatório de informações gerais
if st.button("📄 Relatório de informações gerais", key="botao_relatorio_geral"):
    with st.spinner("Gerando relatório 🦜"):
        ferramenta_info = tools_by_name.get("Informações DataFrame")
        if ferramenta_info is None:
            st.error("Ferramenta 'Informações DataFrame' não encontrada. Verifique o arquivo ferramentas.py.")
        else:
            texto_relatorio = ferramenta_info.run("Quero um relatório com informações sobre os dados.")
            st.session_state["relatorio_geral"] = texto_relatorio

# Exibe o relatório com botão de download
if "relatorio_geral" in st.session_state:
    with st.expander("Resultado: Relatório de informações gerais"):
        st.markdown(st.session_state["relatorio_geral"])
        st.download_button(
            label="📥 Baixar relatório",
            data=st.session_state["relatorio_geral"],
            file_name="relatorio_informacoes_gerais.md",
            mime="text/markdown",
        )

# Relatório de estatísticas descritivas
if st.button("📄 Relatório de estatísticas descritivas", key="botao_relatorio_estatisticas"):
    with st.spinner("Gerando relatório 🦜"):
        ferramenta_est = tools_by_name.get("Resumo Estatístico")
        if ferramenta_est is None:
            st.error("Ferramenta 'Resumo Estatístico' não encontrada. Verifique o arquivo ferramentas.py.")
        else:
            texto_est = ferramenta_est.run("Quero um relatório de estatísticas descritivas.")
            st.session_state["relatorio_estatisticas"] = texto_est

# Exibe o relatório salvo com opção de download
if "relatorio_estatisticas" in st.session_state:
    with st.expander("Resultado: Relatório de estatísticas descritivas"):
        st.markdown(st.session_state["relatorio_estatisticas"])
        st.download_button(
            label="📥 Baixar relatório",
            data=st.session_state["relatorio_estatisticas"],
            file_name="relatorio_estatisticas_descritivas.md",
            mime="text/markdown",
        )


# =====================================================
# 💬 Perguntas livres para o agente
# =====================================================
st.markdown("---")
st.markdown("## 💬 Perguntar algo sobre os dados")

pergunta_dados = st.text_input(
    "Digite sua pergunta (ex.: 'Qual é a média do tempo de entrega por tipo de clima?')",
    key="pergunta_dados",
)

if st.button("Fazer pergunta", key="botao_pergunta_dados"):
    if not pergunta_dados.strip():
        st.warning("Digite uma pergunta antes de continuar.")
    else:
        with st.spinner("Consultando o agente 🦜"):
            resposta = orquestrador.invoke({"input": pergunta_dados})
            st.session_state["resposta_pergunta_dados"] = resposta.get("output", "")

if "resposta_pergunta_dados" in st.session_state:
    st.markdown("### Resposta")
    st.markdown(st.session_state["resposta_pergunta_dados"])


# =====================================================
# 📊 Geração de gráficos via agente
# =====================================================
st.markdown("---")
st.markdown("## 📊 Criar gráfico com base em uma pergunta")

pergunta_grafico = st.text_input(
    "Digite o que deseja visualizar (ex.: 'Crie um gráfico da média de tempo_entrega por clima.')",
    key="pergunta_grafico",
)

if st.button("Gerar gráfico", key="gerar_grafico"):
    if not pergunta_grafico.strip():
        st.warning("Digite uma descrição do gráfico.")
    else:
        with st.spinner("Gerando o gráfico 🦜"):
            orquestrador.invoke({"input": pergunta_grafico})
        st.success("Se a solicitação foi compreendida, o gráfico deve aparecer acima.")
