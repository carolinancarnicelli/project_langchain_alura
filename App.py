import streamlit as st
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from ferramentas import criar_ferramentas, llm   # <– REUSAR o llm global


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
arquivo_carregado = st.file_uploader("Selecione um arquivo CSV", type="csv", label_visibility="collapsed")

if arquivo_carregado:
    df = pd.read_csv(arquivo_carregado)
    st.success("Arquivo carregado com sucesso!")
    st.markdown("### 🔍 Primeiras linhas do DataFrame")
    st.dataframe(df.head())

    # Ferramentas
    tools = criar_ferramentas(df)

    # Prompt ReAct (em português), com regra para NÃO repetir ferramenta
    #df_head = df.head().to_markdown(index=False)
    df_sample = df.iloc[:5, :10]  # 5 linhas, 10 colunas
    df_head = df_sample.to_markdown(index=False)


    prompt_react_pt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        partial_variables={"df_head": df_head},
        template="""
Você é um assistente que SEMPRE responde em português, de forma clara e objetiva.

Você tem acesso a um DataFrame pandas chamado `df`.
Aqui estão as primeiras linhas, obtidas com `df.head().to_markdown()`:

{df_head}

Você tem acesso às seguintes ferramentas:

{tools}

Use SEMPRE o seguinte formato de raciocínio:

Question: a pergunta de entrada que você deve responder  
Thought: você deve sempre pensar no que fazer  
Action: a ação a ser tomada, deve ser uma das [{tool_names}]  
Action Input: a entrada para a ação (apenas o que a ferramenta precisa)  
Observation: o resultado da ação  

... (este bloco Thought / Action / Action Input / Observation pode se repetir N vezes)

REGRA IMPORTANTE:
- Se você já obteve, a partir de uma ferramenta, a informação necessária para responder à pergunta,
  NÃO chame a mesma ferramenta novamente.
- Em vez disso, faça:
  Thought: Agora eu sei a resposta final  
  Final Answer: <explique a resposta em português usando o resultado das ferramentas>
- Para perguntas de RELATÓRIO GERAL use SOMENTE a ferramenta "Informações DataFrame" UMA vez.
- Para perguntas de ESTATÍSTICAS DESCRITIVAS use SOMENTE a ferramenta "Resumo Estatístico" UMA vez.
- Para perguntas específicas que envolvem código Python (média, soma, filtro, etc.), use a ferramenta "Códigos Python".
- NÃO chame a mesma ferramenta mais de uma vez na mesma pergunta.

A resposta final SEMPRE deve aparecer no formato:

Final Answer: <texto da resposta em português>

Agora comece.

Question: {input}  
Thought: {agent_scratchpad}"""
    )

    # Agente ReAct + Executor com limite de iterações
    agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_pt)

    orquestrador = AgentExecutor(
        agent=agente,
        tools=tools,
        verbose=True,                 # mostra o log no terminal
        handle_parsing_errors=True,   # tenta se recuperar de erros de parsing
        max_iterations=6,             # limite de passos para evitar loops
        early_stopping_method="force" # força uma resposta final usando o último estado
    )

    # AÇÕES RÁPIDAS
    st.markdown("---")
    st.markdown("## ⚡ Ações rápidas")

    # Relatório de informações gerais (SEM agente)
    if st.button("📄 Relatório de informações gerais", key="botao_relatorio_geral"):
        with st.spinner("Gerando relatório 🦜"):
            texto_relatorio = informacoes_dataframe.run(
                {"pergunta": "Quero um relatório com informações sobre os dados", "df": df}
            )
            st.session_state['relatorio_geral'] = texto_relatorio


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

    # Relatório de estatísticas descritivas (SEM agente)
    if st.button("📄 Relatório de estatísticas descritivas", key="botao_relatorio_estatisticas"):
        with st.spinner("Gerando relatório 🦜"):
            texto_relatorio = resumo_estatistico.run(
                {"pergunta": "Quero um relatório de estatísticas descritivas", "df": df}
            )
            st.session_state['relatorio_estatisticas'] = texto_relatorio


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
#"""     st.markdown("---")
#    st.markdown("## 🔎 Perguntas sobre os dados")
#    pergunta_sobre_dados = st.text_input("Faça uma pergunta sobre os dados (ex: 'Qual é a média do tempo de entrega?')")
#    if st.button("Responder pergunta", key="responder_pergunta_dados"):
#        with st.spinner("Analisando os dados 🦜"):
#            resposta = orquestrador.invoke({"input": pergunta_sobre_dados})
#            st.markdown((resposta["output"])) """

    st.markdown("---")
    st.markdown("## 🔎 Perguntas sobre os dados")

    pergunta_dados = st.text_input(
        "Faça uma pergunta sobre os dados "
        "(ex: 'Qual é a média do tempo de entrega?')",
        key="pergunta_dados",
    )

    if st.button("Fazer pergunta", key="botao_pergunta_dados"):
        if not pergunta_dados.strip():
            st.warning("Digite uma pergunta antes de continuar.")
        else:
            with st.spinner("Consultando o agente 🦜"):
                resposta = orquestrador.invoke({"input": pergunta_dados})
                # guarda na sessão para não perder na reexecução do Streamlit
                st.session_state["resposta_pergunta_dados"] = resposta.get("output", "")

    if "resposta_pergunta_dados" in st.session_state:
        st.markdown("### Resposta")
        st.markdown(st.session_state["resposta_pergunta_dados"])

    # GERAÇÃO DE GRÁFICOS
    st.markdown("---")
    st.markdown("## 📊 Criar gráfico com base em uma pergunta")

    pergunta_grafico = st.text_input("Digite o que deseja visualizar (ex: 'Crie um gráfico da média de tempo de entrega por clima.')")
    if st.button("Gerar gráfico", key="gerar_grafico"):
        with st.spinner("Gerando o gráfico 🦜"):
            orquestrador.invoke({"input": pergunta_grafico})






