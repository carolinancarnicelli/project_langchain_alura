# 📊 Projeto: Analisador Inteligente de Dados com LangChain, Groq e Streamlit

Este projeto implementa uma aplicação interativa em **Streamlit** que permite:

* Carregar arquivos CSV grandes.
* Visualizar os dados.
* Fazer perguntas para um **agente inteligente baseado em LLM (Groq + LangChain)**.
* Gerar gráficos automatizados com base em perguntas em linguagem natural.
* Criar um fluxo modular com ferramentas personalizadas (“tools”) para análise de dados.

O objetivo é fornecer uma plataforma simples, rápida e eficiente para automatizar análises exploratórias e geração de insights a partir de datasets tabulares.

---

# 🚀 Tecnologias Utilizadas

| Tecnologia                          | Uso                                                                  |
| ----------------------------------- | -------------------------------------------------------------------- |
| **LangChain 0.3+ (versão moderna)** | Criação das ferramentas, agentes e cadeia de raciocínio.             |
| **Groq API (LLaMA 3.x)**            | Modelo LLM ultrarrápido para consultas e geração de código/insights. |
| **Streamlit 1.44+**                 | Interface web interativa.                                            |
| **Pandas 2.2+**                     | Processamento de dados tabulares.                                    |
| **Matplotlib / Seaborn**            | Criação de gráficos automatizados.                                   |
| **python-dotenv**                   | Controle de variáveis de ambiente e segurança.                       |

---

# 📁 Estrutura do Projeto

```
/
├── app.py                     # Aplicação Streamlit principal
├── agent_tools.py             # Ferramentas personalizadas para o agente
├── agent_setup.py             # Configuração do LLM, agente e execução
├── requirements.txt           # Dependências do projeto
├── dados/                     # Arquivos CSV enviados pelo usuário (em runtime)
└── README.md                  # Este arquivo
```

---

# ⚙️ Funcionalidades

### ✔️ Upload de arquivo CSV

O usuário faz upload de um arquivo `.csv`, que é automaticamente lido com tratamento de encoding e separadores.

### ✔️ Pré-visualização automática dos dados

O app exibe:

* Número de linhas e colunas
* Cabeçalho
* Preview das primeiras linhas

### ✔️ Agente inteligente para análise

O usuário pode escrever perguntas, como:

* *“Qual é a média do tempo_entrega por clima?”*
* *“Mostre a contagem de entregas por cidade.”*

O LLM interpreta a pergunta, gera código **pandas** seguro e executa automaticamente no backend.

### ✔️ Geração automática de gráficos

Perguntas como:

* *“Crie um gráfico de barras com a média do tempo_entrega por clima.”*

resultam em um gráfico renderizado diretamente no Streamlit.

### ✔️ Ferramentas personalizadas (LangChain Tools)

Criamos ferramentas que o agente pode usar:

* `Informações DataFrame`
* `Resumo Estatístico`
* `Gerar Gráfico`
* `Executar Código Python` (sandbox controlado)

### ✔️ Otimização de tokens para Groq

* Limpeza do contexto
* Limitação de histórico
* Uso de modelos menores quando possível
* Execução real de código para evitar respostas verbosas

---

# 🔧 Instalação

### 1. Clonar este repositório

```bash
git clone https://github.com/<seu-usuario>/<seu-repositorio>.git
cd <seu-repositorio>
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Criar arquivo `.env`

```ini
GROQ_API_KEY="sua-chave-groq"
```

---

# ▶️ Execução Local

```bash
streamlit run app.py
```

Abra no navegador:
`http://localhost:8501`

---

# 📦 Deploy na Streamlit Cloud

1. Subir o repo para o GitHub
2. Acessar: [https://share.streamlit.io](https://share.streamlit.io)
3. Selecionar o repositório
4. Adicionar variável de ambiente:

```
GROQ_API_KEY
```

### Cuidados importantes:

* A Streamlit Cloud tem **limites de memória** → trate arquivos CSV pesados.
* Evite enviar o DataFrame completo ao LLM → use *row sampling* (ex. 200 primeiras linhas).
* Evite loops infinitos no agente → configure limites de interação.

---

# ⚠️ Problemas conhecidos e Soluções

### 1. ❗ “Agent stopped due to iteration limit”

O agente gerou um loop.
**Solução:** reduzir número de steps, limitar ferramentas, revisar prompts.

### 2. ❗ “Request too large — TPM limit exceeded”

Pedido grande demais para o Groq.
**Solução:**

* reduzir tamanho do DataFrame enviado ao prompt
* remover histórico
* usar modelo *llama-3.1-8b-instant*

### 3. ❗ Gráficos não aparecem no Streamlit

Ocorria quando o LLM gerava código com `plt.show()`.
**Solução atual:** código modernizado usa `st.pyplot(fig)`.

---

# 🧠 Arquitetura do Agente

O projeto segue um fluxo moderno do LangChain 0.3+:

1. LLM Groq configurado
2. Tools registradas
3. AgentExecutor criado em modo *tool-calling*
4. Pergunta → LLM decide qual tool usar
5. Se gerar código → sandbox executa
6. Resultado retornado ao Streamlit

Isso garante segurança, controle e menor consumo de tokens. ⚡

---

# 📚 Exemplo de Pergunta

### Pergunta:

> Qual é a média do tempo de entrega por tipo de clima?

### Processo:

1. LLM interpreta a intenção
2. Gera código Pandas:

```python
df.groupby("clima")["tempo_entrega"].mean()
```

3. Executa
4. Retorna o DataFrame formatado
5. Pode opcionalmente gerar gráfico

---

# 🌟 Melhorias Futuras

* Adicionar cache inteligente com `st.cache_data`
* Suporte a múltiplos arquivos simultâneos
* Exportação automática de relatórios PDF/Excel
* Histórico de consultas
* Modo batch para pipelines de ETL

---

# 📜 Licença

Este projeto está sob a licença MIT – livre para uso e modificação.
