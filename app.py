import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pickle
from unidecode import unidecode
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# Carregar a logo e exibi-la na barra lateral
def exibir_logo():
    st.sidebar.image("logo.png", use_container_width=True)

# Definindo um SYSTEM_PROMPT que orienta o comportamento do agente de IA
SYSTEM_PROMPT = '''
Você é Victor, um Analista Contábil da empresa VW Contabilidade, especializado nos dados financeiros da empresa Rudy LTDA.
Sua função é fornecer análises financeiras e respostas precisas baseadas nos dados disponíveis, ajudando a empresa a tomar decisões informadas.
Por favor, responda todas as perguntas em português.
'''

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Diretório onde as mensagens serão armazenadas
PASTA_MENSAGENS = Path(__file__).parent / 'mensagens'
PASTA_MENSAGENS.mkdir(exist_ok=True)

# Função para invocar o modelo de IA e obter uma resposta
def retorna_resposta_modelo(dataframe, prompt, modelo='gpt-4', temperatura=0.2):
    llm = ChatOpenAI(api_key=openai_key, model=modelo, temperature=temperatura)
    agent = create_pandas_dataframe_agent(llm, dataframe, allow_dangerous_code=True)
    try:
        resposta = agent.invoke([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}])
        resposta_texto = resposta.get('output', "Nenhuma resposta foi gerada pelo agente.") if isinstance(resposta, dict) else str(resposta)
        return f"Rudy, {resposta_texto}"
    except Exception as e:
        print("Erro ao invocar o modelo:", e)
        return "Desculpe, ocorreu um erro ao processar a resposta."

# Função para normalizar o nome de uma mensagem para um formato válido de arquivo
def converte_nome_mensagem(nome_mensagem):
    """
    Remove caracteres inválidos de um nome de mensagem e retorna um nome de arquivo seguro.
    """
    if not nome_mensagem:
        return "nome_invalido"
    nome_arquivo = unidecode(nome_mensagem)  # Remove acentuação.
    nome_arquivo = re.sub(r'[^\w\s-]', '', nome_arquivo).strip()  # Remove caracteres especiais, exceto espaços e traços.
    nome_arquivo = re.sub(r'[\s]+', '-', nome_arquivo)  # Substitui espaços por traços.
    return nome_arquivo[:50].lower()  # Limita a 50 caracteres e converte para minúsculas.

# Função para salvar as mensagens da conversa no disco
def salvar_mensagens(mensagens):
    """
    Salva uma conversa em um arquivo de forma segura e legível.
    """
    if len(mensagens) == 0:
        return False

    # Usa a primeira mensagem do usuário como base para o nome do arquivo.
    nome_mensagem = ''
    for mensagem in mensagens:
        if mensagem['role'] == 'user':
            nome_mensagem = mensagem['content'][:50]  # Usa os primeiros 50 caracteres da mensagem do usuário.
            break

    nome_arquivo = converte_nome_mensagem(nome_mensagem)
    caminho_arquivo = PASTA_MENSAGENS / f"{nome_arquivo}.pkl"

    try:
        with open(caminho_arquivo, 'wb') as f:
            pickle.dump({'nome_mensagem': nome_mensagem, 'mensagens': mensagens}, f)
        return True
    except Exception as e:
        print(f"Erro ao salvar o arquivo: {e}")
        return False

# Função para listar todas as conversas salvas
def listar_conversas():
    conversas = list(PASTA_MENSAGENS.glob('*.pkl'))
    conversas = sorted(conversas, key=lambda item: item.stat().st_mtime_ns, reverse=True)
    return [c.stem for c in conversas]

# Função para carregar uma conversa pelo nome do arquivo
def ler_mensagem_por_nome_arquivo(nome_arquivo):
    """
    Lê e retorna as mensagens salvas em um arquivo.
    """
    caminho_arquivo = PASTA_MENSAGENS / f"{nome_arquivo}.pkl"
    if not caminho_arquivo.exists():
        st.error(f"Arquivo {caminho_arquivo} não encontrado.")
        return []
    try:
        with open(caminho_arquivo, 'rb') as f:
            dados = pickle.load(f)
            return dados.get('mensagens', [])
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return []

# Função para selecionar uma conversa salva e atualizar o estado da aplicação
def carregar_conversa(nome_arquivo):
    """
    Carrega uma conversa específica no estado da aplicação.
    """
    mensagens = ler_mensagem_por_nome_arquivo(nome_arquivo)
    if mensagens:
        st.session_state.mensagens = mensagens
        st.session_state.saudacao_enviada = True  # Marcar como saudação enviada
        st.success(f"Conversa '{nome_arquivo}' carregada com sucesso!")
    else:
        st.error("Não foi possível carregar a conversa.")


# Gráfico interativo de faturamento mensal
def gerar_grafico_faturamento(dataframe):
    fig = px.line(dataframe, x='data', y='faturamento', title='Faturamento Mensal', markers=True)
    fig.update_layout(xaxis_title="Mês", yaxis_title="Faturamento (R$)", xaxis_tickangle=45)
    st.plotly_chart(fig)

# Gráfico interativo de comparação de tributos
def gerar_grafico_tributos(dataframe):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['irpj'], name='IRPJ'))
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['csll'], name='CSLL'))
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['cofins'], name='COFINS'))

    fig.update_layout(
        title="Comparação de Tributos Mensais",
        xaxis_title="Mês",
        yaxis_title="Valor (R$)",
        barmode='stack',
        xaxis_tickangle=45
    )
    st.plotly_chart(fig)

# Gráfico interativo de alíquota efetiva
def gerar_grafico_aliquota(dataframe):
    fig = px.line(dataframe, x='data', y='aliquota efetiva', title='Alíquota Efetiva Mensal', markers=True)
    fig.update_layout(xaxis_title="Mês", yaxis_title="Alíquota Efetiva (%)", xaxis_tickangle=45)
    st.plotly_chart(fig)

# Aba para listar e selecionar conversas
def tab_conversas():
    """
    Interface para gerenciar conversas salvas.
    """
    st.sidebar.subheader("Conversas Salvas")
    if st.sidebar.button("➕ Nova Conversa"):
        st.session_state.mensagens = []
        st.session_state.saudacao_enviada = False
        st.sidebar.success("Nova conversa iniciada!")

    conversas = listar_conversas()
    if not conversas:
        st.sidebar.info("Nenhuma conversa salva.")
        return

    for nome_arquivo in conversas:
        if st.sidebar.button(nome_arquivo):
            carregar_conversa(nome_arquivo)
# Aba de configurações
def tab_configuracoes():
    st.sidebar.subheader("Configurações")
    st.write("**Configurações Gerais**")
    st.write(f"Modelo GPT Utilizado: `gpt-4`")
    st.write("Configuração de Temperatura: `0.2`")
    st.write("Configuração de segurança de código: Permitido")

# Página principal da aplicação
def pagina_principal():
    if 'mensagens' not in st.session_state:
        st.session_state.mensagens = []
    if 'saudacao_enviada' not in st.session_state:
        st.session_state.saudacao_enviada = False

    st.header('🤖 VW Consultor', divider=True)

    csv_path = os.path.join(os.path.dirname(__file__), "rudy_dados_empresa.csv")
    if not os.path.exists(csv_path):
        st.error("O arquivo CSV não foi encontrado.")
        return

    dataframe = pd.read_csv(csv_path)

    if st.session_state.mensagens:
        for mensagem in st.session_state.mensagens:
            st.chat_message(mensagem['role']).markdown(mensagem['content'])

    if not st.session_state.saudacao_enviada:
        saudacao = 'Olá, meu nome é Victor, sou Analista Contábil da empresa VW Contabilidade. Rudy, como posso ajudá-lo?'
        st.session_state.mensagens.append({'role': 'assistant', 'content': saudacao})
        st.session_state.saudacao_enviada = True
        st.chat_message('assistant').markdown(saudacao)

    prompt = st.chat_input('Fale com o Analista Contábil')
    if prompt:
        st.session_state.mensagens.append({'role': 'user', 'content': prompt})
        st.chat_message('user').markdown(prompt)
        resposta_completa = retorna_resposta_modelo(dataframe, prompt)
        st.session_state.mensagens.append({'role': 'assistant', 'content': resposta_completa})
        st.chat_message('assistant').markdown(resposta_completa)

    # Alternância para visualizar ou esconder os gráficos
    if 'exibir_graficos' not in st.session_state:
        st.session_state.exibir_graficos = False  # Inicializa o estado como False (gráficos ocultos)

    if st.button("Visualizar Gráficos"):
        # Alterna o estado de exibição dos gráficos
        st.session_state.exibir_graficos = not st.session_state.exibir_graficos

    # Exibe os gráficos apenas se o estado for True
    if st.session_state.exibir_graficos:
        st.subheader("Visualização de Dados")
        st.markdown("Clique nos botões abaixo para ver os gráficos.")

        if st.button("Faturamento Mensal"):
            gerar_grafico_faturamento(dataframe)

        if st.button("Comparação de Tributos"):
            gerar_grafico_tributos(dataframe)

        if st.button("Alíquota Efetiva"):
            gerar_grafico_aliquota(dataframe)

    if st.button("Salvar Conversa"):
        if salvar_mensagens(st.session_state.mensagens):
            st.success("Conversa salva com sucesso!")
        else:
            st.error("Erro ao salvar a conversa.")



# Função principal da aplicação
def main():
    exibir_logo()
    aba_selecionada = st.sidebar.radio("Menu", ["Conversas", "Configurações"], index=0)
    if aba_selecionada == "Conversas":
        pagina_principal()
        tab_conversas()
    elif aba_selecionada == "Configurações":
        tab_configuracoes()

if __name__ == '__main__':
    main()
