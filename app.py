import re
import streamlit as st 
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from unidecode import unidecode
from pathlib import Path
import pickle

# Definindo um SYSTEM_PROMPT simplificado para o contexto do agente
SYSTEM_PROMPT = '''
Você é Victor, um Analista Contábil da empresa VW Contabilidade, especializado nos dados financeiros da empresa Rudy LTDA.
Sua função é fornecer análises financeiras e respostas precisas baseadas nos dados disponíveis, ajudando a empresa a tomar decisões informadas.
Por favor, responda todas as perguntas em português.
'''

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

PASTA_MENSAGENS = Path(__file__).parent / 'mensagens'
PASTA_MENSAGENS.mkdir(exist_ok=True)

CACHE_DESCONVERT = {}

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

def converte_nome_mensagem(nome_mensagem):
    if not nome_mensagem:
        return "nome_invalido"
    nome_arquivo = unidecode(nome_mensagem)
    nome_arquivo = re.sub(r'\W+', '', nome_arquivo).lower()
    return nome_arquivo

def salvar_mensagens(mensagens):
    if len(mensagens) == 0:
        return False

    nome_mensagem = ''
    for mensagem in mensagens:
        if mensagem['role'] == 'user':
            nome_mensagem = mensagem['content'][:30]
            break

    nome_arquivo = converte_nome_mensagem(nome_mensagem)
    arquivo_salvar = {'nome_mensagem': nome_mensagem, 'nome_arquivo': nome_arquivo, 'mensagens': mensagens}
    
    caminho_arquivo = PASTA_MENSAGENS / f"{nome_arquivo}.pkl"
    try:
        with open(caminho_arquivo, 'wb') as f:
            pickle.dump(arquivo_salvar, f)
        return True
    except Exception as e:
        print(f"Erro ao salvar o arquivo: {e}")
        return False

def ler_mensagem_por_nome_arquivo(nome_arquivo, key='mensagens'):
    caminho_arquivo = PASTA_MENSAGENS / f"{nome_arquivo}.pkl"
    if not caminho_arquivo.exists():
        print(f"Arquivo {caminho_arquivo} não encontrado.")
        return [] if key == 'mensagens' else "Conversa Não Encontrada"
    try:
        with open(caminho_arquivo, 'rb') as f:
            mensagens = pickle.load(f)
            return mensagens[key]
    except Exception as e:
        print(f"Erro ao ler o arquivo {caminho_arquivo}: {e}")
        return [] if key == 'mensagens' else "Conversa Não Encontrada"

def desconverte_nome_mensagem(nome_arquivo):
    if nome_arquivo not in CACHE_DESCONVERT:
        nome_mensagem = ler_mensagem_por_nome_arquivo(nome_arquivo, key='nome_mensagem')
        if nome_mensagem == "Conversa Não Encontrada":
            CACHE_DESCONVERT[nome_arquivo] = "Conversa Não Encontrada"
        else:
            CACHE_DESCONVERT[nome_arquivo] = nome_mensagem
    return CACHE_DESCONVERT.get(nome_arquivo, "Conversa Não Encontrada")

def listar_conversas():
    conversas = list(PASTA_MENSAGENS.glob('*.pkl'))
    conversas = sorted(conversas, key=lambda item: item.stat().st_mtime_ns, reverse=True)
    return [c.stem for c in conversas]

def pagina_principal():
    if 'mensagens' not in st.session_state:
        st.session_state.mensagens = []
    if 'saudacao_enviada' not in st.session_state:
        st.session_state.saudacao_enviada = False

    st.header('🤖 VW ChatBot', divider=True)

    # Verificação do caminho do arquivo CSV
    csv_path = os.path.join(os.path.dirname(__file__), "rudy_dados_empresa.csv")
    if not os.path.exists(csv_path):
        st.error("O arquivo CSV não foi encontrado. Verifique o caminho do arquivo.")
        return  # Interrompe a execução se o arquivo não existir

    # Carregando o CSV em um DataFrame
    dataframe = pd.read_csv(csv_path)

    # Exibe mensagens salvas ou carregadas
    if st.session_state.mensagens:
        for mensagem in st.session_state.mensagens:
            chat = st.chat_message(mensagem['role'])
            chat.markdown(mensagem['content'])

    # Se o chatbot ainda não enviou a saudação, faça isso agora
    if not st.session_state.saudacao_enviada:
        saudacao = 'Olá, meu nome é Victor, sou Analista Contábil da empresa VW Contabilidade. Rudy, como posso ajudá-lo?'
        st.session_state.mensagens.append({'role': 'assistant', 'content': saudacao})
        st.session_state.saudacao_enviada = True
        chat = st.chat_message('assistant')
        chat.markdown(saudacao)

    # Captura a entrada do usuário
    prompt = st.chat_input('Fale com o Analista Contábil')
    if prompt:
        # Adiciona a mensagem do usuário à conversa
        st.session_state.mensagens.append({'role': 'user', 'content': prompt})
        chat = st.chat_message('user')
        chat.markdown(prompt)

        # Obtém a resposta do agente
        resposta_completa = retorna_resposta_modelo(dataframe, prompt)
        st.session_state.mensagens.append({'role': 'assistant', 'content': resposta_completa})
        chat = st.chat_message('assistant')
        chat.markdown(resposta_completa)

    # Botão para salvar mensagens
    if st.button("Salvar Conversa"):
        if salvar_mensagens(st.session_state.mensagens):
            st.success("Conversa salva com sucesso!")
        else:
            st.error("Erro ao salvar a conversa.")


def tab_conversas(tab):
    tab.button('➕ Nova Conversa', on_click=seleciona_conversa, args=('', ), use_container_width=True)
    tab.markdown('')
    conversas = listar_conversas()
    for nome_arquivo in conversas:
        nome_mensagem = desconverte_nome_mensagem(nome_arquivo)
        if nome_mensagem != "Conversa Não Encontrada":
            tab.button(nome_mensagem, on_click=seleciona_conversa, args=(nome_arquivo, ), use_container_width=True)

def seleciona_conversa(nome_arquivo):
    if nome_arquivo == '':
        st.session_state.mensagens = []
    else:
        mensagens = ler_mensagem_por_nome_arquivo(nome_arquivo)
        if mensagens:
            st.session_state.mensagens = mensagens
        else:
            st.error("Conversa não encontrada.")

def main():
    pagina_principal()
    tab1, tab2 = st.sidebar.tabs(['Conversas', 'Configurações'])
    tab_conversas(tab1)

if __name__ == '__main__':
    main()
