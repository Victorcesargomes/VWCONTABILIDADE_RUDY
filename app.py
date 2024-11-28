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
    # Exibe uma imagem chamada "logo.png" na barra lateral.
    st.sidebar.image("logo.png", use_container_width=True)

# Prompt base para o agente de IA, definindo o contexto do chatbot.
SYSTEM_PROMPT = '''
Você é Victor, um Analista Contábil da empresa VW Contabilidade, especializado nos dados financeiros da empresa Rudy LTDA.
Sua função é fornecer análises financeiras e respostas precisas baseadas nos dados disponíveis, ajudando a empresa a tomar decisões informadas.
Por favor, responda todas as perguntas em português.
'''

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()  # Configura as variáveis de ambiente definidas no arquivo .env.
openai_key = os.getenv("OPENAI_API_KEY")  # Busca a chave da API OpenAI.

# Diretório onde as mensagens serão armazenadas
PASTA_MENSAGENS = Path(__file__).parent / 'mensagens'  # Define um diretório para armazenar mensagens.
PASTA_MENSAGENS.mkdir(exist_ok=True)  # Garante que o diretório exista.

# Função para invocar o modelo de IA e obter uma resposta
def retorna_resposta_modelo(dataframe, prompt, modelo='gpt-4o-mini', temperatura=0.2):
    """
    Usa o modelo GPT para processar um dataframe e retornar uma resposta para um prompt dado.
    """
    # Inicializa o modelo GPT-4 com a API do OpenAI.
    llm = ChatOpenAI(api_key=openai_key, model=modelo, temperature=temperatura)
    # Cria um agente que interage com o dataframe.
    agent = create_pandas_dataframe_agent(llm, dataframe, allow_dangerous_code=True)
    try:
        # Invoca o modelo com o prompt e o contexto definido.
        resposta = agent.invoke([
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": prompt}
        ])
        # Processa a resposta para extrair o texto.
        resposta_texto = resposta.get('output', "Nenhuma resposta foi gerada pelo agente.") if isinstance(resposta, dict) else str(resposta)
        return f"Rudy, {resposta_texto}"
    except Exception as e:
        # Trata erros na invocação do modelo.
        print("Erro ao invocar o modelo:", e)
        return "Desculpe, ocorreu um erro ao processar a resposta."

# Função para normalizar o nome de uma mensagem para um formato válido de arquivo
def converte_nome_mensagem(nome_mensagem):
    """
    Converte o nome de uma mensagem para um formato seguro para usar como nome de arquivo.
    """
    if not nome_mensagem:
        return "nome_invalido"
    nome_arquivo = unidecode(nome_mensagem)  # Remove acentos.
    nome_arquivo = re.sub(r'[^\w\s-]', '', nome_arquivo).strip()  # Remove caracteres especiais.
    nome_arquivo = re.sub(r'[\s]+', '-', nome_arquivo)  # Substitui espaços por traços.
    return nome_arquivo[:50].lower()  # Limita a 50 caracteres e converte para minúsculas.

# Função para salvar as mensagens da conversa no disco
def salvar_mensagens(mensagens):
    """
    Salva as mensagens de uma conversa em um arquivo no formato pickle.
    """
    if len(mensagens) == 0:
        return False

    # Usa a primeira mensagem do usuário como base para o nome do arquivo.
    nome_mensagem = ''
    for mensagem in mensagens:
        if mensagem['role'] == 'user':
            nome_mensagem = mensagem['content'][:50]
            break

    nome_arquivo = converte_nome_mensagem(nome_mensagem)
    caminho_arquivo = PASTA_MENSAGENS / f"{nome_arquivo}.pkl"

    try:
        # Salva a conversa no arquivo.
        with open(caminho_arquivo, 'wb') as f:
            pickle.dump({'nome_mensagem': nome_mensagem, 'mensagens': mensagens}, f)
        return True
    except Exception as e:
        # Trata erros ao salvar.
        print(f"Erro ao salvar o arquivo: {e}")
        return False

# Função para listar todas as conversas salvas
def listar_conversas():
    """
    Retorna uma lista de conversas salvas, ordenadas por data de modificação.
    """
    conversas = list(PASTA_MENSAGENS.glob('*.pkl'))
    conversas = sorted(conversas, key=lambda item: item.stat().st_mtime_ns, reverse=True)
    return [c.stem for c in conversas]

# Função para carregar uma conversa pelo nome do arquivo
def ler_mensagem_por_nome_arquivo(nome_arquivo):
    """
    Lê mensagens salvas de um arquivo específico.
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
    Carrega uma conversa específica no estado da aplicação (st.session_state).
    """
    mensagens = ler_mensagem_por_nome_arquivo(nome_arquivo)
    if mensagens:
        st.session_state.mensagens = mensagens  # Atualiza o estado com as mensagens carregadas.
        st.session_state.saudacao_enviada = True  # Marca que a saudação já foi enviada.
        st.success(f"Conversa '{nome_arquivo}' carregada com sucesso!")  # Exibe mensagem de sucesso.
    else:
        st.error("Não foi possível carregar a conversa.")  # Exibe mensagem de erro.

# Gráfico interativo de faturamento mensal
def gerar_grafico_faturamento(dataframe):
    """
    Gera um gráfico de linha para o faturamento mensal usando Plotly.
    """
    fig = px.line(dataframe, x='data', y='faturamento', title='Faturamento Mensal', markers=True)
    # Personaliza os títulos dos eixos.
    fig.update_layout(xaxis_title="Mês", yaxis_title="Faturamento (R$)", xaxis_tickangle=45)
    st.plotly_chart(fig)  # Renderiza o gráfico no Streamlit.

# Gráfico interativo de comparação de tributos
def gerar_grafico_tributos(dataframe):
    """
    Gera um gráfico de barras empilhadas para comparação de tributos mensais.
    """
    fig = go.Figure()
    # Adiciona cada tributo como uma barra separada.
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['irpj'], name='IRPJ'))
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['csll'], name='CSLL'))
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['cofins'], name='COFINS'))
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['pis'], name='PIS'))
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['cpp'], name='CPP'))
    fig.add_trace(go.Bar(x=dataframe['data'], y=dataframe['iss'], name='ISS'))

    # Configura o layout do gráfico.
    fig.update_layout(
        title="Comparação de Tributos Mensais",
        xaxis_title="Mês",
        yaxis_title="Valor (R$)",
        barmode='stack',  # Barras empilhadas.
        xaxis_tickangle=45
    )
    st.plotly_chart(fig)  # Renderiza o gráfico no Streamlit.


# Gráfico interativo de alíquota efetiva
def gerar_grafico_aliquota(dataframe):
    """
    Gera um gráfico de linha para a alíquota efetiva mensal.
    """
    fig = px.line(dataframe, x='data', y='aliquota efetiva', title='Alíquota Efetiva Mensal', markers=True)
    # Personaliza os títulos dos eixos.
    fig.update_layout(xaxis_title="Mês", yaxis_title="Alíquota Efetiva (%)", xaxis_tickangle=45)
    st.plotly_chart(fig)  # Renderiza o gráfico no Streamlit.

# Aba para listar e selecionar conversas
def tab_conversas():
    """
    Interface para gerenciar conversas salvas na barra lateral.
    """
    st.sidebar.subheader("Conversas Salvas")  # Subtítulo na barra lateral.

    # Botão para iniciar uma nova conversa.
    if st.sidebar.button("➕ Nova Conversa"):
        st.session_state.mensagens = []  # Limpa as mensagens existentes.
        st.session_state.saudacao_enviada = False  # Marca que uma saudação ainda não foi enviada.
        st.sidebar.success("Nova conversa iniciada!")  # Mensagem de sucesso.

    conversas = listar_conversas()  # Obtém a lista de conversas salvas.
    if not conversas:
        st.sidebar.info("Nenhuma conversa salva.")  # Mensagem informativa se não houver conversas.
        return

    # Lista as conversas na barra lateral.
    for nome_arquivo in conversas:
        if st.sidebar.button(nome_arquivo):  # Adiciona um botão para cada conversa.
            carregar_conversa(nome_arquivo)  # Carrega a conversa selecionada.

# Aba de configurações
def tab_configuracoes():
    """
    Aba de configurações para exibir informações do modelo utilizado.
    """
    st.sidebar.subheader("Configurações")
    st.write("**Configurações Gerais**")
    st.write(f"Modelo GPT Utilizado: `gpt-4`")
    st.write("Configuração de Temperatura: `0.2`")
    st.write("Configuração de segurança de código: Permitido")

# Página principal da aplicação
def pagina_principal():
    """
    Página principal da aplicação para exibir o chat e os gráficos.
    """
    # Inicializa as variáveis de sessão caso ainda não existam.
    if 'mensagens' not in st.session_state:
        st.session_state.mensagens = []
    if 'saudacao_enviada' not in st.session_state:
        st.session_state.saudacao_enviada = False

    st.header('🤖 VW Consultor', divider=True)  # Cabeçalho da página principal.

    # Caminho do arquivo CSV contendo os dados financeiros.
    csv_path = os.path.join(os.path.dirname(__file__), "rudy_dados_empresa.csv")
    if not os.path.exists(csv_path):  # Verifica se o arquivo CSV existe.
        st.error("O arquivo CSV não foi encontrado.")
        return

    dataframe = pd.read_csv(csv_path)  # Carrega os dados do CSV em um dataframe.

    # Exibe as mensagens existentes no estado da aplicação.
    if st.session_state.mensagens:
        for mensagem in st.session_state.mensagens:
            st.chat_message(mensagem['role']).markdown(mensagem['content'])

    # Envia uma mensagem de saudação caso ainda não tenha sido enviada.
    if not st.session_state.saudacao_enviada:
        saudacao = 'Olá, meu nome é Victor, sou Analista Contábil da empresa VW Contabilidade. Rudy, como posso ajudá-lo?'
        st.session_state.mensagens.append({'role': 'assistant', 'content': saudacao})
        st.session_state.saudacao_enviada = True
        st.chat_message('assistant').markdown(saudacao)

    # Input de mensagens pelo usuário.
    prompt = st.chat_input('Fale com o Analista Contábil')
    if prompt:
        st.session_state.mensagens.append({'role': 'user', 'content': prompt})  # Armazena a mensagem do usuário.
        st.chat_message('user').markdown(prompt)  # Exibe a mensagem na interface.
        resposta_completa = retorna_resposta_modelo(dataframe, prompt)  # Obtém a resposta do modelo.
        st.session_state.mensagens.append({'role': 'assistant', 'content': resposta_completa})  # Armazena a resposta.
        st.chat_message('assistant').markdown(resposta_completa)  # Exibe a resposta na interface.

    # Botão para alternar a exibição dos gráficos.
    if 'exibir_graficos' not in st.session_state:
        st.session_state.exibir_graficos = False  # Inicializa o estado como False (gráficos ocultos).

    if st.button("Visualizar Gráficos"):
        st.session_state.exibir_graficos = not st.session_state.exibir_graficos  # Alterna o estado.

    # Exibe os gráficos caso o estado seja True.
    if st.session_state.exibir_graficos:
        st.subheader("Visualização de Dados")
        st.markdown("Clique nos botões abaixo para ver os gráficos.")

        # Botões para exibir gráficos específicos.
        if st.button("Faturamento Mensal"):
            gerar_grafico_faturamento(dataframe)
        if st.button("Comparação de Tributos"):
            gerar_grafico_tributos(dataframe)
        if st.button("Alíquota Efetiva"):
            gerar_grafico_aliquota(dataframe)

    # Botão para salvar a conversa atual.
    if st.button("Salvar Conversa"):
        if salvar_mensagens(st.session_state.mensagens):
            st.success("Conversa salva com sucesso!")
        else:
            st.error("Erro ao salvar a conversa.")

# Função principal da aplicação
def main():
    """
    Função principal que gerencia a interface e o fluxo da aplicação.
    """
    exibir_logo()  # Exibe a logo na barra lateral.
    aba_selecionada = st.sidebar.radio("Menu", ["Conversas", "Configurações"], index=0)  # Seleção de abas.
    if aba_selecionada == "Conversas":
        pagina_principal()  # Exibe a página principal.
        tab_conversas()  # Exibe a aba de conversas.
    elif aba_selecionada == "Configurações":
        tab_configuracoes()  # Exibe a aba de configurações.

# Inicia a aplicação
if __name__ == '__main__':
    main()
