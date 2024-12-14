import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from unidecode import unidecode
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Carregar a logo e exibi-la na barra lateral
def exibir_logo():
    st.sidebar.image("logo.png", use_container_width=True)

# Prompt base para o agente de IA
SYSTEM_PROMPT = '''
Você é Victor, um Analista Contábil da empresa VW Contabilidade, especializado nos dados financeiros da empresa Rudy LTDA.
Sua função é fornecer análises financeiras e respostas precisas baseadas nos dados disponíveis, ajudando a empresa a tomar decisões informadas. Respostas somente em português. 
Sempre termine a resposta com o nome Rudy.
'''

# Classe Pydantic para validar a saída do modelo
class FinancialResponse(BaseModel):
    resposta: str = Field(description="A resposta gerada pelo modelo")

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Função para criar o parser de saída com correção
def criar_parser_correcoes():
    """
    Cria um parser que utiliza o OutputFixingParser para corrigir saídas inválidas do modelo.
    """
    try:
        print("Inicializando o modelo LLM...")
        llm = ChatOpenAI(api_key=openai_key, model="gpt-4o")
        
        print("Criando o parser Pydantic...")
        pydantic_parser = PydanticOutputParser(pydantic_object=FinancialResponse)
        
        print("Configurando o OutputFixingParser...")
        fixing_parser = OutputFixingParser.from_llm(
            parser=pydantic_parser,
            llm=llm,
        )
        return fixing_parser
    except Exception as e:
        print(f"Erro ao criar o parser com correções: {e}")
        raise

def retorna_resposta_modelo(dataframe, prompt, modelo='gpt-4o', temperatura=0.2):
    """
    Usa o modelo GPT para processar um dataframe e retornar uma resposta para um prompt dado.
    Retorna o texto puro, sem aspas e sem prefixos como 'resposta'.
    """
    try:
        # Inicializa o modelo GPT
        llm = ChatOpenAI(api_key=openai_key, model=modelo, temperature=temperatura)
        agent = create_pandas_dataframe_agent(llm, dataframe, allow_dangerous_code=True)

        # Criar o parser com correções
        output_parser = criar_parser_correcoes()

        # Invocar o modelo com o prompt e contexto definido
        resposta = agent.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])

        # Validar e corrigir a saída
        parsed_response = output_parser.parse(resposta.get("output", ""))
        
        # Retorna apenas o texto da resposta
        return parsed_response.resposta.strip()
    except Exception as e:
        print(f"Erro ao processar o modelo ou corrigir a saída: {e}")
        return "Desculpe, ocorreu um erro ao processar a resposta."

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

# Aba para iniciar nova conversa
def tab_nova_conversa():
    """
    Interface para iniciar uma nova conversa na barra lateral.
    """
    st.sidebar.subheader("Gerenciar Conversas")  # Subtítulo na barra lateral.

    # Botão para iniciar uma nova conversa
    if st.sidebar.button("➕ Nova Conversa"):
        st.session_state.mensagens = []  # Limpa as mensagens existentes.
        st.session_state.saudacao_enviada = False  # Marca que uma saudação ainda não foi enviada.
        st.sidebar.success("Nova conversa iniciada!")  # Mensagem de sucesso.

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

# Função principal da aplicação
def main():
    """
    Função principal que gerencia a interface e o fluxo da aplicação.
    """
    exibir_logo()  # Exibe a logo na barra lateral.
    tab_nova_conversa()  # Adiciona a funcionalidade de iniciar nova conversa.
    pagina_principal()  # Exibe a página principal.

if __name__ == '__main__':
    main()
