import streamlit as st

import os
import json
import requests
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Rossmann Store Sales Prediction",
    page_icon=":bar_chart:",
    layout="centered")


col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(os.path.join(os.getcwd(), "images", "logo.png"), width=300)

# Title
st.title("Rossmann Store Sales Prediction")

# Explicacao
st.expander("Sobre o projeto", expanded=False).markdown(
    """
    O projeto consiste em um modelo de previsão de vendas para lojas da rede Rossmann. 
    O modelo desenvolvido(`XGBoostRegressor`) utiliza dados históricos de vendas, informações sobre as lojas e variáveis externas,
    como feriados e promoções, para prever o quanto cada loja irá vender nas próximas 6 semanas.
    O objetivo do app é disponibilizar essa previsão para os tomadores de decisão e para outros usuários da empresa que tiverem interesse em consultar essa previsão.
    """
)


# Dataframe com o numero e os dados das lojas
df_test = pd.read_csv(os.path.join(os.getcwd(), "data", "test.csv"))
df_store = pd.read_csv(os.path.join(os.getcwd(), "data", "store.csv"))

# Merge dos dois dataframes
df_full = pd.merge(df_test, df_store, on='Store', how='left')

col1, col2 = st.columns([2,2], gap="large", vertical_alignment="center")
with col1:
    # Selecionar os números possíveis das lojas 
    lojas = df_full['Store'].unique()
    loja_selecionada =st.selectbox("Qual loja deseja saber a previsão de vendas?", sorted(lojas), key="store")
    gerar = st.button("Gerar previsão")
    st.markdown("**ATENÇÃO:** *A previsão pode demorar alguns segundos quando gerada pela primeira vez.*")

with col2:
    if gerar:
        # Dataframe com os dados das lojas
        df_loja = df_full[df_full['Store'] == loja_selecionada]

        if df_loja.empty:
            st.warning(f"Não existem dados disponíveis para a previsão da loja {loja_selecionada}.")
        else:
            # Remove closed days
            df_loja = df_loja[df_loja['Open'] != 0]
            df_loja = df_loja[-df_loja['Open'].isnull()]
            df_loja = df_loja.drop('Id', axis = 1)

    
            # Converte Dataframe para JSON
            data = json.dumps(df_loja.to_dict(orient='records'))
            # API Call
            url = f'https://api-rossmann-prediction.onrender.com/rossmann/predict'
            header = {'Content-Type': 'application/json'}
            response = requests.post(url, data=data, headers=header)

            if response.status_code == 200:
                previsao = pd.DataFrame(response.json())
                previsao = previsao[['store', 'prediction']].groupby('store').sum().reset_index()
                st.markdown(f"A previsão de vendas nas próximas 6 semanas para a loja {loja_selecionada} é:")
                st.metric(label="", value=f"R$ {previsao['prediction'].values[0]:,.2f}")
            else:
                st.error("Erro ao obter a previsão. Tente novamente mais tarde.")
                st.write(f"Código de status: {response.status_code}")
                st.write(response.text)

df_train = pd.read_csv(os.path.join(os.getcwd(), "data", "train.csv"), low_memory=False)
# Vendas últimas 6 semanas
# Ultima data de venda 2015-07-31
df_train_ultimas = df_train[df_train['Date'] >= '2015-06-15']
df_train_ultimas = df_train_ultimas[df_train_ultimas['Open'] != 0]
df_train_ultimas = df_train_ultimas[df_train_ultimas['Open'].notnull()]
valor_ultimas_semanas = df_train_ultimas['Sales'].sum()
lojas_ultimas_semanas = df_train_ultimas['Store'].nunique()

# Previsão de vendas proximas 6 semanas
df_previsoes = pd.read_csv(os.path.join(os.getcwd(), "data", "previsoes.csv"))
df_proximas_semanas = (df_previsoes[df_previsoes['date'] >= '2015-08-06T00:00:00.000'])
df_proximas_semanas['prediction'].sum()
valor_proximas_semannas = df_proximas_semanas['prediction'].sum()
lojas_proximas_semanas = df_full['Store'].nunique()

st.divider()
col1, col2 = st.columns(2, gap="large", vertical_alignment="center")
with col1:
    st.metric("Vendas das últimas 6 semanas", value=f"R$ {valor_ultimas_semanas:,.2f}")
    st.metric("Número de lojas ativas nas últimas 6 semanas", value=lojas_ultimas_semanas)
    grafico_col = st.columns([1]) # Cria uma coluna que ocupa toda a largura disponível
    with grafico_col[0]:
        st.markdown("<h5 style='text-align: left;'>As 10 lojas que mais venderam(últimas 6 semanas)</h5>", unsafe_allow_html=True)
        lojas_mais_vendas = df_train_ultimas.groupby('Store')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        dados_vendas = lojas_mais_vendas.copy()
        mais_vendas = alt.Chart(lojas_mais_vendas).mark_bar().encode(
            x=alt.X('Sales:Q', title='Vendas Totais',),
            y=alt.Y('Store:N', sort='-x'),
            tooltip=['Store', 'Sales']
        ).properties(height=350,
                     width=360) 
        st.altair_chart(mais_vendas, use_container_width=False, key='grafico_vendas_aninhado')
    
with col2:
    st.metric("Previsão de vendas para as próximas 6 semanas", value=f"R$ {valor_proximas_semannas:,.2f}")
    st.metric("Número de lojas ativas nas últimas 6 semanas", value=lojas_proximas_semanas)
    
    grafico2_col = st.columns([1]) # Cria uma coluna que ocupa toda a largura disponível
    with grafico2_col[0]: 
        st.markdown("<h5 style='text-align: left;'>As 10 lojas com maior previsão de vendas</h5>", unsafe_allow_html=True)     
        lojas_mais_previsoes = df_proximas_semanas.groupby('store')['prediction'].sum().sort_values(ascending=False).head(10).reset_index()
        lojas_mais_previsoes.columns = ['Store', 'Prediction']
        mais_previsao = alt.Chart(lojas_mais_previsoes).mark_bar().encode(x=alt.X('Prediction:Q', title='Previsão de vendas totais'),
                                                            y=alt.Y('Store:N', sort='-x'),
                                                            tooltip=['Store', 'Prediction']
                                                            ).properties(height=350,
                                                                         width=360)                                                       
        st.altair_chart(mais_previsao, use_container_width=False, key='grafico_previsao')
    
st.divider()
st.expander("Considerações finais e próximos passos", expanded=False).markdown(
    """
    O objetivo princiapl desteprojeto de previsão de vendas é ser um projeto de aprendizado. Ele teve como principais objetivos
    entender as etapas de desenvolvimento de um projeto end-to-end de ciência de dados e desenvolver senso de negócio e pensamento crítico.
    Há muitas possibilidades de melhorias, como por exemplo:
    - Testar outros algoritmos de machine learning que sejam mais rosbustos;
    - Utilizar abordagens que lidem melhor com séries temporais;
    - Acrescentar mais features ao app para que o projeto ganhe outros utilidades, como gráficos de comparação entre a loja consultada e as demais lojas;
    - Implementar um sistema de monitoramento do modelo, para que o mesmo seja atualizado periodicamente e a qualidade da previsão seja garantida. 
"""
)

st.markdown("Projeto desenvolvido por [Marcela M. P. Amorim](https://www.linkedin.com/in/marcela-de-pretto-amorim/)")