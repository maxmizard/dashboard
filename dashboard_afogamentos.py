#=================== PREPARAÇÃO DO DATASET ===================
# Importar bibliotecas:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
import plotly.express as px
import streamlit as st
import time

# Ler Dataset:
df_afogamentos = pd.read_csv('df_afogamentos.csv')

# Converter a coluna hora:
df_afogamentos['hora'] = pd.to_datetime(df_afogamentos['hora'])

# Criar um novo Dataset para cada ano:
df_afogamentos_2020 = df_afogamentos[df_afogamentos['ano']==2020]
df_afogamentos_2020.reset_index(drop=True)
df_afogamentos_2020 = df_afogamentos_2020.drop(columns=['ano'])

df_afogamentos_2021 = df_afogamentos[df_afogamentos['ano']==2021]
df_afogamentos_2021.reset_index(drop=True)
df_afogamentos_2021 = df_afogamentos_2021.drop(columns=['ano'])

df_afogamentos_2022 = df_afogamentos[df_afogamentos['ano']==2022]
df_afogamentos_2022.reset_index(drop=True)
df_afogamentos_2022 = df_afogamentos_2022.drop(columns=['ano'])

df_afogamentos_2023 = df_afogamentos[df_afogamentos['ano']==2023]
df_afogamentos_2023.reset_index(drop=True)
df_afogamentos_2023 = df_afogamentos_2023.drop(columns=['ano'])

df_afogamentos_2024 = df_afogamentos[df_afogamentos['ano']==2024]
df_afogamentos_2024.reset_index(drop=True)
df_afogamentos_2024 = df_afogamentos_2024.drop(columns=['ano'])

#=================== STREAMLIT ===================

# Definir layout da página:
st.set_page_config(page_title="Afogamentos", layout="wide")

# Definir título:
st.title("📊 Chamados de afogamentos recebidos pelo CBMPE")

# Rodapé fixo na parte inferior:
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #666;
    }
    </style>
    <div class="footer">
        Desenvolvido por Maxmiliano Augusto - https://github.com/maxmizard | 📅 2025
    </div>
    """,
    unsafe_allow_html=True
)

#=================== GRÁFICO 01 ===================

# Definir espaços dos gráficos 1 e 2:
col1, col2 = st.columns(2)

# Criar tabela:
df_historico = df_afogamentos.groupby('ano')['mês'].count().reset_index().rename(columns={'mês': 'Qtd. Afogamentos'})
df_historico['cor'] = df_historico['ano'].apply(lambda x: '#443983' if x == 2022 else '#31688e')

# Gerar visualização:
fig_grafico_1 = px.bar(df_historico, x="ano", y="Qtd. Afogamentos", color='cor', color_discrete_map={'#443983': '#443983', '#31688e': '#31688e'})

fig_grafico_1.update_xaxes(showgrid=False)
fig_grafico_1.update_yaxes(showgrid=False)

fig_grafico_1.update_traces(texttemplate='%{y}', textposition='outside')

fig_grafico_1.update_layout(showlegend=False)
fig_grafico_1.update_layout(bargap=0.5)
fig_grafico_1.update_layout(xaxis_title='')
fig_grafico_1.update_layout(yaxis_title='Quantidade de acionamentos')
fig_grafico_1.update_layout(title="Registros de Afogamentos por Ano", title_x=0.3, title_font=dict(size=20))
fig_grafico_1.update_layout(height=500)
fig_grafico_1.update_layout(width=300)

col1.plotly_chart(fig_grafico_1)

#=================== GRÁFICO 02 ===================

# Inicialmente, vamos criar uma função para converter faixas de horário para os turnos correspondentes:
def periodo(hora):
    try:
        if pd.isna(hora):
            return "Valor Nulo"
        elif 0 <= hora.hour < 6:  # Madrugada
            return 'Madrugada'
        elif 6 <= hora.hour < 12:  # Manhã
            return 'Manhã'
        elif 12 <= hora.hour < 18:  # Tarde (inclui 12h)
            return 'Tarde'
        else:  # 18 <= hora.hour <= 23  # Noite (inclui 0h)
            return 'Noite'
    except AttributeError:
        return "Valor Inválido"


# Agora, podemos aplicar a função a todos os Datasets:
df_afogamentos_2020['periodo'] = df_afogamentos_2020['hora'].apply(periodo)
df_afogamentos_2021['periodo'] = df_afogamentos_2021['hora'].apply(periodo)
df_afogamentos_2022['periodo'] = df_afogamentos_2022['hora'].apply(periodo)
df_afogamentos_2023['periodo'] = df_afogamentos_2023['hora'].apply(periodo)
df_afogamentos_2024['periodo'] = df_afogamentos_2024['hora'].apply(periodo)


# Vamos criar uma nova tabela com o percentual de ocorrências registradas por turno.
# OBS: Aqui, a preferência foi pela porcentagem, pois há valores nulos na coluna 'hora'.
df_periodo_2020 = df_afogamentos_2020['periodo'].value_counts(normalize=True).rename('2020').mul(100).reset_index()
df_periodo_2021 = df_afogamentos_2021['periodo'].value_counts(normalize=True).rename('2021').mul(100).reset_index()
df_periodo_2022 = df_afogamentos_2022['periodo'].value_counts(normalize=True).rename('2022').mul(100).reset_index()
df_periodo_2023 = df_afogamentos_2023['periodo'].value_counts(normalize=True).rename('2023').mul(100).reset_index()
df_periodo_2024 = df_afogamentos_2024['periodo'].value_counts(normalize=True).rename('2024').mul(100).reset_index()

df_periodo = df_periodo_2020.merge(df_periodo_2021, how='outer')
df_periodo = df_periodo.merge(df_periodo_2022, how='outer')
df_periodo = df_periodo.merge(df_periodo_2023, how='outer')
df_periodo = df_periodo.merge(df_periodo_2024, how='outer')

df_periodo = df_periodo.set_index('periodo').T

# Desconsiderando os valores nulo, temos:
df_periodo = df_periodo[['Madrugada', 'Manhã', 'Tarde', 'Noite']]
df_periodo.reset_index(inplace=True)
df_periodo.rename(columns={'index': 'ano', 'periodo': 'index'}, inplace=True)

# Gerar visualização:
fig_grafico_2 = px.bar(
    df_periodo,  
    x="ano", 
    y=['Madrugada', 'Manhã', 'Tarde', 'Noite'],
    color_discrete_sequence=['#c2df23', '#2d708e', '#86d549', '#482173'],
    barmode="group",
    labels={"value": "Frequência (%)", "variable": "Período"}
)

fig_grafico_2.update_xaxes(showgrid=False)
fig_grafico_2.update_yaxes(showgrid=False)

fig_grafico_2.update_traces(texttemplate='%{y:.2f}', textposition='outside')

fig_grafico_2.update_layout(bargap=0.1)
fig_grafico_2.update_layout(xaxis_title='')
fig_grafico_2.update_layout(yaxis_title='Quantidade de acionamentos (%)')
fig_grafico_2.update_layout(title="Registros de Afogamentos por Período do Dia", title_x=0.2, title_font=dict(size=20))
fig_grafico_2.update_layout(height=500)
fig_grafico_2.update_layout(width=300)

col2.plotly_chart(fig_grafico_2)

#=================== GRÁFICO 03 ===================

# Somar as ocorrências, por mês, em cada um dos Datasets:
df_meses_2020 = df_afogamentos_2020['mês'].value_counts().reset_index()
df_meses_2020.rename(columns={'count': '2020'}, inplace=True)

df_meses_2021 = df_afogamentos_2021['mês'].value_counts().reset_index()
df_meses_2021.rename(columns={'count': '2021'}, inplace=True)

df_meses_2022 = df_afogamentos_2022['mês'].value_counts().reset_index()
df_meses_2022.rename(columns={'count': '2022'}, inplace=True)

df_meses_2023 = df_afogamentos_2023['mês'].value_counts().reset_index()
df_meses_2023.rename(columns={'count': '2023'}, inplace=True)

df_meses_2024 = df_afogamentos_2024['mês'].value_counts().reset_index()
df_meses_2024.rename(columns={'count': '2024'}, inplace=True)

# Criar um Dataset com o nomes dos meses:
# OBS: Ele será útil para a junção desses novos Datasets que possuem o acumulado de cada ano por mês.
meses = {'mês': ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']}
df_meses = pd.DataFrame(meses)

# Criar um novo Dataset com o total por mês em cada ano:
df_meses = df_meses.merge(df_meses_2020, on='mês', how='left')
df_meses = df_meses.merge(df_meses_2021, on='mês', how='left')
df_meses = df_meses.merge(df_meses_2022, on='mês', how='left')
df_meses = df_meses.merge(df_meses_2023, on='mês', how='left')
df_meses = df_meses.merge(df_meses_2024, on='mês', how='left')

# Gerar visualização:
fig_grafico_3 = px.line(df_meses, 
              x="mês", 
              y=['2020', '2021', '2022', '2023', '2024'], 
              color_discrete_sequence=['#482173', '#2d708e', '#2ab07f', '#86d549', '#c2df23'], 
              labels={"value": "Quantidade de acionamentos", "variable": "Ano"}
              )

fig_grafico_3.update_yaxes(showgrid=False)

fig_grafico_3.update_traces(line_width=3)

fig_grafico_3.update_layout(xaxis_title='')
fig_grafico_3.update_layout(yaxis_title='Quantidade de acionamentos')
fig_grafico_3.update_layout(title="Registros de Afogamentos por Mês", title_x=0.37, title_font=dict(size=20))
fig_grafico_3.update_layout(height=500)
fig_grafico_3.update_layout(width=800)

with st.container():
    st.plotly_chart(fig_grafico_3) 

#=================== GRÁFICO 04 ===================

# Definir espaços dos gráficos 4 e 5:
col4, col5 = st.columns(2)

# Vamos criar tabelas com a situação das vítimas em cada um dos anos:
# OBS: Aqui devemos atentar para os valores nulos (onde não foi informada a classificação da vítima).
# Para essas casos, será atribuída o valor "Não classificado".
df_vitimas_2020 = df_afogamentos_2020['classificacao_vitima'].value_counts().reset_index()
df_vitimas_2020.rename(columns={'count': '2020'}, inplace=True)
vitimas_nulas_2020 = len(df_afogamentos_2020[df_afogamentos_2020['classificacao_vitima'].isnull()])
df_vitimas_2020.loc[len(df_vitimas_2020.index)] = ['Não classificado', vitimas_nulas_2020]

df_vitimas_2021 = df_afogamentos_2021['classificacao_vitima'].value_counts().reset_index()
df_vitimas_2021.rename(columns={'count': '2021'}, inplace=True)
vitimas_nulas_2021 = len(df_afogamentos_2021[df_afogamentos_2021['classificacao_vitima'].isnull()])
df_vitimas_2021.loc[len(df_vitimas_2021.index)] = ['Não classificado', vitimas_nulas_2021]

df_vitimas_2022 = df_afogamentos_2022['classificacao_vitima'].value_counts().reset_index()
df_vitimas_2022.rename(columns={'count': '2022'}, inplace=True)
vitimas_nulas_2022 = len(df_afogamentos_2022[df_afogamentos_2022['classificacao_vitima'].isnull()])
df_vitimas_2022.loc[len(df_vitimas_2022.index)] = ['Não classificado', vitimas_nulas_2022]

df_vitimas_2023 = df_afogamentos_2023['classificacao_vitima'].value_counts().reset_index()
df_vitimas_2023.rename(columns={'count': '2023'}, inplace=True)
vitimas_nulas_2023 = len(df_afogamentos_2023[df_afogamentos_2023['classificacao_vitima'].isnull()])
df_vitimas_2023.loc[len(df_vitimas_2023.index)] = ['Não classificado', vitimas_nulas_2023]

df_vitimas_2024 = df_afogamentos_2024['classificacao_vitima'].value_counts().reset_index()
df_vitimas_2024.rename(columns={'count': '2024'}, inplace=True)
vitimas_nulas_2024 = len(df_afogamentos_2024[df_afogamentos_2024['classificacao_vitima'].isnull()])
df_vitimas_2024.loc[len(df_vitimas_2024.index)] = ['Não classificado', vitimas_nulas_2024]

# Concatenando todas as tabelas, temos:
df_vitimas = df_vitimas_2020.merge(df_vitimas_2021, on='classificacao_vitima', how='left')
df_vitimas = df_vitimas.merge(df_vitimas_2022, on='classificacao_vitima', how='left')
df_vitimas = df_vitimas.merge(df_vitimas_2023, on='classificacao_vitima', how='left')
df_vitimas = df_vitimas.merge(df_vitimas_2024, on='classificacao_vitima', how='left')

df_vitimas.rename(columns={'classificacao_vitima': 'classificacao_vitima'}, inplace=True)

# Como há muitos casos onde a vítima não foi classificada, vamos remover essa linha e pivotar a tabela:
df_vitimas = df_vitimas.set_index('classificacao_vitima').T

# Resetando o índice:
df_vitimas.reset_index(inplace=True)
df_vitimas.rename(columns={'classificacao_vitima': 'index', 'index': 'ano'}, inplace=True)

# Gerar visualização:
fig_grafico_4 = px.bar(
    df_vitimas,  
    x="ano", 
    y=['Vítima ilesa', 'Vítima fatal', 'Vítima ferida', 'Vítima desaparecida', 'Não classificado'],
    color_discrete_sequence=['#c2df23', '#2d708e', '#86d549', '#482173', 'gray'],
    barmode="group",
    labels={"value": "Frequência (%)", "variable": "Vítima"}
)

fig_grafico_4.update_xaxes(showgrid=False)
fig_grafico_4.update_yaxes(showgrid=False)

fig_grafico_4.update_traces(texttemplate='%{y}', textposition='outside')

fig_grafico_4.update_layout(bargap=0.3)
fig_grafico_4.update_layout(xaxis_title='')
fig_grafico_4.update_layout(yaxis_title='Quantidade de acionamentos')
fig_grafico_4.update_layout(title="Registros de Afogamentos pela Classificação da Vítima", title_x=0.1, title_font=dict(size=20))
fig_grafico_4.update_layout(height=500)
fig_grafico_4.update_layout(width=800)

col4.plotly_chart(fig_grafico_4)

#=================== GRÁFICO 05 ===================

# Vamos criar recortes onde a coluna 'classificacao_vitima' possui o valor 'Vítima fatal' e a coluna 'sexo' possui os valores 'M' ou 'F'.
df_fatal_2020 = df_afogamentos_2020[(df_afogamentos_2020['classificacao_vitima']=='Vítima fatal') & (df_afogamentos_2020['sexo']!="Não identificado") & (df_afogamentos_2020['sexo'].notnull())]
df_fatal_2021 = df_afogamentos_2021[(df_afogamentos_2021['classificacao_vitima']=='Vítima fatal') & (df_afogamentos_2021['sexo']!="Não identificado") & (df_afogamentos_2021['sexo'].notnull())]
df_fatal_2022 = df_afogamentos_2022[(df_afogamentos_2022['classificacao_vitima']=='Vítima fatal') & (df_afogamentos_2022['sexo']!="Não identificado") & (df_afogamentos_2022['sexo'].notnull())]
df_fatal_2023 = df_afogamentos_2023[(df_afogamentos_2023['classificacao_vitima']=='Vítima fatal') & (df_afogamentos_2023['sexo']!="Não identificado") & (df_afogamentos_2023['sexo'].notnull())]
df_fatal_2024 = df_afogamentos_2024[(df_afogamentos_2024['classificacao_vitima']=='Vítima fatal') & (df_afogamentos_2024['sexo']!="Não identificado") & (df_afogamentos_2024['sexo'].notnull())]

# Criar lista com os valores de vítima fatais por sexo para cada ano:
df_fatal_2020_M = df_fatal_2020[df_fatal_2020['sexo']=='M']
df_fatal_2020_M = df_fatal_2020_M.reset_index(drop=True)

df_fatal_2021_M = df_fatal_2021[df_fatal_2021['sexo']=='M']
df_fatal_2021_M = df_fatal_2021_M.reset_index(drop=True)

df_fatal_2022_M = df_fatal_2022[df_fatal_2022['sexo']=='M']
df_fatal_2022_M = df_fatal_2022_M.reset_index(drop=True)

df_fatal_2023_M = df_fatal_2023[df_fatal_2023['sexo']=='M']
df_fatal_2023_M = df_fatal_2023_M.reset_index(drop=True)

df_fatal_2024_M = df_fatal_2024[df_fatal_2024['sexo']=='M']
df_fatal_2024_M = df_fatal_2024_M.reset_index(drop=True)

df_fatal_2020_F = df_fatal_2020[df_fatal_2020['sexo']=='F']
df_fatal_2020_F = df_fatal_2020_F.reset_index(drop=True)

df_fatal_2021_F = df_fatal_2021[df_fatal_2021['sexo']=='F']
df_fatal_2021_F = df_fatal_2021_F.reset_index(drop=True)

df_fatal_2022_F = df_fatal_2022[df_fatal_2022['sexo']=='F']
df_fatal_2022_F = df_fatal_2022_F.reset_index(drop=True)

df_fatal_2023_F = df_fatal_2023[df_fatal_2023['sexo']=='F']
df_fatal_2023_F = df_fatal_2023_F.reset_index(drop=True)

df_fatal_2024_F = df_fatal_2024[df_fatal_2024['sexo']=='F']
df_fatal_2024_F = df_fatal_2024_F.reset_index(drop=True)

# Gerar tabela apenas com o acumulado de cada ano por sexo:
anos = [2020, 2021, 2022, 2023, 2024]

fatais_M = np.array([df_fatal_2020_M.shape[0],
            df_fatal_2021_M.shape[0],
            df_fatal_2022_M.shape[0],
            df_fatal_2023_M.shape[0],
            df_fatal_2024_M.shape[0]])

fatais_F = np.array([df_fatal_2020_F.shape[0],
            df_fatal_2021_F.shape[0],
            df_fatal_2022_F.shape[0],
            df_fatal_2023_F.shape[0],
            df_fatal_2024_F.shape[0]])


# Gerar tabela com total de ocorrências por sexo:
df_vitimas_fatais = pd.DataFrame({'Ano': anos, 'Masculino': fatais_M, 'Feminino': fatais_F})

# Gerar visualização:
fig_grafico_5 = px.bar(df_vitimas_fatais, 
             x='Ano', 
             y=['Masculino', 'Feminino'],
             color_discrete_sequence=['#2d708e', '#482173'],
             labels={"value": "Quantidade de acionamentos", "variable": "Sexo"}, 
             height=400)

fig_grafico_5.update_xaxes(showgrid=False)
fig_grafico_5.update_yaxes(showgrid=False)

fig_grafico_5.update_traces(texttemplate='%{y}', textposition='outside')

fig_grafico_5.update_layout(title="Registros de Vítimas Fatais de Afogamentos por Sexo",
                  title_x=0.1,
                  title_font=dict(size=20),
                  xaxis_title='',
                  yaxis_title='Quantidade de acionamentos',
                  height=500,
                  width=1000)


col5.plotly_chart(fig_grafico_5)

#=================== GRÁFICO 06 ===================

# Criar uma nova tabela com as cidades:
df_cidades = df_afogamentos.groupby('ano')['cidade'].value_counts().rename('afogamentos')
df_cidades = df_cidades.unstack(fill_value=0)
df_cidades = df_cidades.reset_index()

# Filtrar cidades que serão destacadas no gráfico:
df_cidades.reset_index(inplace=True)
df_top_cidades = df_cidades.filter(['ano', 'Recife', 'Olinda', 'Jaboatão dos Guararapes', 'Fernando de Noronha', 'Petrolina', 'Caruaru'])

# Transformando os dados para formato "long"
df_long = df_top_cidades.melt(id_vars=["ano"], var_name="Município", value_name="Quantidade de acionamentos")

# Gerar visualização:
fig_grafico_6 = px.line(
    df_long, 
    x="ano", 
    y="Quantidade de acionamentos", 
    color="Município", 
    markers=True,
    color_discrete_map={
        "Recife": "#482173",
        "Olinda": "#2d708e",
        "Jaboatão dos Guararapes": "#25858e",
        "Fernando de Noronha": "#52c569",
        "Petrolina": "#86d549",
        "Caruaru": "#c2df23"
    })

fig_grafico_6.update_layout(
    title="Registros de Afogamentos por Municípios",
    title_x=0.37,
    title_font=dict(size=20),
    xaxis_title='',
    xaxis=dict(tickmode="linear"),
    yaxis_title="Quantidade de acionamentos",
    legend_title="Município",
    height=500,
    width=1000
)

fig_grafico_6.update_traces(line_width=3)

fig_grafico_6.update_yaxes(showgrid=False)

with st.container():
    st.plotly_chart(fig_grafico_6) 
