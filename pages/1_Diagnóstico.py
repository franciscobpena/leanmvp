# =====================================
# Parte 1: Importações e Configurações Iniciais (Atualizado)
# =====================================

from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import math
import os
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime
import matplotlib.colors as mcolors
import graphviz as gv
from plotly.subplots import make_subplots 
from scipy import stats
from scipy import special

#======================================
# Título da Página
#======================================

st.set_page_config(page_title="🔎 Diagnóstico Hospitalar", layout="wide")


# =====================================
# Funções Auxiliares
# =====================================

# Função para converter porcentagens em formato string para float
def porcentagem_para_float(valor):
    if isinstance(valor, str):
        valor = valor.strip()
        if valor.endswith('%'):
            return float(valor.strip('%').replace(',', '.')) / 100
        else:
            return float(valor.replace(',', '.'))
    elif isinstance(valor, (int, float)):
        return valor
    else:
        return 0.0

# Função para definir o período baseado no horário
def definir_periodo(hora):
    if pd.isna(hora):
        return 'Indefinido'
    else:
        try:
            hora = int(hora)
            if 0 <= hora <= 6:
                return 'Madrugada'
            elif 7 <= hora <= 13:
                return 'Manhã'
            elif 14 <= hora <= 19:
                return 'Tarde'
            elif 20 <= hora <= 23:
                return 'Noite'
            else:
                return 'Indefinido'
        except ValueError:
            return 'Indefinido'
            
# =====================================
# Definir constantes para as abas e colunas (Atualizado)
# =====================================

ABAS = {
    "MENSAL": "dados_mensais_pacientes",
    "SEMANAL": "dados_semanais_pacientes",
    "HORA": "dados_hora_paciente",
    "HORIZ_VERTIC": "pacientes_horiz_vertic",
    "PONTOS_CUIDADO": "pontos_de_cuidado",
    "CLASSIFICACAO": "classificacao_de_pacientes",
    "RETORNO": "retorno_pacientes",
    "SAIDA": "saida_evasao_abandono",
    "ORIENTADOS": "orientados_para_rede",
    "TRIAGEM_URGENCIA": "distribuicao_triagem",
    "TRIAGEM_ENFERMEIROS": "media_enfermeiros_triagem",
    "TRIAGEM_SALAS": "quantidade_salas_triagem",
    "TRIAGEM_TEMPO": "tempo_medio_atendimento_triagem",
    "EXAMES_SADT": "exames_sadt",
    "CONSULTA_TEMPO": "consulta_tempo",
    "MEDIA_MEDICOS_CONSULTA": "media_medicos_consulta",
    "DADOS_SEMANAIS_MEDICOS": "dados_semanais_medicos",
    "MEDIA_MEDICOS_ESPECIALIDADE": "media_medicos_dia_especialidade",

    # Abas do Centro Cirúrgico
    "FUNCIONAMENTO_CC": "funcionamento_centro_cirurgico",
    "MOTIVOS_CANCELAMENTO": "motivos_cancelamento_cirurgia",
    "TEMPO_ATRASO_CIRURGIA": "tempo_medio_atraso_primeira",
    "TEMPO_SUBSTIT_SALA": "tempo_medio_substit_sala",
    "TEMPO_SETUP_SALA": "tempo_medio_setup_sala",
    "MEDIA_HORAS_AGENDADAS": "media_horas_agendadas_cirurgia",
    "MEDIA_HORAS_GASTAS": "media_horas_gastas_cirurgia",
    "TAXA_INDICADORES_CC": "taxa_media_indicadores_cc",
    "MOTIVOS_ATRASO_CIRURGIA": "motivos_atraso_cirurgia",
    "MOTIVOS_TEMPO_PERMANENCIA_RPA": "motivos_tempo_permanencia_rpa",
    "TEMPO_PERMANENCIA_LEITOS": "tempo_permanencia_leitos",  # Corrigido
    "CLASSIFICACAO_SALAS_CIRURGICAS": "classificacao_salas_cirurgicas",
    "SALAS_CIRURGICAS_PORTE": "salas_cirurgicas_porte",
    "QTD_CIRURGIAS_ELETIVAS_ESPEC": "qtd_cirurgia_eletivas_espec",
    "QTD_CIRURGIAS_NAO_PROGRAMADAS": "qtd_cirurgias_nao_programadas",
    "TEMPO_MEDIO_SOLICITACAO_CIRURGIA": "tempo_medio_solicitacao_cirurgi",
    "MEDIA_MEDICOS_CC": "media_medicos_centro_cirurgico",
    "CIRURGIAS_MES": "cirurgias_mes",

    # Abas adicionais para Passagem & Internação
    "PASSAGEM_SETORES": "passagem_setores",
    "INTERNACAO_DEMANDA": "internacao_demanda",
    "INTERNACAO_SAIDA": "internacao_saida",
    "TAXA_INTERNACAO": "taxa_internacao",
}

COLUNAS = {
    "MENSAL": {
        "MES": "mes",
        "ANO": "ano",
        "QUANTIDADE_PACIENTES": "quantidade_pacientes_mes"
    },
    "SEMANAL": {
        "DIA": "dia",
        "QUANTIDADE_MEDIA": "quantidade_media_dia"
    },
    "HORA": {
        "HORARIO": "horario",
        "QUANTIDADE_MEDIA": "quantidade_media_pacientes"
    },
    "HORIZ_VERTIC": {
        "CARACTERISTICA": "caracteristica",
        "QUANTIDADE_MEDIA": "quantidade_media_pacientes_dia"
    },
    "PONTOS_CUIDADO": {
        "LOCAL": "local",
        "PONTO_CUIDADO": "ponto_cuidado",
        "QUANTIDADE": "quantidade"
    },
    "CLASSIFICACAO": {
        "CLASSIFICACAO": "classificacao",
        "QUANTIDADE_PACIENTES": "quantidade_pacientes"
    },
    "RETORNO": {
        "INDICADORES": "retorno_pacientes",
        "QUANTIDADE_MEDIA": "quantidade_media_dia_retorno"
    },
    "SAIDA": {
        "INDICADORES": "saida_paciente",
        "QUANTIDADE_MEDIA": "quantidade_media_dia_saida_evasao/abandono"
    },
    "ORIENTADOS": {
        "INDICADORES": "orientados_para_rede",
        "QUANTIDADE_MEDIA": "quantidade_media_pacientes_dia_orientados"
    },
    "TRIAGEM_URGENCIA": {
        "URGENCIA": "urgencia",
        "QUANTIDADE_PACIENTES": "quantidade_triagem_por_urgencia"
    },
    "TRIAGEM_ENFERMEIROS": {
        "HORARIO": "horario_triagem",
        "MEDIA_ENFERMEIROS": "quantidade_media_enfermeiros_triagem"
    },
    "TRIAGEM_SALAS": {
        "NUM_SALAS": "quantidade_salas_triagem"
    },
    "TRIAGEM_TEMPO": {
        "TEMPO_MEDIO_ATENDIMENTO": "tempo_medio_atendimento_triagem"
    },
    "EXAMES_SADT": {
        "TIPO_EXAME": "tipo_exame",
        "TEMPO_MEDIO_EXAME": "tempo_medio_exame",
        "QUANTIDADE_PACIENTE_EXAME_MES": "quantidade_paciente_exame_mes"
    },
    "CONSULTA_TEMPO": {
        "ETAPA": "etapa",
        "TEMPO_MEDIO_ETAPA": "tempo_medio_etapa_min"
    },
    "MEDIA_MEDICOS_CONSULTA": {
        "HORARIO": "horario",
        "QUANTIDADE_MEDIA_MEDICOS": "quantidade_media_medicos"
    },
    "DADOS_SEMANAIS_MEDICOS": {
        "DIA": "dia",
        "MEDICOS_MANHA_TARDE": "quantidade_media_dia_medicos_manha/tarde",
        "MEDICOS_NOITE_MADRUGADA": "quantidade_media_dia_medicos_noite/madrugada"
    },
    "MEDIA_MEDICOS_ESPECIALIDADE": {
        "ESPECIALIDADE": "especialidade",
        "QUANTIDADE_MEDIA_MEDICOS": "quantidade_media_dia_medicos",
        "PERCENTUAL_ATENDIMENTO_DIA": "percentual_atendimento_dia"
    },
    "CLASSIFICACAO_SALAS_CIRURGICAS": {
        "CLASSIFICACAO_SALAS_CIRURGICAS": "classificacao_salas_cirurgicas",
        "QUANTIDADE_SALAS_CIRURGICAS": "quantidade_salas_cirurgicas"
    },
    "SALAS_CIRURGICAS_PORTE": {
        "PORTE_SALAS": "porte_salas",
        "QTD_ELETIVAS": "quantidade_salas_cirurgicas_eletivas",
        "QTD_URGENCIA": "quantidade_salas_cirurgicas_urgencia"
    },
    "TEMPO_PERMANENCIA_LEITOS": {  # Corrigido
        "TIPO_DE_LEITO": "tipo_de_leito",
        "CLASSIFICACAO_SALAS_CIRURGICAS": "classificacao_salas_cirurgicas",
        "QUANTIDADE_DE_LEITO": "quantidade_de_leito",
        "TEMPO_MEDIO_PERMANENCIA_LEITO": "tempo_medio_permanencia_leito (min)"  # Corrigido
    },
    "MOTIVOS_TEMPO_PERMANENCIA_RPA": {
        "MOTIVOS_RPA": "Tempo_permanencia_RPA_maior_3h_motivos",
        "PERCENTUAL_MOTIVOS": "Percentual_motivos"
    },
    "FUNCIONAMENTO_CC": {
        "PERIODO": "periodo",
        "HORARIO_INICIO": "horario_inicio",
        "HORARIO_TERMINO": "horario_termino"
    },
    "QUANTIDADE_CIRURGIAS": {
        "CLASSIFICACAO_CIRURGIAS": "classificacao_tipo_cirurgias",
        "QTD_MEDIA": "quantidade_media_cirurgias_realizadas_mes"
    },
    "TEMPO_MEDIO_SOLICITACAO_CIRURGIA": {
        "TEMPO_MEDIO_SOLICITACAO": "mediana_horario_pedido_ate_cirurgia_urgencia (min)"
    },
    "QTD_CIRURGIAS_ELETIVAS_ESPEC": {
        "ESPECIALIDADE_CIRURGIA": "especialidade_cirurgia",
        "QTD_ELETIVAS_ESPEC": "quantidade"
    },
    "MOTIVOS_CANCELAMENTO": {
        "CLASSIFICACAO_CANCELAMENTO": "classificacao",
        "MOTIVO_CANCELAMENTO": "motivos_cancelamento",
        "QTD_CANCELAMENTO_MEDIA": "quantidade_por_cancelamento_media_mes"
    },
    "QTD_CIRURGIAS_NAO_PROGRAMADAS": {
        "QTD_CIRURGIAS_NAO_PROGRAMADAS": "quantidade_cirurgias_complemento_p/dia"
    },
    "TEMPO_MEDIO_ATRASO_CIRURGIA": {
        "TEMPO_ATRASO": "tempo_medio_atraso_primeira_cirurgia(min)"
    },
    "MOTIVOS_ATRASO_CIRURGIA": {
        "MOTIVOS_ATRASO": "motivos_atraso_primeira_cirurgia",
        "PERCENTUAL_MOTIVOS": "percentual_motivos"
    },
    "TEMPO_SETUP_SALA": {
        "TEMPO_SETUP": "tempo_medio_setup_sala(min)"
    },
    "TEMPO_SUBSTIT_SALA": {
        "TEMPO_SUBSTITUICAO": "tempo_medio_substituicao_sala_sus"
    },
    "MEDIA_HORAS_AGENDADAS": {
        "HORAS_AGENDADAS": "media_horas_dia_agendadas_cirurgia(min)"
    },
    "MEDIA_HORAS_GASTAS": {
        "HORAS_GASTAS": "media_horas_dia_agendas_cirurgia (min)"
    },
    "MEDIA_MEDICOS_CC": {
        "DIA": "dia",
        "MEDICOS_CIRURGIAO": "media_medicos_cirurgiao_dia",
        "MEDICOS_ANESTESISTA": "media_medicos_anestesista_dia"
    },
    "TAXA_INDICADORES_CC": {
        "INDICADOR": "indicador",
        "TAXA_MEDIA": "resultado_percentual"
    },
    "CIRURGIAS_MES": {
        "MES": "mes",
        "ANO": "ano",
        "ELETIVAS_SUS": "eletivas/sus",
        "ELETIVAS_SUPLEMENTAR": "eletivas/suplementar",
        "URGENCIA_SUS": "urgencia/sus",
        "URGENCIA_SUPLEMENTAR": "urgencia/suplementar"
    },
    "PASSAGEM_SETORES": {
        "SETORES": "setores",
        "QUANTIDADE_LEITOS": "quantidade_leitos",
        "TEMPO_MEDIO_PERMANENCIA_DIAS": "tempo_medio_permanencia_dias",
        "TAXA_OCUPACAO": "taxa_ocupacao"
    },
    "INTERNACAO_DEMANDA": {
        "SOLICITACOES_LEITO": "solicitacoes_leito",
        "MEDIA_SOLICITACOES_DIA": "media_solicitacoes_dia"
    },
    "INTERNACAO_SAIDA": {
        "SAIDA_INTERNACAO": "saida_internacao",
        "MEDIA_SAIDA_DIA": "media_saida_dia"
    },
    "TAXA_INTERNACAO": {
        "INDICADOR": "indicador",
        "RESULTADO_PERCENTUAL": "resultado_percentual"
    },
}

# =====================================
# Sidebar - Barra Lateral
# =====================================
image_path = 'app2.png'
if os.path.exists(image_path):
    image = Image.open(image_path)
    st.sidebar.image(image, width=190)
else:
    st.sidebar.write("LeanFlow")

st.sidebar.markdown("""
    <h1 style='display: inline; font-size: 28px;'>LeanFlow</h1>
    <h2 style='display: inline; font-size: 18px;'>➤</h2>
    """, unsafe_allow_html=True)
st.sidebar.markdown("""---""")

# =====================================
# Parte 2: Upload de Arquivo e Carregamento dos Dados
# =====================================

# Inicializar missing_sheets
missing_sheets = []

# Upload de Arquivos na Sidebar
st.sidebar.subheader("📂 Upload de Arquivos Excel")
uploaded_file = st.sidebar.file_uploader("Faça upload de um arquivo Excel", type=["xlsx"])

# Verificação se o arquivo foi carregado
if uploaded_file:
    try:
        # Ler o arquivo Excel
        xls = pd.ExcelFile(uploaded_file)

        # Verificar se as abas estão corretas
        missing_sheets = [sheet for sheet in ABAS.values() if sheet not in xls.sheet_names]
        if missing_sheets:
            st.error(f"As seguintes abas estão faltando no arquivo: {', '.join(missing_sheets)}")
        else:
            # Carregar os dados das abas usando as constantes
            # Dados Gerais
            df_mensal = pd.read_excel(xls, sheet_name=ABAS["MENSAL"])
            df_semana = pd.read_excel(xls, sheet_name=ABAS["SEMANAL"])
            df_horarios = pd.read_excel(xls, sheet_name=ABAS["HORA"])
            df_hv = pd.read_excel(xls, sheet_name=ABAS["HORIZ_VERTIC"])
            df_pontos_cuidado = pd.read_excel(xls, sheet_name=ABAS["PONTOS_CUIDADO"])
            df_classificacao = pd.read_excel(xls, sheet_name=ABAS["CLASSIFICACAO"])
            df_retorno = pd.read_excel(xls, sheet_name=ABAS["RETORNO"])
            df_saida = pd.read_excel(xls, sheet_name=ABAS["SAIDA"])
            df_orientados = pd.read_excel(xls, sheet_name=ABAS["ORIENTADOS"])

            # Dados de Triagem
            df_triagem_urgencia = pd.read_excel(xls, sheet_name=ABAS["TRIAGEM_URGENCIA"])
            df_triagem_enfermeiros = pd.read_excel(xls, sheet_name=ABAS["TRIAGEM_ENFERMEIROS"])
            df_triagem_salas = pd.read_excel(xls, sheet_name=ABAS["TRIAGEM_SALAS"])
            df_triagem_tempo = pd.read_excel(xls, sheet_name=ABAS["TRIAGEM_TEMPO"])

            # Dados de SADT
            df_exames_sadt = pd.read_excel(xls, sheet_name=ABAS["EXAMES_SADT"])

            # Dados de Consulta
            df_consulta_tempo = pd.read_excel(xls, sheet_name=ABAS["CONSULTA_TEMPO"])
            df_media_medicos_consulta = pd.read_excel(xls, sheet_name=ABAS["MEDIA_MEDICOS_CONSULTA"])
            df_dados_semanais_medicos = pd.read_excel(xls, sheet_name=ABAS["DADOS_SEMANAIS_MEDICOS"])
            df_media_medicos_especialidade = pd.read_excel(
                xls,
                sheet_name=ABAS["MEDIA_MEDICOS_ESPECIALIDADE"],
                usecols="A:C"  # Inclui as colunas A até C
            )

            # Converter a coluna 'percentual_atendimento_dia' para float
            df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["PERCENTUAL_ATENDIMENTO_DIA"]] = df_media_medicos_especialidade[
                COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["PERCENTUAL_ATENDIMENTO_DIA"]
            ].apply(porcentagem_para_float)

            # Converter a coluna 'quantidade_media_dia_medicos' para numérico
            df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["QUANTIDADE_MEDIA_MEDICOS"]] = pd.to_numeric(
                df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["QUANTIDADE_MEDIA_MEDICOS"]],
                errors='coerce'
            ).fillna(0)

            # Dados do Centro Cirúrgico
            df_funcionamento_cc = pd.read_excel(xls, sheet_name=ABAS["FUNCIONAMENTO_CC"])
            df_motivos_cancelamento = pd.read_excel(xls, sheet_name=ABAS["MOTIVOS_CANCELAMENTO"])
            df_tempo_medio_atraso_primeira = pd.read_excel(xls, sheet_name=ABAS["TEMPO_ATRASO_CIRURGIA"])
            df_tempo_setup_sala = pd.read_excel(xls, sheet_name=ABAS["TEMPO_SETUP_SALA"])
            df_tempo_substit_sala = pd.read_excel(xls, sheet_name=ABAS["TEMPO_SUBSTIT_SALA"])
            df_media_horas_agendadas = pd.read_excel(xls, sheet_name=ABAS["MEDIA_HORAS_AGENDADAS"])
            df_media_horas_gastas = pd.read_excel(xls, sheet_name=ABAS["MEDIA_HORAS_GASTAS"])
            df_taxa_indicadores_cc = pd.read_excel(xls, sheet_name=ABAS["TAXA_INDICADORES_CC"])
            df_motivos_atraso_cirurgia = pd.read_excel(xls, sheet_name=ABAS["MOTIVOS_ATRASO_CIRURGIA"])
            df_tempo_permanencia_leitos = pd.read_excel(xls, sheet_name=ABAS["TEMPO_PERMANENCIA_LEITOS"])
            df_classificacao_salas_cirurgicas = pd.read_excel(xls, sheet_name=ABAS["CLASSIFICACAO_SALAS_CIRURGICAS"])
            df_salas_cirurgicas_porte = pd.read_excel(xls, sheet_name=ABAS["SALAS_CIRURGICAS_PORTE"])
            df_qtd_cirurgia_eletivas_espec = pd.read_excel(xls, sheet_name=ABAS["QTD_CIRURGIAS_ELETIVAS_ESPEC"])
            df_qtd_cirurgias_nao_programadas = pd.read_excel(xls, sheet_name=ABAS["QTD_CIRURGIAS_NAO_PROGRAMADAS"])
            df_tempo_medio_solicitacao_cirurgi = pd.read_excel(xls, sheet_name=ABAS["TEMPO_MEDIO_SOLICITACAO_CIRURGIA"])
            df_media_medicos_cc = pd.read_excel(xls, sheet_name=ABAS["MEDIA_MEDICOS_CC"])
            df_cirurgias_mes = pd.read_excel(xls, sheet_name=ABAS["CIRURGIAS_MES"])

            # Passagem & Internação
            df_passagem_setores = pd.read_excel(xls, sheet_name=ABAS["PASSAGEM_SETORES"])
            df_internacao_demanda = pd.read_excel(xls, sheet_name=ABAS["INTERNACAO_DEMANDA"])
            df_internacao_saida = pd.read_excel(xls, sheet_name=ABAS["INTERNACAO_SAIDA"])
            df_taxa_internacao = pd.read_excel(xls, sheet_name=ABAS["TAXA_INTERNACAO"])

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {str(e)}")
        st.stop() 
        
#=====================================================================
# Título da Página
#=====================================================================

st.title("🔎🏥 Diagnóstico de Processos Hospitalares")

# Verificar se o arquivo foi carregado corretamente
if uploaded_file and not missing_sheets:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        '🚪 Porta',
        '🩺 Triagem',
        '👨‍⚕️ Consulta',
        '🧪 SADT',
        '⏩ Passagem & Internação',
        '🏩 Centro Cirúrgico',
        '🔗 Desempenho dos Processos'
    ])
else:
    st.info("Por favor, carregue o arquivo Excel com os dados necessários para visualizar as análises.")

# =====================================
# Parte 3: Preparação dos Dados
# =====================================

if uploaded_file and not missing_sheets:

    # Preparar dados de 'df_horarios'
    df_horarios[COLUNAS["HORA"]["QUANTIDADE_MEDIA"]] = pd.to_numeric(
        df_horarios[COLUNAS["HORA"]["QUANTIDADE_MEDIA"]], errors='coerce').fillna(0)
    df_horarios["quantidade_media_pacientes (arredondado)"] = df_horarios[
        COLUNAS["HORA"]["QUANTIDADE_MEDIA"]].apply(lambda x: math.ceil(x))

    # Extrair a hora do 'horario' em 'df_horarios'
    df_horarios['hora'] = df_horarios[COLUNAS["HORA"]["HORARIO"]].astype(str).str.split(':').str[0]
    df_horarios['hora'] = pd.to_numeric(df_horarios['hora'], errors='coerce')
    df_horarios['Período'] = df_horarios['hora'].apply(definir_periodo)

    # Preparar dados de 'df_triagem_enfermeiros'
    df_triagem_enfermeiros[COLUNAS["TRIAGEM_ENFERMEIROS"]["MEDIA_ENFERMEIROS"]] = pd.to_numeric(
        df_triagem_enfermeiros[COLUNAS["TRIAGEM_ENFERMEIROS"]["MEDIA_ENFERMEIROS"]], errors='coerce').fillna(0)
    df_triagem_enfermeiros["quantidade_media_enfermeiros (arredondado)"] = df_triagem_enfermeiros[
        COLUNAS["TRIAGEM_ENFERMEIROS"]["MEDIA_ENFERMEIROS"]].apply(lambda x: math.ceil(x))

    # Extrair a hora do 'horario_triagem' em 'df_triagem_enfermeiros'
    df_triagem_enfermeiros['hora'] = df_triagem_enfermeiros[
        COLUNAS["TRIAGEM_ENFERMEIROS"]["HORARIO"]].astype(str).str.split(':').str[0]
    df_triagem_enfermeiros['hora'] = pd.to_numeric(df_triagem_enfermeiros['hora'], errors='coerce')
    df_triagem_enfermeiros['Período'] = df_triagem_enfermeiros['hora'].apply(definir_periodo)

    # Preparar dados de 'df_exames_sadt'
    df_exames_sadt[COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"]] = pd.to_numeric(
        df_exames_sadt[COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"]], errors='coerce')
    df_exames_sadt[COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]] = pd.to_numeric(
        df_exames_sadt[COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]], errors='coerce')
    df_exames_sadt.dropna(subset=[
        COLUNAS["EXAMES_SADT"]["TIPO_EXAME"],
        COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"],
        COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]
    ], inplace=True)

    # Calcular o total de pacientes que realizaram exames
    total_pacientes_exames = df_exames_sadt[
        COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]].sum()

    # Evitar divisão por zero
    if total_pacientes_exames > 0:
        df_exames_sadt['percentual_pacientes'] = (
            df_exames_sadt[COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]] /
            total_pacientes_exames) * 100
    else:
        df_exames_sadt['percentual_pacientes'] = 0

    # Ordenar o DataFrame após calcular o percentual
    df_exames_sadt_sorted = df_exames_sadt.sort_values(
        by=COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"], ascending=False)

    # =====================================
    # Preparação dos dados para a aba "Consulta"
    # =====================================

    # Remover linhas com valores nulos nas colunas 'ETAPA' e 'TEMPO_MEDIO_ETAPA'
    df_consulta_tempo.dropna(subset=[
        COLUNAS["CONSULTA_TEMPO"]["ETAPA"],
        COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]
    ], inplace=True)

    # Preparar dados de 'df_media_medicos_consulta'
    df_media_medicos_consulta[COLUNAS["MEDIA_MEDICOS_CONSULTA"]["QUANTIDADE_MEDIA_MEDICOS"]] = pd.to_numeric(
        df_media_medicos_consulta[COLUNAS["MEDIA_MEDICOS_CONSULTA"]["QUANTIDADE_MEDIA_MEDICOS"]],
        errors='coerce').fillna(0)

    # Extrair a hora e converter para numérico
    df_media_medicos_consulta['hora'] = df_media_medicos_consulta[
        COLUNAS["MEDIA_MEDICOS_CONSULTA"]["HORARIO"]].astype(str).str.split(':').str[0]
    df_media_medicos_consulta['hora'] = pd.to_numeric(
        df_media_medicos_consulta['hora'], errors='coerce')

    # Aplicar a função 'definir_periodo' para definir o período baseado na hora
    df_media_medicos_consulta['Período'] = df_media_medicos_consulta['hora'].apply(definir_periodo)

    # Preparar dados de 'df_dados_semanais_medicos'
    df_dados_semanais_medicos[COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_MANHA_TARDE"]] = pd.to_numeric(
        df_dados_semanais_medicos[COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_MANHA_TARDE"]],
        errors='coerce').fillna(0)
    df_dados_semanais_medicos[COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_NOITE_MADRUGADA"]] = pd.to_numeric(
        df_dados_semanais_medicos[COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_NOITE_MADRUGADA"]],
        errors='coerce').fillna(0)

    # Preparar dados de 'df_media_medicos_especialidade'
    # Remover linhas onde 'especialidade' é NaN ou vazia
    df_media_medicos_especialidade = df_media_medicos_especialidade[
        df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["ESPECIALIDADE"]].notna() &
        (df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["ESPECIALIDADE"]]
         .astype(str).str.strip() != '')
    ]

    # Converter 'quantidade_media_dia_medicos' para numérico, preenchendo NaN com 0
    df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["QUANTIDADE_MEDIA_MEDICOS"]] = pd.to_numeric(
        df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["QUANTIDADE_MEDIA_MEDICOS"]],
        errors='coerce').fillna(0)

    # Preparar df_passagem_setores
    df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["TAXA_OCUPACAO"]] = df_passagem_setores[
        COLUNAS["PASSAGEM_SETORES"]["TAXA_OCUPACAO"]].apply(porcentagem_para_float)

    # Converter colunas numéricas para o tipo correto em df_passagem_setores
    df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]] = pd.to_numeric(
        df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]], errors='coerce')
    df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]] = pd.to_numeric(
        df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]], errors='coerce')

    # Preparar df_tempo_permanencia_leitos
    df_tempo_permanencia_leitos[COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["QUANTIDADE_DE_LEITO"]] = pd.to_numeric(
        df_tempo_permanencia_leitos[COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["QUANTIDADE_DE_LEITO"]], errors='coerce')
    df_tempo_permanencia_leitos[COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"]] = pd.to_numeric(
        df_tempo_permanencia_leitos[COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"]], errors='coerce')

    # Preparar df_internacao_demanda
    df_internacao_demanda[COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]] = pd.to_numeric(
        df_internacao_demanda[COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]], errors='coerce')

    # Preparar df_internacao_saida
    df_internacao_saida[COLUNAS["INTERNACAO_SAIDA"]["MEDIA_SAIDA_DIA"]] = pd.to_numeric(
        df_internacao_saida[COLUNAS["INTERNACAO_SAIDA"]["MEDIA_SAIDA_DIA"]], errors='coerce')

    # Preparar df_taxa_internacao
    df_taxa_internacao[COLUNAS["TAXA_INTERNACAO"]["RESULTADO_PERCENTUAL"]] = df_taxa_internacao[
        COLUNAS["TAXA_INTERNACAO"]["RESULTADO_PERCENTUAL"]].apply(porcentagem_para_float)

    # Cálculos das Métricas

    try:
        # 1. FATOR DE UTILIZAÇÃO DO HOSPITAL – GERAL
        tempo_medio_permanencia_geral = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "Geral", COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]].iloc[0]
        media_solicitacoes_leitos_geral = df_internacao_demanda.loc[df_internacao_demanda[COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"]] == "Leitos Geral", COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]].iloc[0]
        total_leitos_geral = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "Geral", COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]].iloc[0]

        fator_utilizacao_geral = (tempo_medio_permanencia_geral * media_solicitacoes_leitos_geral) / total_leitos_geral

        # 2. FATOR DE UTILIZAÇÃO DOS LEITOS DISPONÍVEIS PARA O P.A. (ENF.)
        tempo_medio_permanencia_pa_enf = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "P.A. (ENF.)", COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]].iloc[0]
        media_solicitacoes_leitos_enfermaria = df_internacao_demanda.loc[df_internacao_demanda[COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"]] == "Leitos Enfermaria", COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]].iloc[0]
        total_leitos_pa_enf = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "P.A. (ENF.)", COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]].iloc[0]

        fator_utilizacao_pa_enf = (tempo_medio_permanencia_pa_enf * media_solicitacoes_leitos_enfermaria) / total_leitos_pa_enf

        # 3. FATOR DE UTILIZAÇÃO DOS LEITOS DISPONÍVEIS PARA O P.A. (UTI)
        tempo_medio_permanencia_pa_uti = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "P.A. (UTI)", COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]].iloc[0]
        media_solicitacoes_leitos_uti = df_internacao_demanda.loc[df_internacao_demanda[COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"]] == "Leitos UTI", COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]].iloc[0]
        total_leitos_pa_uti = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "P.A. (UTI)", COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]].iloc[0]

        fator_utilizacao_pa_uti = (tempo_medio_permanencia_pa_uti * media_solicitacoes_leitos_uti) / total_leitos_pa_uti

        # 4. FATOR DE UTILIZAÇÃO DOS LEITOS DISPONÍVEIS PARA O P.A. (CLÍNICOS)
        tempo_medio_permanencia_pa_clinicos = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "P.A. (CLÍNICOS)", COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]].iloc[0]
        media_solicitacoes_leitos_cirurgicos = df_internacao_demanda.loc[df_internacao_demanda[COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"]] == "Leitos Cirúrgicos", COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]].iloc[0]
        media_solicitacoes_leitos_enfermaria = df_internacao_demanda.loc[df_internacao_demanda[COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"]] == "Leitos Enfermaria", COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]].iloc[0]
        media_solicitacoes_leitos_clinicos = media_solicitacoes_leitos_enfermaria - media_solicitacoes_leitos_cirurgicos
        total_leitos_pa_clinicos = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "P.A. (CLÍNICOS)", COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]].iloc[0]

        fator_utilizacao_pa_clinicos = (tempo_medio_permanencia_pa_clinicos * media_solicitacoes_leitos_clinicos) / total_leitos_pa_clinicos

        # 5. FATOR DE UTILIZAÇÃO DOS LEITOS DISPONÍVEIS PARA O P.A. (CIRÚRGICOS)
        tempo_medio_permanencia_pa_cirurgicos = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "P.A. (CIRÚRGICOS)", COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]].iloc[0]
        media_solicitacoes_leitos_cirurgicos = df_internacao_demanda.loc[df_internacao_demanda[COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"]] == "Leitos Cirúrgicos", COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]].iloc[0]
        total_leitos_pa_cirurgicos = df_passagem_setores.loc[df_passagem_setores[COLUNAS["PASSAGEM_SETORES"]["SETORES"]] == "P.A. (CIRÚRGICOS)", COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]].iloc[0]

        fator_utilizacao_pa_cirurgicos = (tempo_medio_permanencia_pa_cirurgicos * media_solicitacoes_leitos_cirurgicos) / total_leitos_pa_cirurgicos
        pass
    except Exception as e:
        st.error(f"Ocorreu um erro ao calcular as métricas: {e}")
        st.stop()

# =====================================
# Parte 5: Aba "Porta de Entrada"
# =====================================

    with tab1:
        st.markdown("Nesta seção, você poderá analisar os dados referentes à porta de entrada do hospital, incluindo a quantidade de pacientes atendidos por mês e por hora, além da distribuição entre pacientes horizontais e verticais. Essas informações são essenciais para entender o fluxo de pacientes e otimizar os recursos hospitalares.")

        # Cálculos principais
        total_pacientes_ano = df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]].sum()
        total_pacientes_dia = df_horarios[COLUNAS["HORA"]["QUANTIDADE_MEDIA"]].sum()
        total_pacientes_semana = df_semana[COLUNAS["SEMANAL"]["QUANTIDADE_MEDIA"]].sum()
        total_pacientes_dia_safe = total_pacientes_dia if total_pacientes_dia > 0 else 1  # Proteção contra divisão por zero

        # Quantidade de Pontos de Cuidado
        quantidade_pontos_cuidado = df_pontos_cuidado[COLUNAS["PONTOS_CUIDADO"]["QUANTIDADE"]].sum()

        # Extração dos indicadores complementares
        evasao_percent = (
            df_saida.loc[df_saida[COLUNAS["SAIDA"]["INDICADORES"]] == 'EVASÃO', COLUNAS["SAIDA"]["QUANTIDADE_MEDIA"]].iloc[0]
            / total_pacientes_dia_safe * 100
        )
        abandono_percent = (
            df_saida.loc[df_saida[COLUNAS["SAIDA"]["INDICADORES"]] == 'ABANDONO', COLUNAS["SAIDA"]["QUANTIDADE_MEDIA"]].iloc[0]
            / total_pacientes_dia_safe * 100
        )
        orientados_percent = (
            df_orientados[COLUNAS["ORIENTADOS"]["QUANTIDADE_MEDIA"]].iloc[0]
            / total_pacientes_dia_safe * 100
        )
        retorno_48h_percent = (
            df_retorno.loc[df_retorno[COLUNAS["RETORNO"]["INDICADORES"]] == 'RETORNO EM 48 HORAS', COLUNAS["RETORNO"]["QUANTIDADE_MEDIA"]].iloc[0]
            / total_pacientes_dia_safe * 100
        )
        retorno_72h_percent = (
            df_retorno.loc[df_retorno[COLUNAS["RETORNO"]["INDICADORES"]] == 'RETORNO EM 72 HORAS', COLUNAS["RETORNO"]["QUANTIDADE_MEDIA"]].iloc[0]
            / total_pacientes_dia_safe * 100
        )

        st.markdown("#### 📊 Métricas Gerais")

        # Exibição das métricas
        col1, col2, col3, col4, _ = st.columns(5, gap="small")
        with col1:
            st.metric("Atendimentos/Dia", f"{total_pacientes_dia:.2f}")
        with col2:
            st.metric("Atendimentos/Semana", f"{total_pacientes_semana:.2f}")
        with col3:
            st.metric("Atendimentos/Ano", f"{total_pacientes_ano:.0f}")
        with col4:
            st.metric("Qtd de Pontos de Cuidado", f"{quantidade_pontos_cuidado:.0f}")

        col5, col6, col7, col8, col9 = st.columns(5, gap="small")
        with col5:
            st.metric("Orientados p/Rede (%)", f"{orientados_percent:.2f}%")
        with col6:
            st.metric("Taxa Evasão (%)", f"{evasao_percent:.2f}%")
        with col7:
            st.metric("Taxa Abandono (%)", f"{abandono_percent:.2f}%")
        with col8:
            st.metric("Retorno em 48h (%)", f"{retorno_48h_percent:.2f}%")
        with col9:
            st.metric("Retorno em 72h (%)", f"{retorno_72h_percent:.2f}%")

        st.markdown("""---""")
        
# =====================================
# Parte 6: Aba "Porta de Entrada" - Análises Gráficas
# =====================================

        st.markdown("#### 📈 Análises Gráficas")

        # Gráfico 1: Quantidade de Pacientes por Mês
        st.markdown("###### 1️⃣ Quantidade de Pacientes por Mês")
        
        # Mapeamento dos meses em português para números
        mes_map = {
            'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Mai': 5, 'Jun': 6,
            'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12
        }
        
        # Garantir que o DataFrame possui as colunas necessárias
        df_mensal = df_mensal.copy()
        df_mensal['mes_num'] = df_mensal[COLUNAS["MENSAL"]["MES"]].map(mes_map)
        df_mensal['ano'] = df_mensal[COLUNAS["MENSAL"]["ANO"]]
        
        # Criar a coluna 'data' combinando ano e número do mês
        df_mensal['data'] = pd.to_datetime(
            df_mensal[['ano', 'mes_num']].rename(columns={'ano': 'year', 'mes_num': 'month'}).assign(day=1)
        )
        
        # Ordenar o DataFrame pela coluna 'data'
        df_mensal.sort_values('data', inplace=True)
        
        # Criar a coluna 'mes_ano_pt' com os meses em português
        mes_num_map = {
            1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
        }
        df_mensal['mes_ano_pt'] = df_mensal['data'].dt.month.map(mes_num_map) + '/' + df_mensal['data'].dt.year.astype(str)
        
        # Calcular a média anual
        media_anual = df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]].mean()
        
        # Calcular a linha de tendência
        import statsmodels.api as sm
        
        df_mensal['x_numeric'] = range(len(df_mensal))
        X = sm.add_constant(df_mensal['x_numeric'])
        model = sm.OLS(df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]], X).fit()
        df_mensal['trend'] = model.predict(X)
        
        # Criar o gráfico de barras
        fig_barras = go.Figure()
        
        # Adicionar as barras
        fig_barras.add_trace(go.Bar(
            x=df_mensal['mes_ano_pt'],
            y=df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]],
            marker_color=df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]],
            marker=dict(color=df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]], colorscale='Blues'),
            name='Quantidade de Pacientes',
            text=df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]],
            textposition='inside',
            texttemplate='%{text:.0f}'
        ))
        
        # Adicionar a linha da média anual
        fig_barras.add_trace(go.Scatter(
            x=df_mensal['mes_ano_pt'],
            y=[media_anual]*len(df_mensal),
            mode='lines',
            line=dict(color='Red', width=2, dash='dash'),
            name='Média Anual'
        ))
        
        # Adicionar a linha de tendência
        fig_barras.add_trace(go.Scatter(
            x=df_mensal['mes_ano_pt'],
            y=df_mensal['trend'],
            mode='lines',
            line=dict(color='green', width=2),
            name='Tendência'
        ))
        
        # Adicionar anotação para o valor médio
        fig_barras.add_annotation(
            x=df_mensal['mes_ano_pt'].iloc[-1],
            y=media_anual,
            text=f"Média: {media_anual:.2f}",
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            yshift=10,
            font=dict(color="Red")
        )
        
        # Atualizar o layout
        fig_barras.update_layout(
            xaxis_title='Mês/Ano',
            yaxis_title='Quantidade de Pacientes',
            xaxis_tickangle=-45,
            template='plotly_white',
            coloraxis_showscale=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        st.plotly_chart(fig_barras, use_container_width=True)
        
        # Observação analítica
        st.markdown("**Observação analítica:**")
        
        pacientes_max = df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]].max()
        pacientes_min = df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]].min()
        media_anual = df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]].mean()
        
        mes_max = df_mensal[df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]] == pacientes_max]['mes_ano_pt'].values
        mes_min = df_mensal[df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]] == pacientes_min]['mes_ano_pt'].values
        
        meses_acima_media = df_mensal[df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]] > media_anual]
        meses_abaixo_media = df_mensal[df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]] <= media_anual]
        
        if len(meses_acima_media) > 0:
            st.write(f"{len(meses_acima_media)} meses ficaram acima da média anual, indicando sazonalidade ou aumento da demanda em períodos específicos.")
        if len(meses_abaixo_media) > 0:
            st.write(f"{len(meses_abaixo_media)} meses ficaram abaixo da média anual, sugerindo uma queda na demanda durante esses meses.")
        if pacientes_max > (media_anual * 1.2):
            st.write(f"O mês com o maior pico de atendimentos foi **{', '.join(mes_max)}**, com um volume {pacientes_max / media_anual:.2f} vezes maior do que a média anual.")
        if pacientes_min < (media_anual * 0.8):
            st.write(f"O mês com o menor volume de atendimentos foi **{', '.join(mes_min)}**, com uma queda de {media_anual - pacientes_min:.0f} pacientes em relação à média.")
        
        st.markdown("---")
        
        # Gráfico 2: Média de Chegada de Pacientes por Dia da Semana
        st.markdown("###### 2️⃣ Média de Chegada de Pacientes por Dia da Semana")
        fig_semana = px.bar(
            df_semana,
            x=COLUNAS["SEMANAL"]["DIA"],
            y=COLUNAS["SEMANAL"]["QUANTIDADE_MEDIA"],
            color=COLUNAS["SEMANAL"]["QUANTIDADE_MEDIA"],
            color_continuous_scale='Blues',
            text=COLUNAS["SEMANAL"]["QUANTIDADE_MEDIA"]
        )
        
        fig_semana.update_traces(texttemplate='%{text:.1f}', textposition='inside')
        
        # Adicionar linha de tendência
        fig_semana.add_trace(
            go.Scatter(
                x=df_semana[COLUNAS["SEMANAL"]["DIA"]],
                y=df_semana[COLUNAS["SEMANAL"]["QUANTIDADE_MEDIA"]].rolling(window=2, min_periods=1).mean(),
                mode='lines+markers',
                name='Tendência',
                line=dict(color='red')
            )
        )
        st.plotly_chart(fig_semana, use_container_width=True)
        
        st.markdown("**Observação analítica:**")
        dia_pico = df_semana[COLUNAS["SEMANAL"]["QUANTIDADE_MEDIA"]].idxmax()
        st.write(f"O maior volume de atendimentos ocorre na {df_semana[COLUNAS['SEMANAL']['DIA']][dia_pico]}, o que pode indicar maior demanda hospitalar nos dias de semana comparado aos finais de semana.")
        
        st.markdown("---")
        
        # Gráfico 3: Média de Chegada de Pacientes por Hora
        st.markdown("###### 3️⃣ Média de Chegada de Pacientes por Hora")
        
        with st.container():
            fig_horarios = px.bar(
                df_horarios,
                x='hora',
                y="quantidade_media_pacientes (arredondado)",
                color="quantidade_media_pacientes (arredondado)",
                color_continuous_scale='Blues',
                text="quantidade_media_pacientes (arredondado)"
            )
            fig_horarios.update_traces(texttemplate='%{text:.1f}', textposition='inside')
            st.plotly_chart(fig_horarios, use_container_width=True)
        
        st.markdown("**Observação analítica:**")
        pico_hora = df_horarios.loc[df_horarios["quantidade_media_pacientes (arredondado)"].idxmax(), 'hora']
        st.write(f"A hora com maior média de chegada de pacientes é **{int(pico_hora)}h**, indicando um pico nesse horário.")
        st.write("Isso sugere que recursos adicionais podem ser necessários para atender à demanda nesse período.")
        
        st.markdown("---")
        
        # Gráfico 4: Distribuição de Pacientes por Período e Hora (Sunburst)
        st.markdown("###### 4️⃣ Distribuição de Pacientes por Período e Hora")
        
        if df_horarios['quantidade_media_pacientes (arredondado)'].sum() > 0:
            with st.container():
                fig_sunburst_pacientes = px.sunburst(
                    df_horarios,
                    path=['Período', 'hora'],
                    values="quantidade_media_pacientes (arredondado)",
                    color="quantidade_media_pacientes (arredondado)",
                    color_continuous_scale='Blues',
                    branchvalues="total"
                )
                fig_sunburst_pacientes.update_traces(textinfo='label+percent entry')
                st.plotly_chart(fig_sunburst_pacientes, use_container_width=True)
        
            st.markdown("**Observação analítica do período:**")
            pico_periodo = df_horarios.groupby('Período')["quantidade_media_pacientes (arredondado)"].sum().idxmax()
            st.write(f"O período com maior fluxo de pacientes é **{pico_periodo}**.")
            st.write("Isso ajuda na alocação de recursos humanos e materiais conforme a demanda.")
        else:
            st.warning("Não há dados suficientes para gerar o gráfico de sunburst.")
        
        st.markdown("---")
        
        # Gráfico 5: Distribuição de Pacientes por Classificação
        st.markdown("###### 5️⃣ Distribuição de Pacientes por Classificação")
        
        df_classificacao_sorted = df_classificacao.sort_values(
            by=COLUNAS["CLASSIFICACAO"]["QUANTIDADE_PACIENTES"], ascending=False
        )
        
        fig_classificacao = px.bar(
            df_classificacao_sorted,
            x=COLUNAS["CLASSIFICACAO"]["CLASSIFICACAO"],
            y=COLUNAS["CLASSIFICACAO"]["QUANTIDADE_PACIENTES"],
            color=COLUNAS["CLASSIFICACAO"]["QUANTIDADE_PACIENTES"],
            color_continuous_scale='Blues',
            labels={
                COLUNAS["CLASSIFICACAO"]["CLASSIFICACAO"]: 'Classificação',
                COLUNAS["CLASSIFICACAO"]["QUANTIDADE_PACIENTES"]: 'Quantidade de Pacientes'
            },
            template='plotly_white',
            text=COLUNAS["CLASSIFICACAO"]["QUANTIDADE_PACIENTES"]
        )
        
        fig_classificacao.update_traces(texttemplate='%{text:.0f}', textposition='inside')
        fig_classificacao.update_layout(
            xaxis_title='Classificação',
            yaxis_title='Quantidade de Pacientes',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_classificacao, use_container_width=True)
        
        st.markdown("---")
        
        # Gráfico 6: Detalhamento dos Pontos de Cuidado por Local e Ponto de Cuidado (TreeMap)
        st.markdown("###### 6️⃣ Detalhamento dos Pontos de Cuidado por Local e Ponto de Cuidado")
        
        df_pontos_por_local = df_pontos_cuidado.groupby('local')['quantidade'].sum().reset_index()
        df_pontos_por_local = df_pontos_por_local.sort_values(by='quantidade', ascending=False)
        
        with st.container():
            df_pontos_cuidado_nonzero = df_pontos_cuidado[df_pontos_cuidado['quantidade'] > 0]
        
            fig_pontos_detalhado = px.treemap(
                df_pontos_cuidado_nonzero,
                path=['local', 'ponto_cuidado'],
                values='quantidade',
                color='quantidade',
                color_continuous_scale='Blues'
            )
            fig_pontos_detalhado.update_traces(textinfo='label+value')
            st.plotly_chart(fig_pontos_detalhado, use_container_width=True)
        
        st.markdown("---")
        st.markdown("---")

        # 7️⃣ Gráfico de Pizza: Pacientes Horizontais vs Verticais
        
        # Gráfico 7: Distribuição de Pacientes Horizontais vs Verticais
        st.markdown("###### 7️⃣ Distribuição de Pacientes Horizontais vs Verticais")
        labels = ['Horizontais', 'Verticais']
        values = [
            df_hv.loc[df_hv[COLUNAS["HORIZ_VERTIC"]["CARACTERISTICA"]] == 'Horizontais', COLUNAS["HORIZ_VERTIC"]["QUANTIDADE_MEDIA"]].sum(),
            df_hv.loc[df_hv[COLUNAS["HORIZ_VERTIC"]["CARACTERISTICA"]] == 'Verticais', COLUNAS["HORIZ_VERTIC"]["QUANTIDADE_MEDIA"]].sum()
        ]
        
        if sum(values) > 0:
            fig_pizza = px.pie(
                names=labels,
                values=values,
            )
            fig_pizza.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pizza, use_container_width=True)
        
            st.markdown("**Observação analítica:**")
            total_pacientes_hv = sum(values)
            percent_horizontais = (values[0] / total_pacientes_hv) * 100
            percent_verticais = (values[1] / total_pacientes_hv) * 100
            st.write(f"Pacientes horizontais representam {percent_horizontais:.2f}% e pacientes verticais {percent_verticais:.2f}% do total diário, o que pode ser útil para a gestão de leitos.")
        else:
            st.warning("Não há dados suficientes para gerar o gráfico de pacientes horizontais vs verticais.")
        
        st.markdown("---")
        
        # Gráfico 8: Pacientes Acumulados ao Longo do Ano
        st.markdown("###### 8️⃣ Pacientes Acumulados ao Longo do Ano")
        
        # Certificar-se de que 'data' e 'mes_ano_pt' já foram criados
        if 'data' not in df_mensal.columns or 'mes_ano_pt' not in df_mensal.columns:
            # Mapeamento dos meses em português para números
            mes_map = {
                'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4, 'Mai': 5, 'Jun': 6,
                'Jul': 7, 'Ago': 8, 'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12
            }
            df_mensal['mes_num'] = df_mensal[COLUNAS["MENSAL"]["MES"]].map(mes_map)
            df_mensal['ano'] = df_mensal[COLUNAS["MENSAL"]["ANO"]]
            df_mensal['data'] = pd.to_datetime(
                df_mensal[['ano', 'mes_num']].rename(columns={'ano': 'year', 'mes_num': 'month'}).assign(day=1)
            )
            df_mensal.sort_values('data', inplace=True)
            mes_num_map = {
                1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
            }
            df_mensal['mes_ano_pt'] = df_mensal['data'].dt.month.map(mes_num_map) + '/' + df_mensal['data'].dt.year.astype(str)
        
        # Calcular o total acumulado de pacientes
        df_mensal['pacientes_acumulados'] = df_mensal[COLUNAS["MENSAL"]["QUANTIDADE_PACIENTES"]].cumsum()
        
        # Criar o gráfico de linha
        fig_acumulado = go.Figure()
        
        fig_acumulado.add_trace(go.Scatter(
            x=df_mensal['mes_ano_pt'],
            y=df_mensal['pacientes_acumulados'],
            mode='lines+markers+text',
            line=dict(color='blue'),
            name='Pacientes Acumulados',
            text=df_mensal['pacientes_acumulados'],
            textposition="top center",
            texttemplate='%{text:.0f}'
        ))
        
        # Atualizar o layout
        fig_acumulado.update_layout(
            xaxis_title='Mês/Ano',
            yaxis_title='Pacientes Acumulados',
            xaxis_tickangle=-45,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig_acumulado, use_container_width=True)
        
        # Observação analítica baseada em condições
        st.markdown("**Observação analítica condicional:**")
        
        col_acumulado = 'pacientes_acumulados'
        
        mes_maior_pico = df_mensal.loc[df_mensal[col_acumulado].idxmax(), 'mes_ano_pt']
        valor_maior_pico = df_mensal[col_acumulado].max()
        
        crescimento_diferenca = df_mensal[col_acumulado].diff().fillna(0)
        meses_sem_crescimento = df_mensal.loc[crescimento_diferenca <= 0, 'mes_ano_pt']
        
        media_acumulado = df_mensal[col_acumulado].mean()
        if valor_maior_pico > media_acumulado * 1.2:
            st.write(f"O maior pico de pacientes acumulados ocorreu no mês de **{mes_maior_pico}**, com um aumento expressivo de pacientes.")
        else:
            st.write(f"O fluxo de pacientes acumulados foi relativamente estável ao longo do período, sem picos extremos.")
        
        if not meses_sem_crescimento.empty:
            meses_lista = meses_sem_crescimento.tolist()
            st.write(f"Os seguintes meses apresentaram crescimento acumulado estável ou queda: {', '.join(meses_lista)}.")
        else:
            st.write("Não houve períodos significativos de queda ou estagnação no número acumulado de pacientes ao longo do período.")
        
        if crescimento_diferenca.mean() > 0:
            st.write("O crescimento contínuo ao longo dos meses sugere uma demanda hospitalar consistente.")
        else:
            st.write("Houve interrupções no crescimento acumulado de pacientes ao longo do período, o que pode refletir em períodos de baixa demanda.")
        
        st.markdown("---")
        
        # Gráfico 9: Estimativa Anual: Comparativo entre Diferentes Métodos
        st.markdown("###### 9️⃣ Estimativa Anual: Comparativo entre Diferentes Métodos")
        
        # Estimativas anuais baseadas nas médias diária, semanal e mensal
        estimativa_anual_dia = total_pacientes_dia * 365
        media_diaria_semana = total_pacientes_semana / 7
        estimativa_anual_semana = media_diaria_semana * 365
        estimativa_anual_mes = total_pacientes_ano
        
        # Gráfico de barras comparando as diferentes estimativas
        fig_comparacao = px.bar(
            x=['Estimativa Anual (Média/Dia)', 'Estimativa Anual (Média/Semana)', 'Estimativa Anual (Volumetria/Mês)'],
            y=[estimativa_anual_dia, estimativa_anual_semana, estimativa_anual_mes],
            labels={"x": "Método", "y": "Estimativa Anual de Pacientes"},
            text=[estimativa_anual_dia, estimativa_anual_semana, estimativa_anual_mes]
        )
        fig_comparacao.update_traces(texttemplate='%{text:.0f}', textposition='inside')
        st.plotly_chart(fig_comparacao, use_container_width=True)
        
        # Observação analítica aprimorada
        st.markdown("**Observação analítica:**")
        
        diferenca_dia_mes = abs(estimativa_anual_dia - estimativa_anual_mes)
        percentual_diferenca_dia_mes = (diferenca_dia_mes / estimativa_anual_mes) * 100
        
        diferenca_semana_mes = abs(estimativa_anual_semana - estimativa_anual_mes)
        percentual_diferenca_semana_mes = (diferenca_semana_mes / estimativa_anual_mes) * 100
        
        st.write(f"A diferença entre a estimativa anual baseada na média diária e a volumetria mensal é de {diferenca_dia_mes:.0f} pacientes, "
                 f"o que representa uma variação de {percentual_diferenca_dia_mes:.2f}% em relação ao total anual.")
        
        st.write(f"A diferença entre a estimativa anual baseada na média semanal e a volumetria mensal é de {diferenca_semana_mes:.0f} pacientes, "
                 f"com uma variação de {percentual_diferenca_semana_mes:.2f}% em relação ao total anual.")
        
        if percentual_diferenca_dia_mes > 5 or percentual_diferenca_semana_mes > 5:
            st.write("As diferenças significativas entre os métodos sugerem uma variação considerável nas médias diárias e semanais. "
                     "Isso pode indicar inconsistências nos fluxos diários ou semanais de pacientes ou sazonalidade, impactando as médias calculadas.")
        else:
            st.write("As diferenças entre os métodos são mínimas, sugerindo que os fluxos de pacientes estão relativamente consistentes ao longo do tempo.")
        

# =====================================
# Parte 7: Aba "Triagem"
# =====================================
    
    with tab2:
        st.markdown("""
        Nesta seção, você vai analisar o processo de triagem dos pacientes, incluindo a distribuição por urgência, a disponibilidade média de enfermeiros por horário, o número de salas de triagem e o tempo médio de atendimento.
        """)

        # Métricas Gerais - Triagem
        st.markdown("#### 📊 Métricas Gerais")

        num_salas = df_triagem_salas[COLUNAS["TRIAGEM_SALAS"]["NUM_SALAS"]].iloc[0]
        tempo_medio_atendimento = df_triagem_tempo[COLUNAS["TRIAGEM_TEMPO"]["TEMPO_MEDIO_ATENDIMENTO"]].iloc[0]

        # Cálculo do Percentual de Risco Maior (%)
        triagens_emergencia = df_triagem_urgencia.loc[df_triagem_urgencia[COLUNAS["TRIAGEM_URGENCIA"]["URGENCIA"]] == 'EMERGÊNCIA', COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"]].values[0]
        triagens_muito_urgente = df_triagem_urgencia.loc[df_triagem_urgencia[COLUNAS["TRIAGEM_URGENCIA"]["URGENCIA"]] == 'MUITO URGENTE', COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"]].values[0]
        triagens_urgente = df_triagem_urgencia.loc[df_triagem_urgencia[COLUNAS["TRIAGEM_URGENCIA"]["URGENCIA"]] == 'URGENTE', COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"]].values[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Número de Salas de Triagem", f"{num_salas}")

        with col2:
            st.metric("Tempo Médio de Atendimento Triagem (min)", f"{tempo_medio_atendimento}")

        # Total de pacientes
        total_pacientes_triagem = df_triagem_urgencia[COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"]].sum()

        # Aplicando a fórmula (C2 + C3) + (C4 / 2)
        percentual_risco_maior = ((triagens_emergencia + triagens_muito_urgente) + (triagens_urgente / 2)) / total_pacientes_triagem * 100

        with col3:
            st.metric("Percentual de Risco Maior (%)", f"{percentual_risco_maior:.2f}%")

        st.markdown("---")

#==================================================================
# Análises Gráficas
#==================================================================
        
        st.markdown("#### 📈 Análises Gráficas")

        # Primeiro Gráfico: Pareto - Distribuição de Triagens por Urgência
        st.markdown("###### 1️⃣ Distribuição de Triagens por Urgência (Gráfico de Pareto)")
        with st.container():
            # Calculando o total de triagens
            total_triagens = df_triagem_urgencia[COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"]].sum()
        
            # Ordenando os dados de triagem por urgência
            df_triagem_urgencia_sorted = df_triagem_urgencia.sort_values(COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"], ascending=False)
        
            # Calculando o percentual e o percentual acumulado
            df_triagem_urgencia_sorted['percentual (%)'] = (df_triagem_urgencia_sorted[COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"]] / total_triagens) * 100
            df_triagem_urgencia_sorted["percentual_acumulado"] = df_triagem_urgencia_sorted['percentual (%)'].cumsum()
        
            # Criando o gráfico de barras
            fig_pareto_triagem = go.Figure()
        
            # Gráfico de barras (Quantidade de Pacientes)
            fig_pareto_triagem.add_trace(go.Bar(
                x=df_triagem_urgencia_sorted[COLUNAS["TRIAGEM_URGENCIA"]["URGENCIA"]],
                y=df_triagem_urgencia_sorted[COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"]],
                name="Quantidade de Pacientes",
                marker_color='blue',
                yaxis="y1",
                text=df_triagem_urgencia_sorted[COLUNAS["TRIAGEM_URGENCIA"]["QUANTIDADE_PACIENTES"]],
                textposition='inside'
            ))
        
            # Curva de porcentagem acumulada (linha)
            fig_pareto_triagem.add_trace(go.Scatter(
                x=df_triagem_urgencia_sorted[COLUNAS["TRIAGEM_URGENCIA"]["URGENCIA"]],
                y=df_triagem_urgencia_sorted["percentual_acumulado"],
                name="Porcentagem Acumulada",
                mode="lines+markers+text",
                line=dict(color="orange"),
                yaxis="y2",
                text=[f"{x:.1f}%" for x in df_triagem_urgencia_sorted["percentual_acumulado"]],
                textposition="top center"
            ))
        
            # Atualizando o layout
            fig_pareto_triagem.update_layout(
                title="Gráfico de Pareto: Distribuição de Triagens por Urgência",
                template="plotly_white",
                yaxis=dict(
                    title="Quantidade de Pacientes",
                    showgrid=False,
                ),
                yaxis2=dict(
                    title="Porcentagem Acumulada",
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                    showgrid=False,
                    ticksuffix="%"
                ),
                xaxis=dict(
                    title="Urgência",
                    categoryorder='total descending'
                ),
                legend=dict(
                    x=1.05, y=1.0,
                    xanchor='left',
                    yanchor='top'
                )
            )
        
            # Exibindo o gráfico no Streamlit
            st.plotly_chart(fig_pareto_triagem, use_container_width=True)
        
        # Observação analítica
        st.markdown("**Observação analítica:**")
        urgencia_dominante = df_triagem_urgencia_sorted[COLUNAS["TRIAGEM_URGENCIA"]["URGENCIA"]].iloc[0]
        percentual_dominante = df_triagem_urgencia_sorted['percentual (%)'].iloc[0]
        st.write(f"A categoria de urgência '{urgencia_dominante}' é a mais frequente, representando {percentual_dominante:.2f}% das triagens.")
        st.write("Isso pode indicar uma tendência na gravidade dos casos atendidos e ajudar na alocação de recursos.")
        
        st.markdown("###### 2️⃣ Média de Enfermeiros por Hora de Triagem")
        
        # Gráfico de barras
        with st.container():
            fig_enfermeiros = px.bar(
                df_triagem_enfermeiros,
                x='hora',
                y="quantidade_media_enfermeiros (arredondado)",
                color="quantidade_media_enfermeiros (arredondado)",
                color_continuous_scale='Blues',
                text="quantidade_media_enfermeiros (arredondado)"
            )
            fig_enfermeiros.update_traces(textposition='inside')
            st.plotly_chart(fig_enfermeiros, use_container_width=True)
        
        # Observação analítica
        st.markdown("**Observação analítica:**")
        pico_hora_enfermeiros = df_triagem_enfermeiros.loc[df_triagem_enfermeiros["quantidade_media_enfermeiros (arredondado)"].idxmax(), 'hora']
        st.write(f"A hora com maior média de enfermeiros alocados é **{int(pico_hora_enfermeiros)}h**.")
        st.write("Isso reflete a estratégia de alocar mais profissionais nos horários de maior demanda.")
        
        st.markdown("---")
        
        st.markdown("###### 3️⃣ Distribuição de Enfermeiros por Período e Hora de triagem")
        with st.container():
            fig_sunburst_enfermeiros = px.sunburst(
                df_triagem_enfermeiros,
                path=['Período', 'hora'],
                values="quantidade_media_enfermeiros (arredondado)",
                color="quantidade_media_enfermeiros (arredondado)",
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_sunburst_enfermeiros, use_container_width=True)
        
        # Observação analítica específica do gráfico Sunburst
        st.markdown("**Observação analítica do período:**")
        pico_periodo_enfermeiros = df_triagem_enfermeiros.groupby('Período')["quantidade_media_enfermeiros (arredondado)"].sum().idxmax()
        st.write(f"O período com maior alocação de enfermeiros é **{pico_periodo_enfermeiros}**.")
        st.write("Isso indica uma resposta adequada às necessidades de atendimento nesse período.")
        
        st.markdown("---")

        # Ordenar o DataFrame em ordem decrescente
        df_exames_sadt_sorted = df_exames_sadt.sort_values(by=COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"], ascending=False)

        st.markdown("###### Análise Complementar: Distribuição de Enfermeiros por Período")

        with st.container():
            fig_box_enfermeiros = px.box(
                df_triagem_enfermeiros,
                x='Período',
                y="quantidade_media_enfermeiros (arredondado)",
                color='Período',
                title="Box-Plot da Quantidade de Enfermeiros por Período",
                points="all",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_box_enfermeiros, use_container_width=True)

        # Observação analítica
        st.markdown("**Observação analítica:**")
        st.write("O box-plot permite visualizar a distribuição da quantidade de enfermeiros em cada período, identificando variações e possíveis necessidades de ajuste na alocação de pessoal.")
        st.write("Isso é crucial para garantir a eficiência no atendimento e a satisfação dos pacientes.")

        st.markdown("---")
        
        with st.container():
            # Análise Comparativa
                st.markdown("#### 8️⃣ Análise Comparativa")
                estimativa_anual_dia = total_pacientes_dia * 365
                media_diaria_semana = total_pacientes_semana / 7
                estimativa_anual_semana = media_diaria_semana * 365
                estimativa_anual_mes = total_pacientes_ano
                
                # Gráfico de barras comparando estimativas
                fig_comparacao = px.bar(
                    x=['Estimativa Anual (Média/Dia)', 'Estimativa Anual (Média/Semana)', 'Estimativa Anual (Volumetria/Mês)'],
                    y=[estimativa_anual_dia, estimativa_anual_semana, estimativa_anual_mes],
                    title="Estimativa Anual: Comparativo entre Diferentes Métodos",
                    labels={"x": "Método", "y": "Estimativa Anual"}
                )
                st.plotly_chart(fig_comparacao, use_container_width=True)
                
                st.markdown("**Observação analítica:**")
                diferenca_dia_mes = abs(estimativa_anual_dia - estimativa_anual_mes)
                percentual_diferenca_dia_mes = (diferenca_dia_mes / estimativa_anual_mes) * 100
                diferenca_semana_mes = abs(estimativa_anual_semana - estimativa_anual_mes)
                percentual_diferenca_semana_mes = (diferenca_semana_mes / estimativa_anual_mes) * 100
                
                # Explicação mais detalhada das diferenças
                st.write(f"A diferença entre a estimativa baseada na média diária e a volumetria mensal é de {diferenca_dia_mes:.0f} pacientes, representando uma variação de {percentual_diferenca_dia_mes:.2f}% em relação ao total anual estimado com base nos dados mensais.")
                
                st.write(f"De forma semelhante, a diferença entre a estimativa baseada na média semanal e a volumetria mensal é de {diferenca_semana_mes:.0f} pacientes, uma variação de {percentual_diferenca_semana_mes:.2f}%.")
                
                # Interpretação analítica
                st.write("""
                Essas diferenças refletem a variabilidade dos métodos de projeção:
                - A estimativa diária pode amplificar ou suavizar variações sazonais ou flutuações semanais.
                - A estimativa semanal tenta suavizar as oscilações diárias, mas ainda pode ser influenciada por eventos semanais atípicos.
                - Já a estimativa mensal reflete o comportamento de longo prazo, sendo menos sensível às variações curtas.
                
                Essa análise é crucial para entender como diferentes intervalos de agregação afetam a previsão de demanda hospitalar e podem orientar melhor a alocação de recursos.
                """)

# =====================================
# Parte 7: Aba "Consulta"
# =====================================
    
    with tab3:
        st.markdown("""
        Nesta seção, você poderá analisar os dados referentes às consultas médicas, incluindo o tempo médio por etapa da consulta, a distribuição de médicos por horário, dia da semana e especialidade. Essas informações são essenciais para otimizar o fluxo de consultas e a alocação de recursos humanos.
        """)
    
        # ==========================
        # Métricas Gerais
        # ==========================
        st.markdown("#### 📊 Métricas Gerais")
    
        # Check if data is available
        if df_consulta_tempo.empty or df_media_medicos_consulta.empty or df_dados_semanais_medicos.empty or df_media_medicos_especialidade.empty:
            st.warning("Dados insuficientes para gerar as métricas. Verifique se todas as abas necessárias estão preenchidas corretamente no arquivo Excel.")
        else:
            # Tempo total médio da consulta
            tempo_total_consulta = df_consulta_tempo[COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]].sum()
            tempo_medio_etapa = df_consulta_tempo[COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]].mean()
    
            # Média de médicos por período
            media_medicos_manha_tarde = df_dados_semanais_medicos[COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_MANHA_TARDE"]].mean()
            media_medicos_noite_madrugada = df_dados_semanais_medicos[COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_NOITE_MADRUGADA"]].mean()
    
            # Média de pacientes por médico
            total_pacientes_consulta = df_media_medicos_consulta[COLUNAS["MEDIA_MEDICOS_CONSULTA"]["QUANTIDADE_MEDIA_MEDICOS"]].sum()
            total_medicos = df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["QUANTIDADE_MEDIA_MEDICOS"]].sum()
            media_pacientes_por_medico = total_pacientes_consulta / total_medicos if total_medicos > 0 else 0
    
            # Exibição das métricas
            col1, col2, col3 = st.columns(3,gap="small")
            with col1:
                st.metric("Tempo Total Médio da Consulta (min)", f"{tempo_total_consulta:.2f}")
            with col2:
                st.metric("Tempo Médio por Etapa (min)", f"{tempo_medio_etapa:.2f}")
            with col3:
                st.metric("Média de Pacientes por Médico", f"{media_pacientes_por_medico:.2f}")
    
            col4, col5, _ = st.columns(3,gap="small")
            with col4:
                st.metric("Média de Médicos Manhã/Tarde", f"{media_medicos_manha_tarde:.2f}")
            with col5:
                st.metric("Média de Médicos Noite/Madrugada", f"{media_medicos_noite_madrugada:.2f}")
    
            st.markdown("---")

            # ==========================
            # Visualizações Gráficas
            # ==========================
            st.markdown("#### 📈 Visualizações Gráficas")
            
            # Gráfico 1: Tempo Médio por Etapa da Consulta
            st.markdown("###### 1️⃣ Tempo Médio por Etapa da Consulta")
            with st.container():
                if df_consulta_tempo.empty:
                    st.warning("Dados insuficientes para gerar o gráfico.")
                else:
                    fig_etapas = px.bar(
                        df_consulta_tempo,
                        x=COLUNAS["CONSULTA_TEMPO"]["ETAPA"],
                        y=COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"],
                        color=COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"],
                        color_continuous_scale='Blues',
                        text=COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]  # Adiciona o texto para os rótulos
                    )
                    fig_etapas.update_traces(texttemplate='%{text:.1f}', textposition='inside')  # Formata e posiciona os rótulos
                    st.plotly_chart(fig_etapas, use_container_width=True)
            
                    # Observação analítica
                    st.markdown("**Observação analítica:**")
                    if df_consulta_tempo[COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]].isnull().all():
                        st.write("Os dados de tempo médio por etapa estão indisponíveis.")
                    else:
                        etapa_mais_demorada_idx = df_consulta_tempo[COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]].idxmax()
                        etapa_mais_demorada = df_consulta_tempo.loc[etapa_mais_demorada_idx, COLUNAS["CONSULTA_TEMPO"]["ETAPA"]]
                        tempo_mais_demorado = df_consulta_tempo[COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]].max()
                        st.write(f"A etapa **'{etapa_mais_demorada}'** é a mais demorada, levando em média **{tempo_mais_demorado:.2f} minutos**. Isso indica que há potencial para otimização nesta fase do processo de consulta.")
            
            st.markdown("---")
            
            # Gráfico 2: Distribuição de Médicos por Horário
            st.markdown("###### 2️⃣ Distribuição de Médicos por Horário")
            with st.container():
                if df_media_medicos_consulta.empty:
                    st.warning("Dados insuficientes para gerar o gráfico.")
                else:
                    fig_medicos_horario = px.bar(
                        df_media_medicos_consulta,
                        x='hora',
                        y=COLUNAS["MEDIA_MEDICOS_CONSULTA"]["QUANTIDADE_MEDIA_MEDICOS"],
                        color=COLUNAS["MEDIA_MEDICOS_CONSULTA"]["QUANTIDADE_MEDIA_MEDICOS"],
                        color_continuous_scale='Blues',
                        text=COLUNAS["MEDIA_MEDICOS_CONSULTA"]["QUANTIDADE_MEDIA_MEDICOS"]  # Adiciona o texto para os rótulos
                    )
                    fig_medicos_horario.update_traces(texttemplate='%{text:.1f}', textposition='inside')  # Formata e posiciona os rótulos
                    st.plotly_chart(fig_medicos_horario, use_container_width=True)
            
                    # Observação analítica
                    st.markdown("**Observação analítica:**")
                    pico_horario_medicos = df_media_medicos_consulta.loc[
                        df_media_medicos_consulta[COLUNAS["MEDIA_MEDICOS_CONSULTA"]["QUANTIDADE_MEDIA_MEDICOS"]].idxmax(), 'hora']
                    st.write(f"O horário com maior número médio de médicos é **{int(pico_horario_medicos)}h**. Isso sugere uma concentração de demanda neste período.")
            
            st.markdown("---")
            
            # Gráfico 3: Médicos por Dia da Semana e Período
            st.markdown("###### 3️⃣ Médicos por Dia da Semana e Período")
            with st.container():
                if df_dados_semanais_medicos.empty:
                    st.warning("Dados insuficientes para gerar o gráfico.")
                else:
                    df_dados_semanais_medicos_melt = df_dados_semanais_medicos.melt(
                        id_vars=[COLUNAS["DADOS_SEMANAIS_MEDICOS"]["DIA"]],
                        value_vars=[
                            COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_MANHA_TARDE"],
                            COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_NOITE_MADRUGADA"]
                        ],
                        var_name='Período',
                        value_name='Quantidade de Médicos'
                    )
            
                    # Rename periods for better readability
                    periodo_mapping = {
                        COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_MANHA_TARDE"]: "Manhã/Tarde",
                        COLUNAS["DADOS_SEMANAIS_MEDICOS"]["MEDICOS_NOITE_MADRUGADA"]: "Noite/Madrugada"
                    }
                    df_dados_semanais_medicos_melt['Período'] = df_dados_semanais_medicos_melt['Período'].map(periodo_mapping)
            
                    fig_medicos_semana = px.bar(
                        df_dados_semanais_medicos_melt,
                        x=COLUNAS["DADOS_SEMANAIS_MEDICOS"]["DIA"],
                        y='Quantidade de Médicos',
                        color='Período',
                        barmode='group',
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        text='Quantidade de Médicos'  # Adiciona o texto para os rótulos
                    )
                    fig_medicos_semana.update_traces(texttemplate='%{text:.1f}', textposition='inside')  # Formata e posiciona os rótulos
                    st.plotly_chart(fig_medicos_semana, use_container_width=True)
                
                        
            st.markdown("---")
    
            # Gráfico 4: Distribuição de Médicos por Especialidade
            st.markdown("###### 4️⃣ Distribuição de Médicos por Especialidade")
            with st.container():
                if df_media_medicos_especialidade.empty:
                    st.warning("Dados insuficientes para gerar o gráfico.")
                else:
                    fig_medicos_especialidade = px.pie(
                        df_media_medicos_especialidade,
                        names=COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["ESPECIALIDADE"],
                        values=COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["QUANTIDADE_MEDIA_MEDICOS"],
                        color_discrete_sequence=px.colors.sequential.Blues
                    )
                    st.plotly_chart(fig_medicos_especialidade, use_container_width=True)
    
                    # Observação analítica
                    st.markdown("**Observação analítica:**")
                    especialidade_dominante_idx = df_media_medicos_especialidade[
                        COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["QUANTIDADE_MEDIA_MEDICOS"]].idxmax()
                    especialidade_dominante = df_media_medicos_especialidade.loc[
                        especialidade_dominante_idx, COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["ESPECIALIDADE"]]
                    st.write(f"A especialidade com maior número médio de médicos é **{especialidade_dominante}**. Isso pode indicar uma maior demanda por essa especialidade ou uma estratégia de alocação focada.")
    
            st.markdown("---")
    
            # Gráfico 5: Análise do Tempo Médio de Espera por Etapa
            st.markdown("###### 5️⃣ Análise do Tempo Médio de Espera por Etapa")
            with st.container():
                if df_consulta_tempo.empty:
                    st.warning("Dados insuficientes para gerar o gráfico.")
                else:
                    fig_tempo_etapa = px.funnel(
                        df_consulta_tempo,
                        x=COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"],
                        y=COLUNAS["CONSULTA_TEMPO"]["ETAPA"],
                    )
                    st.plotly_chart(fig_tempo_etapa, use_container_width=True)
    
                    # Observação analítica
                    st.markdown("**Observação analítica:**")
                    st.write("O gráfico de funil destaca as etapas que mais contribuem para o tempo total da consulta. Foco nas etapas com maior duração pode levar a melhorias significativas na eficiência do atendimento.")
    
            st.markdown("---")

# =====================================
# Parte 8: "SADT" 
# =====================================
    
    with tab4:
        st.markdown("""
        Nesta seção, você vai analisar os dados relacionados aos exames realizados (SADT), incluindo o tempo médio de realização de cada tipo de exame e a quantidade de pacientes que realizaram esses exames.
        """)
    
        # Verificando se dos dados estão disponíveis
        if df_exames_sadt_sorted.empty:
            st.warning("Dados insuficientes para gerar as análises. Verifique se os dados estão preenchidos corretamente no arquivo Excel.")
        else:
            # Primeiro Gráfico: Tempo Médio de Exames por Tipo (Gráfico de Barras Horizontais)
            st.markdown("###### 1️⃣ Tempo Médio de Exames por Tipo")
            with st.container():
                # Ordenar o DataFrame pelo tempo médio de exame em ordem decrescente
                df_exames_sadt_sorted = df_exames_sadt_sorted.sort_values(by=COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"], ascending=True)
                
                fig_tempo_barras = px.bar(
                    df_exames_sadt_sorted,
                    x=COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"],
                    y=COLUNAS["EXAMES_SADT"]["TIPO_EXAME"],
                    orientation='h',
                    color=COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"],
                    color_continuous_scale='Blues',
                    template="plotly_white",
                    text=COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"]  # Adiciona o texto nas barras
                )
                
                # Configurar o layout para melhorar a aparência
                fig_tempo_barras.update_traces(texttemplate='%{text:.1f}', textposition='inside')
                fig_tempo_barras.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                
                # Ajustar as margens para garantir que os rótulos externos sejam visíveis
                fig_tempo_barras.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                
                st.plotly_chart(fig_tempo_barras, use_container_width=True)
            
            # Observação analítica para o gráfico de tempo médio
            st.markdown("**Observação analítica - Tempo Médio de Exames:**")
            exame_mais_demorado = df_exames_sadt_sorted.iloc[0][COLUNAS["EXAMES_SADT"]["TIPO_EXAME"]]
            tempo_mais_demorado = df_exames_sadt_sorted.iloc[0][COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"]]
            exame_mais_rapido = df_exames_sadt_sorted.iloc[-1][COLUNAS["EXAMES_SADT"]["TIPO_EXAME"]]
            tempo_mais_rapido = df_exames_sadt_sorted.iloc[-1][COLUNAS["EXAMES_SADT"]["TEMPO_MEDIO_EXAME"]]
            
            st.write(f"O exame mais demorado é '{exame_mais_demorado}' com um tempo médio de {tempo_mais_demorado:.1f} minutos.")
            st.write(f"O exame mais rápido é '{exame_mais_rapido}' com um tempo médio de {tempo_mais_rapido:.1f} minutos.")
        
            st.markdown("---")
    
            # Segundo Gráfico: Pareto - Quantidade de Pacientes por Tipo de Exame
            
            st.markdown("###### 2️⃣ Quantidade de Pacientes por Tipo de Exame (Gráfico de Pareto)")
            
            with st.container():
                # Ordenar o DataFrame pela quantidade de pacientes em ordem decrescente
                df_exames_sadt_sorted = df_exames_sadt.sort_values(by=COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"], ascending=False)
                
                df_exames_sadt_sorted["Porcentagem_Acumulada"] = df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]].cumsum() / df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]].sum() * 100
            
                # Creating the Pareto chart
                fig_pareto = go.Figure()
            
                # Bar chart
                fig_pareto.add_trace(go.Bar(
                    x=df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["TIPO_EXAME"]],
                    y=df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]],
                    name="Quantidade de Pacientes",
                    marker_color='blue',
                    yaxis="y1",
                    text=df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]],
                    textposition='inside'
                ))
            
                # Cumulative percentage curve (line)
                fig_pareto.add_trace(go.Scatter(
                    x=df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["TIPO_EXAME"]],
                    y=df_exames_sadt_sorted["Porcentagem_Acumulada"],
                    name="Porcentagem Acumulada",
                    mode="lines+markers+text",
                    line=dict(color="orange"),
                    yaxis="y2",
                    text=[f"{x:.1f}%" for x in df_exames_sadt_sorted["Porcentagem_Acumulada"]],
                    textposition="top center"
                ))
            
                # Updating the layout
                fig_pareto.update_layout(
                    template="plotly_white",
                    yaxis=dict(
                        title="Quantidade de Pacientes",
                        showgrid=False,
                    ),
                    yaxis2=dict(
                        title="Porcentagem Acumulada",
                        overlaying="y",
                        side="right",
                        range=[0, 100],
                        showgrid=False,
                        ticksuffix="%"
                    ),
                    xaxis=dict(
                        title="Tipo de Exame",
                        tickangle=45
                    ),
                    legend=dict(
                        x=1.05, y=1.0
                    )
                )
            
                # Displaying the chart in Streamlit
                st.plotly_chart(fig_pareto, use_container_width=True)
                        # Observação analítica
            st.markdown("**Observação analítica:**")
            exame_mais_realizado = df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["TIPO_EXAME"]].iloc[0]
            quantidade_mais_realizada = df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["QUANTIDADE_PACIENTE_EXAME_MES"]].iloc[0]
            st.write(f"O exame mais realizado é **{exame_mais_realizado}**, com **{quantidade_mais_realizada} pacientes**.")
            st.write("Esse exame representa a maior demanda e pode necessitar de mais recursos ou atenção especial.")
    
            st.markdown("---")
    
            st.markdown("###### 3️⃣ Percentual de Pacientes por Tipo de Exame")
            with st.container():
                fig_donut = px.pie(
                    df_exames_sadt_sorted,
                    names=COLUNAS["EXAMES_SADT"]["TIPO_EXAME"],
                    values='percentual_pacientes',
                    title="Percentual de Pacientes por Tipo de Exame",
                    hole=0.4,  # Donut 
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                st.plotly_chart(fig_donut, use_container_width=True)
    
                # Observação analítica
                st.markdown("**Observação analítica:**")
    
                # Calculating some statistics
                total_exames = len(df_exames_sadt_sorted)
                exame_mais_frequente = df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["TIPO_EXAME"]].iloc[0]
                percentual_mais_frequente = df_exames_sadt_sorted['percentual_pacientes'].iloc[0]
                exame_menos_frequente = df_exames_sadt_sorted[COLUNAS["EXAMES_SADT"]["TIPO_EXAME"]].iloc[-1]
                percentual_menos_frequente = df_exames_sadt_sorted['percentual_pacientes'].iloc[-1]
                percentual_top_3 = df_exames_sadt_sorted['percentual_pacientes'].iloc[:3].sum().round(2)
    
                # Displaying the analytical observations
                st.write(f"1. **Exame Mais Frequente**: O exame **'{exame_mais_frequente}'** é o mais realizado, representando **{percentual_mais_frequente:.2f}%** do total.")
                st.write(f"2. **Exame Menos Frequente**: O exame **'{exame_menos_frequente}'** é o menos realizado, com apenas **{percentual_menos_frequente:.2f}%** do total.")
                st.write(f"3. **Concentração**: Os três exames mais frequentes juntos representam **{percentual_top_3:.2f}%** do total. " +
                         ("Isso sugere uma alta concentração em poucos tipos de exames." if percentual_top_3 > 50 else "Isso indica uma distribuição relativamente equilibrada entre os tipos de exames."))


# =====================================
# Parte 9: "Passagem & Internação" com Todas as Visualizações
# =====================================

    with tab5:
        st.markdown("""
        ## 🏥 Passagem & Internação
        Nesta seção, você poderá analisar os dados referentes à Passagem de Setores e Internação, incluindo métricas de utilização de leitos, tempo médio de permanência e taxa de ocupação.
        """)
    
        # ==========================
        # Exibição dos Dados e Métricas
        # ==========================
        st.markdown("### 🛏️ Passagem de Setores")
    
        # Selecionar colunas desejadas e ajustar 'taxa_ocupacao'
        cols_passagem = [
            COLUNAS["PASSAGEM_SETORES"]["SETORES"],
            COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"],
            COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"],
            COLUNAS["PASSAGEM_SETORES"]["TAXA_OCUPACAO"]
        ]
        df_passagem_setores_selecionado = df_passagem_setores[cols_passagem].copy()
    
        # Garantir que 'taxa_ocupacao' é numérica e converter para porcentagem
        taxa_ocupacao_col = COLUNAS["PASSAGEM_SETORES"]["TAXA_OCUPACAO"]
        df_passagem_setores_selecionado[taxa_ocupacao_col] = pd.to_numeric(
            df_passagem_setores_selecionado[taxa_ocupacao_col], errors='coerce') * 100
    
        # Formatar 'taxa_ocupacao' como porcentagem com uma casa decimal
        df_passagem_setores_selecionado[taxa_ocupacao_col] = df_passagem_setores_selecionado[taxa_ocupacao_col].map("{:.1f}%".format)
    
        st.dataframe(df_passagem_setores_selecionado)
    
        # Gráfico: Quantidade de Leitos por Setor
        st.markdown("#### 1️⃣ Quantidade de Leitos por Setor")
        df_leitos_sorted = df_passagem_setores_selecionado.sort_values(
            by=COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"], ascending=False)
    
        fig_leitos_setor = px.bar(
            df_leitos_sorted,
            x=COLUNAS["PASSAGEM_SETORES"]["SETORES"],
            y=COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"],
            labels={
                COLUNAS["PASSAGEM_SETORES"]["SETORES"]: "Setores",
                COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]: "Quantidade de Leitos"
            },
            template='plotly_white',
            color=COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"],
            color_continuous_scale='Blues',
            text=COLUNAS["PASSAGEM_SETORES"]["QUANTIDADE_LEITOS"]
        )
        fig_leitos_setor.update_traces(textposition='inside', texttemplate='%{text:.0f}')
        st.plotly_chart(fig_leitos_setor, use_container_width=True)
    
        # Gráfico: Tempo Médio de Permanência por Setor
        st.markdown("#### 2️⃣ Tempo Médio de Permanência por Setor")
        df_tempo_perm_sorted = df_passagem_setores_selecionado.sort_values(
            by=COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"], ascending=False)
    
        fig_tempo_permanencia = px.bar(
            df_tempo_perm_sorted,
            x=COLUNAS["PASSAGEM_SETORES"]["SETORES"],
            y=COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"],
            labels={
                COLUNAS["PASSAGEM_SETORES"]["SETORES"]: "Setores",
                COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]: "Tempo Médio de Permanência (dias)"
            },
            template='plotly_white',
            color=COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"],
            color_continuous_scale='Blues',
            text=COLUNAS["PASSAGEM_SETORES"]["TEMPO_MEDIO_PERMANENCIA_DIAS"]
        )
        fig_tempo_permanencia.update_traces(textposition='inside', texttemplate='%{text:.1f}')
        st.plotly_chart(fig_tempo_permanencia, use_container_width=True)
    
        # Gráfico: Taxa de Ocupação por Setor
        st.markdown("#### 3️⃣ Taxa de Ocupação por Setor")
        df_taxa_ocupacao = df_passagem_setores_selecionado.copy()
        df_taxa_ocupacao[taxa_ocupacao_col] = df_taxa_ocupacao[taxa_ocupacao_col].str.rstrip('%').astype('float')
        df_taxa_ocupacao_sorted = df_taxa_ocupacao.sort_values(by=taxa_ocupacao_col, ascending=False)
    
        fig_taxa_ocupacao = px.bar(
            df_taxa_ocupacao_sorted,
            x=COLUNAS["PASSAGEM_SETORES"]["SETORES"],
            y=taxa_ocupacao_col,
            labels={
                COLUNAS["PASSAGEM_SETORES"]["SETORES"]: "Setores",
                taxa_ocupacao_col: "Taxa de Ocupação (%)"
            },
            template='plotly_white',
            color=taxa_ocupacao_col,
            color_continuous_scale='Blues',
            text=taxa_ocupacao_col
        )
        fig_taxa_ocupacao.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        st.plotly_chart(fig_taxa_ocupacao, use_container_width=True)
    
        st.markdown("---")
    
        # ==========================
        # Tempo de Permanência em Leitos
        # ==========================
        st.markdown("### ⏱️ Tempo de Permanência em Leitos")
    
        cols_tempo_perm = [
            COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TIPO_DE_LEITO"],
            COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["CLASSIFICACAO_SALAS_CIRURGICAS"],
            COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["QUANTIDADE_DE_LEITO"],
            COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"]
        ]
        df_tempo_permanencia_leitos_selecionado = df_tempo_permanencia_leitos[cols_tempo_perm]
    
        st.dataframe(df_tempo_permanencia_leitos_selecionado)
    
        # Gráfico: Tempo Médio de Permanência por Tipo de Leito
        st.markdown("#### 4️⃣ Tempo Médio de Permanência por Tipo de Leito")
        df_tempo_leitos_sorted = df_tempo_permanencia_leitos_selecionado.sort_values(
            by=COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"], ascending=False
        )
    
        unique_classes = df_tempo_leitos_sorted[COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["CLASSIFICACAO_SALAS_CIRURGICAS"]].unique()
        num_classes = len(unique_classes)
        color_map = px.colors.sequential.Blues
        color_sequence = color_map[-num_classes:] if num_classes <= len(color_map) else (color_map * (num_classes // len(color_map) + 1))[-num_classes:]
    
        fig_tempo_leitos = px.bar(
            df_tempo_leitos_sorted,
            x=COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TIPO_DE_LEITO"],
            y=COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"],
            color=COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["CLASSIFICACAO_SALAS_CIRURGICAS"],
            barmode='group',
            labels={
                COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TIPO_DE_LEITO"]: "Tipo de Leito",
                COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"]: "Tempo Médio de Permanência (min)",
                COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["CLASSIFICACAO_SALAS_CIRURGICAS"]: "Classificação"
            },
            template='plotly_white',
            color_discrete_sequence=color_sequence,
            text=COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"]
        )
        fig_tempo_leitos.update_traces(textposition='inside', texttemplate='%{text:.0f}')
        fig_tempo_leitos.update_layout(
            xaxis_tickangle=-45,
            xaxis=dict(title='Tipo de Leito'),
            yaxis=dict(title='Tempo Médio de Permanência (min)'),
            legend_title_text='Classificação'
        )
        st.plotly_chart(fig_tempo_leitos, use_container_width=True)
    
        st.markdown("---")
    
        
        # ==========================
        # Internação - Demanda
        # ==========================
        
        st.markdown("### 📥 Internação - Demanda")
    
        cols_demanda = [
            COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"],
            COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]
        ]
        df_internacao_demanda_selecionado = df_internacao_demanda[cols_demanda]
    
        st.dataframe(df_internacao_demanda_selecionado)
    
        # Gráfico: Média de Solicitações de Leitos por Dia
        st.markdown("#### 5️⃣ Média de Solicitações de Leitos por Dia")
        df_solicitacoes_sorted = df_internacao_demanda_selecionado.sort_values(
            by=COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"], ascending=False)
    
        fig_solicitacoes_leitos = px.bar(
            df_solicitacoes_sorted,
            x=COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"],
            y=COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"],
            labels={
                COLUNAS["INTERNACAO_DEMANDA"]["SOLICITACOES_LEITO"]: "Solicitações de Leito",
                COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]: "Média de Solicitações por Dia"
            },
            template='plotly_white',
            color=COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"],
            color_continuous_scale='Blues',
            text=COLUNAS["INTERNACAO_DEMANDA"]["MEDIA_SOLICITACOES_DIA"]
        )
        fig_solicitacoes_leitos.update_traces(texttemplate='%{text:.1f}', textposition='inside')
        fig_solicitacoes_leitos.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_solicitacoes_leitos, use_container_width=True)
    
        st.markdown("---")
    
        # ==========================
        # Internação - Saída
        # ==========================
        st.markdown("### 📤 Internação - Saída")
    
        cols_saida = [
            COLUNAS["INTERNACAO_SAIDA"]["SAIDA_INTERNACAO"],
            COLUNAS["INTERNACAO_SAIDA"]["MEDIA_SAIDA_DIA"]
        ]
        df_internacao_saida_selecionado = df_internacao_saida[cols_saida]
    
        st.dataframe(df_internacao_saida_selecionado)
    
        # Gráfico: Média de Saídas por Dia
        st.markdown("#### 6️⃣ Média de Saídas por Dia")
        fig_saidas_internacao = px.pie(
            df_internacao_saida_selecionado,
            names=COLUNAS["INTERNACAO_SAIDA"]["SAIDA_INTERNACAO"],
            values=COLUNAS["INTERNACAO_SAIDA"]["MEDIA_SAIDA_DIA"],
            template='plotly_white',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_saidas_internacao.update_traces(textposition='inside', textinfo='percent+label')
        fig_saidas_internacao.update_layout(
            legend_title="Saída de Internação",
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            margin=dict(l=40, r=200, t=80, b=80)
        )
        st.plotly_chart(fig_saidas_internacao, use_container_width=True)
    
        st.markdown("---")
    
        # ==========================
        # Taxa de Internação (Métricas)
        # ==========================
        
        st.markdown("### 📊 Taxa de Internação")
    
        df_taxa_internacao_adjusted = df_taxa_internacao.copy()
        indicador_col = COLUNAS["TAXA_INTERNACAO"]["INDICADOR"]
        resultado_col = COLUNAS["TAXA_INTERNACAO"]["RESULTADO_PERCENTUAL"]
        df_taxa_internacao_adjusted[indicador_col] = df_taxa_internacao_adjusted[indicador_col].str.title()
    
        colunas_metricas = st.columns(len(df_taxa_internacao_adjusted))
        for idx, row in df_taxa_internacao_adjusted.iterrows():
            indicador = row[indicador_col]
            resultado_percentual = row[resultado_col] * 100
            with colunas_metricas[idx]:
                st.metric(indicador, f"{resultado_percentual:.1f}%")
    
        st.markdown("---")
    
        # ==========================
        # Fatores de Utilização (Métricas)
        # ==========================
        st.markdown("### 7️⃣ Fatores de Utilização")
    
        dados_fatores_utilizacao = {
            "Indicador": [
                "Fator de Utilização do Hospital – Geral",
                "Fator de Utilização dos Leitos P.A. (Enf.)",
                "Fator de Utilização dos Leitos P.A. (Uti)",
                "Fator de Utilização dos Leitos P.A. (Clínicos)",
                "Fator de Utilização dos Leitos P.A. (Cirúrgicos)"
            ],
            "Resultado": [
                fator_utilizacao_geral * 100,
                fator_utilizacao_pa_enf * 100,
                fator_utilizacao_pa_uti * 100,
                fator_utilizacao_pa_clinicos * 100,
                fator_utilizacao_pa_cirurgicos * 100
            ]
        }
        df_fatores_utilizacao = pd.DataFrame(dados_fatores_utilizacao)
    
        df_fatores_utilizacao['Resultado'] = df_fatores_utilizacao['Resultado'].map("{:.1f}%".format)
    
        st.dataframe(df_fatores_utilizacao)
    
        df_fatores_utilizacao_graph = df_fatores_utilizacao.copy()
        df_fatores_utilizacao_graph['Resultado'] = df_fatores_utilizacao_graph['Resultado'].str.rstrip('%').astype('float')
    
        df_fatores_utilizacao_sorted = df_fatores_utilizacao_graph.sort_values(by='Resultado', ascending=False)
    
        # Gráfico: Fatores de Utilização
        st.markdown("#### 8️⃣ Fatores de Utilização")
        fig_fatores_utilizacao = px.bar(
            df_fatores_utilizacao_sorted,
            x="Indicador",
            y="Resultado",
            labels={
                "Indicador": "Indicador",
                "Resultado": "Resultado (%)"
            },
            template='plotly_white',
            color='Resultado',
            color_continuous_scale='Blues',
            text='Resultado'
        )
        fig_fatores_utilizacao.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
    
        fig_fatores_utilizacao.update_layout(
            xaxis_tickangle=-45,
            margin=dict(b=150),
            height=600
        )
        st.plotly_chart(fig_fatores_utilizacao, use_container_width=True)


# =====================================
# Parte 10: "Centro Cirúrgico" (Atualizado)
# =====================================

    with tab6:
        st.markdown("""
        Nesta seção, você poderá analisar os dados referentes ao Centro Cirúrgico, incluindo a eficiência global, agendamento, desempenho, além de métricas relacionadas ao tempo médio de cirurgia e atrasos.
        """)
        
        # ==========================
        # Cálculos de Eficiência
        # ==========================
        st.markdown("#### 📊 Cálculos de Eficiência")
        
        # Quantidade de Salas Cirúrgicas (SUS e Suplementar/Particular)
        quantidade_salas_sus = df_classificacao_salas_cirurgicas.loc[
            df_classificacao_salas_cirurgicas[COLUNAS["CLASSIFICACAO_SALAS_CIRURGICAS"]["CLASSIFICACAO_SALAS_CIRURGICAS"]] == "SUS", 
            COLUNAS["CLASSIFICACAO_SALAS_CIRURGICAS"]["QUANTIDADE_SALAS_CIRURGICAS"]
        ].sum()
        
        quantidade_salas_suplementar = df_classificacao_salas_cirurgicas.loc[
            df_classificacao_salas_cirurgicas[COLUNAS["CLASSIFICACAO_SALAS_CIRURGICAS"]["CLASSIFICACAO_SALAS_CIRURGICAS"]] == "Suplementar/Particular", 
            COLUNAS["CLASSIFICACAO_SALAS_CIRURGICAS"]["QUANTIDADE_SALAS_CIRURGICAS"]
        ].sum()
    
        # Prevenção de divisão por zero
        total_salas = max((quantidade_salas_sus + quantidade_salas_suplementar), 1)
        
        # Eficiência de Agendamento (%)
        eficiencia_agendamento = df_media_horas_agendadas[COLUNAS["MEDIA_HORAS_AGENDADAS"]["HORAS_AGENDADAS"]].mean() / (total_salas * 12)
        
        # Eficiência de Desempenho (%)
        eficiencia_desempenho = df_media_horas_gastas[COLUNAS["MEDIA_HORAS_GASTAS"]["HORAS_GASTAS"]].mean() / (total_salas * 12)
        
        # Eficiência Global (%)
        eficiencia_global = eficiencia_agendamento * eficiencia_desempenho
        
        # Exibir Eficiências
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Eficiência de Agendamento (%)", f"{eficiencia_agendamento * 100:.2f}%")
        with col2:
            st.metric("Eficiência de Desempenho (%)", f"{eficiencia_desempenho * 100:.2f}%")
        with col3:
            st.metric("Eficiência Global (%)", f"{eficiencia_global * 100:.2f}%")
    
        st.markdown("---")
    
        # ==========================
        # Métricas Cirúrgicas
        # ==========================
        st.markdown("#### 🏥 Métricas Cirúrgicas")
    
        # Tempo Médio de Solicitação até Cirurgia de Urgência (min)
        tempo_medio_urgencia = df_tempo_medio_solicitacao_cirurgi[COLUNAS["TEMPO_MEDIO_SOLICITACAO_CIRURGIA"]["TEMPO_MEDIO_SOLICITACAO"]].iloc[0]
    
        # Quantidade de Cirurgias Não Programadas
        qtd_cirurgias_nao_programadas = df_qtd_cirurgias_nao_programadas[COLUNAS["QTD_CIRURGIAS_NAO_PROGRAMADAS"]["QTD_CIRURGIAS_NAO_PROGRAMADAS"]].sum()
    
        # Tempo Médio de Atraso da Primeira Cirurgia (min)
        tempo_medio_atraso_primeira = df_tempo_medio_atraso_primeira[COLUNAS["TEMPO_MEDIO_ATRASO_CIRURGIA"]["TEMPO_ATRASO"]].mean()
    
        # Exibir Métricas Cirúrgicas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tempo Mediano do Pedido até Cirurgia (min)", f"{tempo_medio_urgencia:.2f}")
        with col2:
            st.metric("Cirurgias Não Programadas/Mês", f"{qtd_cirurgias_nao_programadas:.0f}")
        with col3:
            st.metric("Tempo Médio Atraso 1ª Cirurgia (min)", f"{tempo_medio_atraso_primeira:.2f}")
    
        st.markdown("---")
    
        # ==========================
        # Visualizações Gráficas
        # ==========================
        st.markdown("#### 📈 Visualizações Gráficas")
    
        # Preparação dos Dados para os Gráficos
        # Converter colunas numéricas
        colunas_numericas = [
            COLUNAS["CIRURGIAS_MES"]["ELETIVAS_SUS"],
            COLUNAS["CIRURGIAS_MES"]["ELETIVAS_SUPLEMENTAR"],
            COLUNAS["CIRURGIAS_MES"]["URGENCIA_SUS"],
            COLUNAS["CIRURGIAS_MES"]["URGENCIA_SUPLEMENTAR"],
        ]
        
        for coluna in colunas_numericas:
            df_cirurgias_mes[coluna] = pd.to_numeric(df_cirurgias_mes[coluna], errors='coerce')
        
        # Mapear as abreviações dos meses em português para números
        month_map = {
            'JAN': 1,
            'FEV': 2,
            'MAR': 3,
            'ABR': 4,
            'MAI': 5,
            'JUN': 6,
            'JUL': 7,
            'AGO': 8,
            'SET': 9,
            'OUT': 10,
            'NOV': 11,
            'DEZ': 12
        }
        
        df_cirurgias_mes['month_num'] = df_cirurgias_mes[COLUNAS["CIRURGIAS_MES"]["MES"]].str.upper().map(month_map)
        
        # Verificar se houve algum mês não mapeado
        if df_cirurgias_mes['month_num'].isnull().any():
            st.error("Há meses não reconhecidos na coluna 'mes'. Verifique se todos os meses estão corretamente abreviados em português.")
        
        # Criar a coluna 'Data' usando ano e número do mês
        df_cirurgias_mes['Data'] = pd.to_datetime({
            'year': df_cirurgias_mes[COLUNAS["CIRURGIAS_MES"]["ANO"]],
            'month': df_cirurgias_mes['month_num'],
            'day': 1
        })
        
        # Ordenar o DataFrame por data
        df_cirurgias_mes.sort_values('Data', inplace=True)
        
        # Criar coluna 'Total' que soma todos os tipos de cirurgias
        df_cirurgias_mes['Total'] = df_cirurgias_mes[colunas_numericas].sum(axis=1)
        
        # Gráfico 1: Evolução Mensal do Total de Cirurgias com Linha de Tendência
        st.markdown("###### 1️⃣ Evolução Mensal do Total de Cirurgias com Linha de Tendência")
        
        # Converter 'Data' para numérico para a regressão
        df_cirurgias_mes['Data_Num'] = df_cirurgias_mes['Data'].map(datetime.toordinal)
        
        # Preparar os dados para o modelo de regressão
        X = df_cirurgias_mes['Data_Num'].values.reshape(-1, 1)
        y = df_cirurgias_mes['Total'].values
        
        # Ajustar o modelo de regressão linear
        model = LinearRegression()
        model.fit(X, y)
        
        # Prever valores para a linha de tendência
        df_cirurgias_mes['Trendline'] = model.predict(X)
        
        # Criar a figura
        fig_total_cirurgias = go.Figure()
        
        # Adicionar a linha original
        fig_total_cirurgias.add_trace(go.Scatter(
            x=df_cirurgias_mes['Data'],
            y=df_cirurgias_mes['Total'],
            mode='lines+markers+text',
            name='Número Total de Cirurgias',
            line=dict(color='#636EFA'),
            marker=dict(color='#636EFA'),
            text=df_cirurgias_mes['Total'],
            textposition='top center',
            texttemplate='%{text:.0f}'
        ))
        
        # Adicionar a linha de tendência
        fig_total_cirurgias.add_trace(go.Scatter(
            x=df_cirurgias_mes['Data'],
            y=df_cirurgias_mes['Trendline'],
            mode='lines',
            name='Linha de Tendência',
            line=dict(color='#EF553B', dash='dash')
        ))
        
        # Atualizar o layout
        fig_total_cirurgias.update_layout(
            title='Evolução Mensal do Número Total de Cirurgias com Linha de Tendência',
            xaxis_title='Mês',
            yaxis_title='Número Total de Cirurgias',
            xaxis=dict(tickformat='%b %Y'),
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        st.plotly_chart(fig_total_cirurgias, use_container_width=True)
        
        # Observação analítica
        st.markdown("**Observação analítica:**")
        
        total_geral_cirurgias = df_cirurgias_mes['Total'].sum()
        media_mensal_cirurgias = df_cirurgias_mes['Total'].mean()
        mes_maior_cirurgias = df_cirurgias_mes.loc[df_cirurgias_mes['Total'].idxmax(), 'Data'].strftime('%b %Y')
        valor_maior_cirurgias = df_cirurgias_mes['Total'].max()
        
        st.write(f"Durante o período analisado, foram realizadas **{int(total_geral_cirurgias)}** cirurgias no total, com uma média mensal de **{media_mensal_cirurgias:.1f}** cirurgias.")
        st.write(f"O mês com o maior número de cirurgias foi **{mes_maior_cirurgias}**, com um total de **{int(valor_maior_cirurgias)}** cirurgias.")
        
        # Interpretar a linha de tendência
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(X, y)
        
        if slope > 0:
            tendencia = "aumentando"
        else:
            tendencia = "diminuindo"
        
        st.write(f"A linha de tendência indica que o número total de cirurgias está **{tendencia}** ao longo do tempo.")
        st.write(f"O coeficiente de determinação (R²) do modelo é **{r_squared:.2f}**, indicando que aproximadamente **{r_squared*100:.1f}%** da variação no número total de cirurgias pode ser explicada pelo tempo.")
        
        st.markdown("---")
        
        # Reorganizar o DataFrame para o formato longo
        df_cirurgias_mes_melted = df_cirurgias_mes.melt(
            id_vars=['Data'],
            value_vars=colunas_numericas,
            var_name='Tipo de Cirurgia',
            value_name='Quantidade'
        )
        
        # Renomear os tipos de cirurgia para facilitar a leitura
        df_cirurgias_mes_melted['Tipo de Cirurgia'] = df_cirurgias_mes_melted['Tipo de Cirurgia'].map({
            COLUNAS["CIRURGIAS_MES"]["ELETIVAS_SUS"]: 'Eletivas/SUS',
            COLUNAS["CIRURGIAS_MES"]["ELETIVAS_SUPLEMENTAR"]: 'Eletivas/Suplementar',
            COLUNAS["CIRURGIAS_MES"]["URGENCIA_SUS"]: 'Urgência/SUS',
            COLUNAS["CIRURGIAS_MES"]["URGENCIA_SUPLEMENTAR"]: 'Urgência/Suplementar',
        })
        
        # Gráfico 2: Evolução Mensal das Cirurgias por Tipo e Classificação
        st.markdown("###### 2️⃣ Evolução Mensal das Cirurgias por Tipo e Classificação")
        
        # Criar o gráfico de linhas refinado
        fig_cirurgias_mes = px.line(
            df_cirurgias_mes_melted,
            x='Data',
            y='Quantidade',
            color='Tipo de Cirurgia',
            markers=True,
            labels={
                'Data': 'Mês',
                'Quantidade': 'Número de Cirurgias',
                'Tipo de Cirurgia': 'Tipo de Cirurgia'
            },
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data={'Data': '|%b %Y'}
        )
        fig_cirurgias_mes.update_traces(textposition='top center', texttemplate='%{y:.0f}')
        fig_cirurgias_mes.update_layout(
            xaxis_tickformat='%b %Y',
            xaxis_title='Mês',
            yaxis_title='Número de Cirurgias',
            legend_title='Tipo de Cirurgia',
            hovermode='x unified',
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        st.plotly_chart(fig_cirurgias_mes, use_container_width=True)
        
        # Observação analítica
        st.markdown("**Observação analítica:**")
        
        # Calcular o total de cirurgias por tipo
        totais_cirurgias = df_cirurgias_mes_melted.groupby('Tipo de Cirurgia')['Quantidade'].sum().reset_index()
        
        # Identificar o tipo de cirurgia mais frequente
        tipo_mais_frequente = totais_cirurgias.loc[totais_cirurgias['Quantidade'].idxmax(), 'Tipo de Cirurgia']
        quantidade_mais_frequente = totais_cirurgias['Quantidade'].max()
        
        st.write(f"O tipo de cirurgia mais frequente é **{tipo_mais_frequente}**, com um total de **{quantidade_mais_frequente}** cirurgias realizadas no período analisado.")
        
        st.markdown("---")
        
        # Gráfico 3: Distribuição Mensal das Cirurgias por Classificação
        st.markdown("###### 3️⃣ Distribuição Mensal das Cirurgias por Classificação")
        
        # Calcular o total de cirurgias por data para adicionar os rótulos
        df_totals = df_cirurgias_mes_melted.groupby('Data')['Quantidade'].sum().reset_index()
        
        fig_cirurgias_empilhadas = px.bar(
            df_cirurgias_mes_melted,
            x='Data',
            y='Quantidade',
            color='Tipo de Cirurgia',
            labels={
                'Data': 'Mês',
                'Quantidade': 'Número de Cirurgias',
                'Tipo de Cirurgia': 'Tipo de Cirurgia'
            },
            barmode='stack',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data={'Data': '|%b %Y'},
            text='Quantidade'
        )
        fig_cirurgias_empilhadas.update_traces(texttemplate='%{text:.0f}', textposition='inside')
        fig_cirurgias_empilhadas.update_layout(
            xaxis_tickformat='%b %Y',
            xaxis_title='Mês',
            yaxis_title='Número de Cirurgias',
            legend_title='Tipo de Cirurgia',
            margin=dict(l=40, r=40, t=80, b=40),
            height=500
        )
        
        # Adicionar rótulos de total em cima de cada barra
        for i, total in enumerate(df_totals['Quantidade']):
            fig_cirurgias_empilhadas.add_annotation(
                x=df_totals['Data'][i],
                y=total,
                text=str(int(total)),
                showarrow=False,
                yshift=5,
                font=dict(
                    color='black',
                    size=12
                ),
                align='center'
            )
        
        st.plotly_chart(fig_cirurgias_empilhadas, use_container_width=True)
        
        st.markdown("---")
        
        # Gráfico 4: Eficiência Global, Agendamento e Desempenho
        st.markdown("###### 4️⃣ Gráfico de Eficiência")
        
        with st.container():
            fig_eficiencia = go.Figure()
        
            fig_eficiencia.add_trace(go.Bar(
                x=['Eficiência Global', 'Eficiência de Agendamento', 'Eficiência de Desempenho'],
                y=[eficiencia_global * 100, eficiencia_agendamento * 100, eficiencia_desempenho * 100],
                marker_color=['#636EFA', '#EF553B', '#00CC96'],
                text=[f"{eficiencia_global * 100:.2f}%", f"{eficiencia_agendamento * 100:.2f}%", f"{eficiencia_desempenho * 100:.2f}%"],
                textposition='inside',
                insidetextanchor='middle'
            ))
        
            fig_eficiencia.update_layout(
                title="Eficiências Global, de Agendamento e de Desempenho (%)",
                yaxis=dict(title='Eficiência (%)', range=[0, 100]),
                xaxis=dict(title='Tipo de Eficiência'),
                showlegend=False,
                template='plotly_white',
                margin=dict(l=40, r=40, t=80, b=80)
            )
        
            st.plotly_chart(fig_eficiencia, use_container_width=True)
        
        st.markdown("---")
        
        # Gráfico 5: Distribuição de Salas Cirúrgicas por Classificação
        st.markdown("###### 5️⃣ Distribuição de Salas Cirúrgicas")
        
        with st.container():
            fig_pizza_salas = px.pie(
                df_classificacao_salas_cirurgicas,
                names=COLUNAS["CLASSIFICACAO_SALAS_CIRURGICAS"]["CLASSIFICACAO_SALAS_CIRURGICAS"],
                values=COLUNAS["CLASSIFICACAO_SALAS_CIRURGICAS"]["QUANTIDADE_SALAS_CIRURGICAS"],
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pizza_salas.update_traces(textposition='inside', textinfo='percent+label+value')
            fig_pizza_salas.update_layout(
                margin=dict(l=40, r=200, t=80, b=80),  # Aumentar a margem direita
                legend_title="Classificação",
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                )
            )
            st.plotly_chart(fig_pizza_salas, use_container_width=True)
        
        st.markdown("---")
    
        # Gráfico 6: Cirurgias Eletivas vs Urgência por Porte
        st.markdown("###### 6️⃣ Cirurgias Eletivas vs Urgência por Porte")
        
        # Preparar os dados para o gráfico
        df_salas_cirurgicas_porte_melted = df_salas_cirurgicas_porte.melt(
            id_vars=[COLUNAS["SALAS_CIRURGICAS_PORTE"]["PORTE_SALAS"]],
            value_vars=[
                COLUNAS["SALAS_CIRURGICAS_PORTE"]["QTD_ELETIVAS"],
                COLUNAS["SALAS_CIRURGICAS_PORTE"]["QTD_URGENCIA"]
            ],
            var_name='Tipo de Cirurgia',
            value_name='Quantidade'
        )
        
        df_salas_cirurgicas_porte_melted['Tipo de Cirurgia'] = df_salas_cirurgicas_porte_melted['Tipo de Cirurgia'].map({
            COLUNAS["SALAS_CIRURGICAS_PORTE"]["QTD_ELETIVAS"]: 'Eletivas',
            COLUNAS["SALAS_CIRURGICAS_PORTE"]["QTD_URGENCIA"]: 'Urgência'
        })
        
        with st.container():
            fig_bar_salas_porte = px.bar(
                df_salas_cirurgicas_porte_melted,
                x=COLUNAS["SALAS_CIRURGICAS_PORTE"]["PORTE_SALAS"],
                y='Quantidade',
                color='Tipo de Cirurgia',
                labels={
                    COLUNAS["SALAS_CIRURGICAS_PORTE"]["PORTE_SALAS"]: "Porte da Sala",
                    'Quantidade': "Número de Cirurgias",
                    'Tipo de Cirurgia': "Tipo de Cirurgia"
                },
                barmode="group",
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Set2,
                text='Quantidade'
            )
            fig_bar_salas_porte.update_traces(texttemplate='%{text}', textposition='inside')
            fig_bar_salas_porte.update_layout(
                xaxis_title='Porte da Sala',
                yaxis_title='Número de Cirurgias',
                legend_title='Tipo de Cirurgia',
                margin=dict(l=40, r=40, t=80, b=80)
            )
            st.plotly_chart(fig_bar_salas_porte, use_container_width=True)
        
        st.markdown("---")
        
        # Gráfico 7: Tempo Médio de Permanência em Leitos
        st.markdown("###### 7️⃣ Tempo Médio de Permanência em Leitos")
        
        with st.container():
            # Preparar os dados para o gráfico de pizza
            df_tempo_medio = df_tempo_permanencia_leitos.groupby(
                COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["CLASSIFICACAO_SALAS_CIRURGICAS"]
            )[COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"]].mean().reset_index()
        
            fig_pizza_perm_leitos = px.pie(
                df_tempo_medio,
                values=COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"],
                names=COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["CLASSIFICACAO_SALAS_CIRURGICAS"],
                title="Tempo Médio de Permanência por Classificação da Sala Cirúrgica",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
        
            # Adicionar labels internos
            fig_pizza_perm_leitos.update_traces(
                textposition='inside',
                textinfo='percent+label+value',
                insidetextorientation='radial',
                hoverinfo='label+percent+value',
                texttemplate='%{label}<br>%{percent:.1%}<br>%{value:.1f} min'
            )
        
            fig_pizza_perm_leitos.update_layout(
                legend_title="Classificação da Sala",
                margin=dict(l=20, r=20, t=40, b=20),
                height=350,  # Reduz a altura do gráfico
                width=450    # Define uma largura fixa para o gráfico
            )
        
            st.plotly_chart(fig_pizza_perm_leitos, use_container_width=False)  # Usa a largura definida, não a do container
        
            # Adicionar observação analítica
            st.markdown("**Observação analítica:**")
            classificacao_maior_tempo = df_tempo_medio.loc[
                df_tempo_medio[COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"]].idxmax(),
                COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["CLASSIFICACAO_SALAS_CIRURGICAS"]
            ]
            maior_tempo = df_tempo_medio[COLUNAS["TEMPO_PERMANENCIA_LEITOS"]["TEMPO_MEDIO_PERMANENCIA_LEITO"]].max()
        
            st.write(f"A classificação de sala cirúrgica com maior tempo médio de permanência é **{classificacao_maior_tempo}**, "
                     f"com uma média de **{maior_tempo:.1f} minutos**.")
            st.write("Isso pode indicar uma necessidade de otimização dos processos neste tipo de sala para reduzir o tempo de permanência.")
        
        st.markdown("---")
        # Gráfico 8: Atrasos e Substituição de Salas
        st.markdown("###### 8️⃣ Atrasos e Substituição de Salas")
        
        with st.container():
            tempos_medios = [
                tempo_medio_atraso_primeira,
                df_tempo_setup_sala[COLUNAS["TEMPO_SETUP_SALA"]["TEMPO_SETUP"]].mean(),
                df_tempo_substit_sala[COLUNAS["TEMPO_SUBSTIT_SALA"]["TEMPO_SUBSTITUICAO"]].mean()
            ]
            labels_tempos = [
                'Atraso 1ª Cirurgia',
                'Setup de Sala',
                'Substituição de Sala'
            ]
        
            fig_atraso_substituicao = go.Figure(data=[
                go.Bar(
                    x=labels_tempos,
                    y=tempos_medios,
                    text=[f'{t:.2f} min' for t in tempos_medios],
                    textposition='inside',
                    marker_color=px.colors.qualitative.Set2,
                )
            ])
        
            fig_atraso_substituicao.update_layout(
                yaxis_title='Tempo Médio (min)',
                xaxis_title='Atividade',
                template='plotly_white',
                margin=dict(l=40, r=40, t=80, b=80)
            )
            st.plotly_chart(fig_atraso_substituicao, use_container_width=True)
        
        st.markdown("---")
        
        # Gráfico 9: Motivos de Cancelamento de Cirurgias (Gráfico de Pareto)
        st.markdown("###### 9️⃣ Motivos de Cancelamento de Cirurgias")
        
        with st.container():
            # Agregar os dados para garantir um único valor por motivo
            df_motivos_cancelamento_agregado = df_motivos_cancelamento.groupby(
                COLUNAS["MOTIVOS_CANCELAMENTO"]["MOTIVO_CANCELAMENTO"]
            )[COLUNAS["MOTIVOS_CANCELAMENTO"]["QTD_CANCELAMENTO_MEDIA"]].sum().reset_index()
        
            # Ordenar os dados em ordem decrescente
            df_motivos_cancelamento_sorted = df_motivos_cancelamento_agregado.sort_values(
                by=COLUNAS["MOTIVOS_CANCELAMENTO"]["QTD_CANCELAMENTO_MEDIA"],
                ascending=False
            )
        
            # Calcular o percentual e o percentual acumulado
            total = df_motivos_cancelamento_sorted[COLUNAS["MOTIVOS_CANCELAMENTO"]["QTD_CANCELAMENTO_MEDIA"]].sum()
            df_motivos_cancelamento_sorted['Percentual'] = df_motivos_cancelamento_sorted[COLUNAS["MOTIVOS_CANCELAMENTO"]["QTD_CANCELAMENTO_MEDIA"]] / total * 100
            df_motivos_cancelamento_sorted['Percentual_Acumulado'] = df_motivos_cancelamento_sorted['Percentual'].cumsum()
        
            # Criar o gráfico de Pareto
            fig_pareto = go.Figure()
        
            # Adicionar as barras
            fig_pareto.add_trace(go.Bar(
                x=df_motivos_cancelamento_sorted[COLUNAS["MOTIVOS_CANCELAMENTO"]["MOTIVO_CANCELAMENTO"]],
                y=df_motivos_cancelamento_sorted[COLUNAS["MOTIVOS_CANCELAMENTO"]["QTD_CANCELAMENTO_MEDIA"]],
                name='Quantidade',
                marker_color='#1f77b4',
                text=df_motivos_cancelamento_sorted[COLUNAS["MOTIVOS_CANCELAMENTO"]["QTD_CANCELAMENTO_MEDIA"]].round(1),
                textposition='inside',
                textfont=dict(color='white'),
            ))
        
            # Adicionar a linha de percentual acumulado
            fig_pareto.add_trace(go.Scatter(
                x=df_motivos_cancelamento_sorted[COLUNAS["MOTIVOS_CANCELAMENTO"]["MOTIVO_CANCELAMENTO"]],
                y=df_motivos_cancelamento_sorted['Percentual_Acumulado'],
                name='Percentual Acumulado',
                mode='lines+markers+text',
                line=dict(color='red'),
                yaxis='y2',
                text=[f'{p:.1f}%' for p in df_motivos_cancelamento_sorted['Percentual_Acumulado']],
                textposition='top center',
            ))
        
            # Atualizar o layout
            fig_pareto.update_layout(
                title="Diagrama de Pareto: Motivos de Cancelamento de Cirurgias",
                xaxis_title="Motivo de Cancelamento",
                yaxis_title="Quantidade Média de Cancelamentos",
                yaxis2=dict(
                    title='Percentual Acumulado',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                xaxis={'categoryorder':'total descending'},
                margin=dict(l=50, r=50, t=80, b=100),
                height=600,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        
            # Rotacionar e ajustar os rótulos do eixo x
            fig_pareto.update_xaxes(tickangle=45, tickfont=dict(size=10))
        
            st.plotly_chart(fig_pareto, use_container_width=True)
        
            # Adicionar observação analítica
            st.markdown("**Observação analítica:**")
            motivo_mais_frequente = df_motivos_cancelamento_sorted.iloc[0][COLUNAS["MOTIVOS_CANCELAMENTO"]["MOTIVO_CANCELAMENTO"]]
            quantidade_mais_frequente = df_motivos_cancelamento_sorted.iloc[0][COLUNAS["MOTIVOS_CANCELAMENTO"]["QTD_CANCELAMENTO_MEDIA"]]
            percentual_acumulado = df_motivos_cancelamento_sorted.iloc[0]['Percentual_Acumulado']
        
            st.write(f"O motivo mais frequente de cancelamento de cirurgias é **'{motivo_mais_frequente}'**, "
                     f"com uma média de **{quantidade_mais_frequente:.1f}** cancelamentos, "
                     f"representando **{percentual_acumulado:.1f}%** do total de cancelamentos.")
            st.write("Este diagrama de Pareto ajuda a identificar os motivos mais significativos de cancelamentos, "
                     "permitindo priorizar ações para reduzir o número total de cancelamentos de cirurgias.")
        
        st.markdown("---")
                
        # Gráfico 11: Eficiência de Médicos no Centro Cirúrgico
        st.markdown("###### 🔟 Eficiência de Médicos no Centro Cirúrgico")
        
        with st.container():
            fig_medicos_cc = px.line(
                df_media_medicos_cc,
                x=COLUNAS["MEDIA_MEDICOS_CC"]["DIA"],
                y=[COLUNAS["MEDIA_MEDICOS_CC"]["MEDICOS_CIRURGIAO"], COLUNAS["MEDIA_MEDICOS_CC"]["MEDICOS_ANESTESISTA"]],
                labels={
                    COLUNAS["MEDIA_MEDICOS_CC"]["DIA"]: "Dia da Semana",
                    "value": "Quantidade Média",
                    "variable": "Tipo de Médico"
                },
                line_shape="spline",
                markers=True,
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_medicos_cc.update_traces(textposition='top center', texttemplate='%{y:.1f}')
            fig_medicos_cc.update_layout(
                xaxis_title='Dia da Semana',
                yaxis_title='Quantidade Média',
                legend_title='Tipo de Médico',
                margin=dict(l=40, r=40, t=80, b=80)
            )
            st.plotly_chart(fig_medicos_cc, use_container_width=True)
        
        st.markdown("---")
                    
# =====================================
# Parte 11: Aba "Fluxo do Processo"
# =====================================

    with tab7:
        st.markdown("## Desempenho dos Processos")
        st.markdown("""
        Nesta seção, você poderá analisar o macrofluxo de processos hospitalares utilizando conceitos da teoria das restrições e Lean.
        """)
    
        # Criar as sub-abas
        subtab1, subtab2, subtab3 = st.tabs([" ⏩ Atendimento Porta/Médico", " ⏩ Demanda/Especialidade", "⏩ Por Setores"])
    
        # ==========================
        # Sub-aba 1: Atendimento Porta/Médico
        # ==========================
        with subtab1:
            st.markdown("### Atendimento Porta/Médico")
    
            # ==========================
            # 1. Demanda de Pacientes por Hora (07:00 às 18:00)
            # ==========================
            df_horarios_filtered = df_horarios[(df_horarios['hora'] >= 7) & (df_horarios['hora'] <= 18)]
            demand_paciente_hora = df_horarios_filtered['quantidade_media_pacientes (arredondado)'].mean()
    
            # ==========================
            # 2. Headcount em Triagem (07:00 às 18:00)
            # ==========================
            df_triagem_enfermeiros_filtered = df_triagem_enfermeiros[
                (df_triagem_enfermeiros['hora'] >= 7) & (df_triagem_enfermeiros['hora'] <= 18)
            ]
            hc_triagem = math.ceil(df_triagem_enfermeiros_filtered['quantidade_media_enfermeiros (arredondado)'].mean())
    
            # ==========================
            # 3. Headcount em Consultório (07:00 às 18:00)
            # ==========================
            df_medicos_consulta_filtered = df_media_medicos_consulta[
                (df_media_medicos_consulta['hora'] >= 7) & (df_media_medicos_consulta['hora'] <= 18)
            ]
            hc_consultorio = math.ceil(df_medicos_consulta_filtered[COLUNAS["MEDIA_MEDICOS_CONSULTA"]["QUANTIDADE_MEDIA_MEDICOS"]].mean())
    
            # ==========================
            # 4. Tempo Médio de Atendimento na Triagem
            # ==========================
            tempo_medio_triagem = df_triagem_tempo[COLUNAS["TRIAGEM_TEMPO"]["TEMPO_MEDIO_ATENDIMENTO"]].iloc[0]
    
            # ==========================
            # 5. Tempo Médio de Atendimento no Consultório
            # ==========================
            df_consulta_tempo['etapa_normalizada'] = df_consulta_tempo[COLUNAS["CONSULTA_TEMPO"]["ETAPA"]].str.strip().str.lower()
            etapa_busca = 'atendimento médico'.lower()
    
            mask = df_consulta_tempo['etapa_normalizada'] == etapa_busca
    
            if mask.any():
                tempo_medio_consultorio = df_consulta_tempo.loc[mask, COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]].iloc[0]
            else:
                st.error("A etapa 'ATENDIMENTO MÉDICO' não foi encontrada em 'df_consulta_tempo'.")
                tempo_medio_consultorio = 0
    
            # ==========================
            # 6. Tempo Porta Médico
            # ==========================
            etapa_porta_medico = 'porta-médico'.lower()
            mask_porta = df_consulta_tempo['etapa_normalizada'] == etapa_porta_medico
    
            if mask_porta.any():
                tempo_porta_medico = df_consulta_tempo.loc[mask_porta, COLUNAS["CONSULTA_TEMPO"]["TEMPO_MEDIO_ETAPA"]].iloc[0]
            else:
                st.error("A etapa 'PORTA-MÉDICO' não foi encontrada.")
                tempo_porta_medico = 0
    
            TE_porta_medico = tempo_porta_medico
    
            # ==========================
            # Definição das Etapas do Processo
            # ==========================
            etapas = ['Triagem', 'Consultório']
    
            headcount_etapas = {
                'Triagem': hc_triagem,
                'Consultório': hc_consultorio
            }
    
            tempo_servico_etapas = {
                'Triagem': tempo_medio_triagem,
                'Consultório': tempo_medio_consultorio
            }
    
            tempo_servico_horas = {etapa: tempo / 60 for etapa, tempo in tempo_servico_etapas.items()}
            demanda_etapas = {etapa: demand_paciente_hora for etapa in etapas}
    
            # ==========================
            # Funções para calcular Wq em M/M/1 e M/M/c
            # ==========================
            def calc_Wq_MM1(lambda_, mu_):
                rho = lambda_ / mu_
                if rho >= 1:
                    return np.inf
                else:
                    return rho / (mu_ * (1 - rho))
    
            def calc_Wq_MMc(lambda_, mu_, c):
                rho = lambda_ / (c * mu_)
                if rho >= 1:
                    return np.inf
                else:
                    sum_terms = sum([((c * rho) ** n) / math.factorial(n) for n in range(int(c))])
                    last_term = ((c * rho) ** c) / (math.factorial(int(c)) * (1 - rho))
                    P0 = 1 / (sum_terms + last_term)
                    Lq = (((c * rho) ** c) * rho) / (math.factorial(int(c)) * ((1 - rho) ** 2)) * P0
                    return Lq / lambda_
    
            # ==========================
            # Cálculo das Variáveis, TE e Lq para cada Etapa
            # ==========================
            TE_etapas = {}
            Lq_etapas = {}
            capacidade_etapas = {}
            fator_utilizacao = {}
    
            for etapa in etapas:
                demanda = demanda_etapas[etapa]
                tempo_servico_h = tempo_servico_horas[etapa]
                mu_servidor = 1 / tempo_servico_h
                c = max(1, math.ceil(headcount_etapas[etapa]))
                capacidade_total = c * mu_servidor
    
                capacidade_etapas[etapa] = capacidade_total
                rho = demanda / capacidade_total
                fator_utilizacao[etapa] = rho
    
                if rho < 1:
                    if c == 1:
                        Wq_horas = calc_Wq_MM1(demanda, mu_servidor)
                        if Wq_horas != np.inf:
                            Lq = (rho ** 2) / (1 - rho)
                        else:
                            Lq = np.inf
                    else:
                        Wq_horas = calc_Wq_MMc(demanda, mu_servidor, c)
                        if Wq_horas != np.inf:
                            sum_terms = sum([((c * rho) ** n) / math.factorial(n) for n in range(int(c))])
                            last_term = ((c * rho) ** c) / (math.factorial(int(c)) * (1 - rho))
                            P0 = 1 / (sum_terms + last_term)
                            Lq = (((c * rho) ** c) * rho) / (math.factorial(int(c)) * ((1 - rho) ** 2)) * P0
                        else:
                            Lq = np.inf
                    TE = Wq_horas * 60  # Converter para minutos
                else:
                    TE = np.inf
                    Lq = np.inf
    
                TE_etapas[etapa] = TE
                Lq_etapas[etapa] = Lq
    
            # ==========================
            # Exibição dos Resultados Calculados
            # ==========================
            st.markdown("#### 📊 Métricas do Processo - Atendimento Porta/Médico")
    
            # Preparação da Tabela com os Dados Calculados
            df_tabela = pd.DataFrame({
                'Etapa': etapas,
                'Headcount': [headcount_etapas[etapa] for etapa in etapas],
                'Demanda – Pacientes/Hora (λ)': [demanda_etapas[etapa] for etapa in etapas],
                'Tempo Médio de Serviço (min)': [tempo_servico_etapas[etapa] for etapa in etapas],
                'Tempo Médio de Serviço (h)': [tempo_servico_horas[etapa] for etapa in etapas],
                'TAF - Taxa de Atendimento Pctes/h (μ_total)': [capacidade_etapas[etapa] for etapa in etapas],
                'Fator de Utilização % (ρ)': [fator_utilizacao[etapa] for etapa in etapas],
                'Número de Clientes na Fila (Lq)': [Lq_etapas[etapa] if not np.isinf(Lq_etapas[etapa]) else 'Infinito' for etapa in etapas],
                'Tempo de Espera (TE) (min)': [TE_etapas[etapa] if not np.isinf(TE_etapas[etapa]) else 'Infinito' for etapa in etapas],
            })
    
            # Exibição da Tabela Formatada
            df_tabela_display = df_tabela.copy()
            df_tabela_display['Fator de Utilização % (ρ)'] = df_tabela_display['Fator de Utilização % (ρ)'].apply(lambda x: f"{x:.2%}")
            df_tabela_display['Número de Clientes na Fila (Lq)'] = df_tabela_display['Número de Clientes na Fila (Lq)'].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
            )
            df_tabela_display['Tempo de Espera (TE) (min)'] = df_tabela_display['Tempo de Espera (TE) (min)'].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
            )
            st.dataframe(df_tabela_display.style.format({
                'Headcount': '{:.2f}',
                'Demanda – Pacientes/Hora (λ)': '{:.2f}',
                'Tempo Médio de Serviço (min)': '{:.2f}',
                'Tempo Médio de Serviço (h)': '{:.4f}',
                'TAF - Taxa de Atendimento Pctes/h (μ_total)': '{:.2f}',
            }))
    
            st.markdown("---")
    
            # ==========================
            # Cálculo de TE Recepção
            # ==========================
            TE_recepcao = TE_porta_medico - (TE_etapas['Triagem'] + tempo_servico_etapas['Triagem'] + TE_etapas['Consultório'])
    
            st.markdown("#### 📈 Visualizações Gráficas")
    
            st.markdown("#### 1️⃣ VSM adaptado - Atendimento Porta/Médico")
    
            # Identificar o gargalo (etapa com menor TAF)
            gargalo = min(etapas, key=lambda e: capacidade_etapas[e])
    
            # Preparar escala de cores para os tempos de espera
            TE_values = [te for te in TE_etapas.values() if not np.isinf(te)] + [TE_recepcao]
            norm = mcolors.Normalize(vmin=min(TE_values), vmax=max(TE_values))
            cmap = mcolors.LinearSegmentedColormap.from_list("", ["#FFCCCC", "#FF0000"])
    
            # Criar o diagrama VSM usando Graphviz
            dot = gv.Digraph(format='png')
            dot.attr(rankdir='LR', size='12,12')
    
            # Adicionar nó de início
            dot.node(
                'Início',
                'Início',
                shape='circle',
                style='filled',
                fillcolor='green',
                width='0.5', height='0.5', fixedsize='true',
                fontsize='14', fontcolor='black'
            )
    
            # Adicionar nó de TE Recepção
            color_te_recepcao = mcolors.to_hex(cmap(norm(TE_recepcao)))
            dot.node(
                'TE_Recepção',
                f"TE Recepção\n{TE_recepcao:.2f} min",
                shape='note',
                style='filled',
                fillcolor=color_te_recepcao,
                fontcolor='white' if color_te_recepcao == '#000000' else 'black',
                fontsize='12'
            )
    
            # Adicionar nó de Recepção
            dot.node(
                'Recepção',
                'Recepção',
                shape='rectangle',
                style='filled',
                fillcolor='lightblue',
                fontsize='14', width='2', height='1', fixedsize='true'
            )
    
            for etapa in etapas:
                tempo_espera = TE_etapas[etapa]
                color = mcolors.to_hex(cmap(norm(tempo_espera)))
                te_node = f"TE_{etapa}"
                dot.node(
                    te_node,
                    f"TE {etapa}\n{tempo_espera:.2f} min",
                    shape='note',
                    style='filled',
                    fillcolor=color,
                    fontcolor='white' if color == '#000000' else 'black',
                    fontsize='12'
                )
    
                # Adicionar o retângulo da etapa com HC e TS
                tempo_servico = tempo_servico_etapas[etapa]
                headcount = headcount_etapas[etapa]
                node_label = f"{etapa}\nTS: {tempo_servico:.2f} min\nHC: {headcount}"
                if etapa == gargalo:
                    node_label += "\n(Gargalo)"
                dot.node(
                    etapa,
                    node_label,
                    shape='rectangle',
                    style='filled',
                    fillcolor='purple' if etapa == gargalo else 'lightblue',
                    fontsize='14', width='2', height='1', fixedsize='true'
                )
    
            # Conectar os nós no diagrama
            dot.edge('Início', 'TE_Recepção')
            dot.edge('TE_Recepção', 'Recepção')
            dot.edge('Recepção', 'TE_Triagem')
            dot.edge('TE_Triagem', 'Triagem')
            dot.edge('Triagem', 'TE_Consultório')
            dot.edge('TE_Consultório', 'Consultório')
    
            # Adicionar a seta paralela para o Tempo Porta Médico
            dot.edge('Início', 'Consultório', label=f'Tempo Porta Médico\n{tempo_porta_medico:.2f} min', style='dashed', color='black', fontsize='14')
    
            # Renderizar o diagrama no Streamlit
            st.graphviz_chart(dot)
    
            st.markdown("#### 2️⃣ Análise de Capacidade, Demanda e Takt Time")
    
            col1, col2 = st.columns(2)
    
            with col1:
                # Gráfico de Capacidade vs Demanda
                fig_cap_dem = go.Figure()
                fig_cap_dem.add_trace(go.Bar(
                    x=etapas,
                    y=[capacidade_etapas[e] for e in etapas],
                    name='TAF - Taxa de Atendimento (μ_total)',
                    marker_color='blue',
                    text=[f"{capacidade_etapas[e]:.2f}" for e in etapas],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='white')
                ))
                fig_cap_dem.add_trace(go.Scatter(
                    x=etapas,
                    y=[demanda_etapas[e] for e in etapas],
                    mode='lines+markers+text',
                    name='Demanda – Pacientes/Hora (λ)',
                    line=dict(color='red', width=2),
                    text=[f"{demanda_etapas[e]:.2f}" for e in etapas],
                    textposition='top center'
                ))
                fig_cap_dem.update_layout(
                    title='Capacidade vs Demanda',
                    xaxis_title='Etapas',
                    yaxis_title='Pacientes/Hora'
                )
                st.plotly_chart(fig_cap_dem, use_container_width=True)
    
            with col2:
                # Gráfico de TS vs Takt Time vs TAF
                takt_time = 60 / demanda_etapas[etapas[0]]  # Assumindo que a demanda é a mesma para todas as etapas
                fig_ts_takt = go.Figure()
                fig_ts_takt.add_trace(go.Bar(
                    x=etapas,
                    y=[tempo_servico_etapas[e] for e in etapas],
                    name='Tempo de Serviço (TS)',
                    marker_color='lightblue',
                    text=[f"{tempo_servico_etapas[e]:.2f}" for e in etapas],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='black')
                ))
                fig_ts_takt.add_trace(go.Scatter(
                    x=etapas,
                    y=[takt_time] * len(etapas),
                    mode='lines',
                    name='Takt Time',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig_ts_takt.add_trace(go.Scatter(
                    x=etapas,
                    y=[60/capacidade_etapas[e] for e in etapas],
                    mode='markers',
                    name='TAF (min/paciente)',
                    marker=dict(color='green', size=10, symbol='diamond'),
                    text=[f"{(60/capacidade_etapas[e]):.2f}" for e in etapas],
                    textposition='top center'
                ))
                fig_ts_takt.update_layout(
                    title='TS vs Takt Time vs TAF',
                    xaxis_title='Etapas',
                    yaxis_title='Tempo (minutos)'
                )
                st.plotly_chart(fig_ts_takt, use_container_width=True)
    
            st.markdown("#### 3️⃣ Fator de Utilização por Etapa")
    
            fig_utilizacao = go.Figure(data=[
                go.Bar(
                    x=etapas,
                    y=[fator_utilizacao[e] * 100 for e in etapas],
                    text=[f"{fator_utilizacao[e]*100:.2f}%" for e in etapas],
                    textposition='inside',
                    insidetextanchor='middle',
                    marker_color='lightblue'
                )
            ])
    
            # Adicionar a linha pontilhada vermelha em 85%
            fig_utilizacao.add_hline(
                y=85,
                line_dash="dash",
                line_color="red",
                annotation_text="85% (ideal)",
                annotation_position="bottom right",
                annotation=dict(
                    font=dict(color="black")
                )
            )
    
            fig_utilizacao.update_layout(
                title='Fator de Utilização por Etapa',
                xaxis_title='Etapas',
                yaxis_title='Fator de Utilização (%)',
                yaxis_range=[0, 100]
            )
    
            st.plotly_chart(fig_utilizacao, use_container_width=True)
    
            # Adicionar observações analíticas
            st.markdown("#### Observações Analíticas:")
            st.write(f"- O gargalo do processo está na etapa de **{gargalo}**.")
            st.write(f"- O Takt Time do processo é de **{takt_time:.2f}** minutos por paciente.")
            etapa_maior_utilizacao = max(fator_utilizacao, key=fator_utilizacao.get)
            st.write(f"- A etapa com maior fator de utilização é **{etapa_maior_utilizacao}** com **{fator_utilizacao[etapa_maior_utilizacao]*100:.2f}%**.")
    
            st.markdown("---")
        # ==========================
        # Sub-aba 2: Demanda/Especialidade
        # ==========================
        with subtab2:
            st.markdown("### Demanda/Especialidade")
            st.markdown("#### 📊 Métricas do Processo - Demanda/Especialidade")
    
            # Filtrar o intervalo de 07:00 às 18:00
            df_horarios_intervalo = df_horarios[(df_horarios['hora'] >= 7) & (df_horarios['hora'] <= 18)]
    
            # Garantir que 'percentual_atendimento_dia' está em formato decimal
            df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["PERCENTUAL_ATENDIMENTO_DIA"]] = df_media_medicos_especialidade[
                COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["PERCENTUAL_ATENDIMENTO_DIA"]
            ].apply(porcentagem_para_float)
    
            # Obter a lista de especialidades
            especialidades = df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["ESPECIALIDADE"]].unique()
    
            # Criar um DataFrame para armazenar os pacientes por hora e por especialidade
            dfs_especialidades = []
    
            for especialidade in especialidades:
                # Obter a Taxa de Atendimento da especialidade
                taxa_atendimento = df_media_medicos_especialidade.loc[
                    df_media_medicos_especialidade[COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["ESPECIALIDADE"]] == especialidade,
                    COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["PERCENTUAL_ATENDIMENTO_DIA"]
                ].values[0]
    
                # Multiplicar a quantidade média de pacientes por hora pela taxa de atendimento da especialidade
                df_especialidade = df_horarios_intervalo.copy()
                df_especialidade['Especialidade'] = especialidade
                df_especialidade['Pacientes_Especialidade'] = df_especialidade['quantidade_media_pacientes'] * taxa_atendimento
    
                # Adicionar ao DataFrame da especialidade
                dfs_especialidades.append(df_especialidade)
    
            # Concatenar todos os DataFrames das especialidades
            df_pacientes_especialidades = pd.concat(dfs_especialidades, ignore_index=True)
    
            # Calcular a Demanda Pacientes/Hora para cada especialidade (média dos pacientes por hora)
            df_demanda_especialidades = df_pacientes_especialidades.groupby('Especialidade').agg({
                'Pacientes_Especialidade': 'mean'
            }).reset_index()
    
            # Unir com o DataFrame das especialidades para obter Headcount e outros dados
            df_especialidades_grouped = df_media_medicos_especialidade.merge(
                df_demanda_especialidades,
                left_on=COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["ESPECIALIDADE"],
                right_on='Especialidade',
                how='left'
            )
    
            # Converter 'quantidade_media_medicos' para 'Headcount'
            df_especialidades_grouped['Headcount'] = df_especialidades_grouped[
                COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["QUANTIDADE_MEDIA_MEDICOS"]
            ]
    
            # Calcular 'Pctes/dia' para cada especialidade usando a soma dos pacientes no intervalo
            df_pacientes_dia_especialidade = df_pacientes_especialidades.groupby('Especialidade').agg({
                'Pacientes_Especialidade': 'sum'
            }).reset_index()
    
            df_especialidades_grouped = df_especialidades_grouped.merge(
                df_pacientes_dia_especialidade,
                on='Especialidade',
                how='left',
                suffixes=('', '_Total')
            )
    
            df_especialidades_grouped.rename(columns={'Pacientes_Especialidade_Total': 'Pctes/dia'}, inplace=True)
    
            # Calcular o 'Tempo Médio de Serviço (min)' para cada especialidade
            df_especialidades_grouped['Tempo Médio de Serviço (min)'] = tempo_medio_consultorio
    
            # Converter 'Tempo Médio de Serviço (min)' para horas
            df_especialidades_grouped['Tempo Médio de Serviço (h)'] = df_especialidades_grouped['Tempo Médio de Serviço (min)'] / 60
    
            # Calcular 'TAF - Taxa de Atendimento Pctes/h (mu_total)'
            df_especialidades_grouped['TAF - Taxa de Atendimento Pctes/h (mu_total)'] = df_especialidades_grouped['Headcount'] / df_especialidades_grouped['Tempo Médio de Serviço (h)']
    
            # Calcular 'Fator de Utilização % (rho)'
            df_especialidades_grouped['Fator de Utilização % (rho)'] = df_especialidades_grouped['Pacientes_Especialidade'] / df_especialidades_grouped['TAF - Taxa de Atendimento Pctes/h (mu_total)']
    
            # Calcular 'Número de Clientes na Fila (Lq)' e 'Tempo de Espera (TE) (min)'
            def calcular_metrica_fila_especialidade(row):
                lambda_ = row['Pacientes_Especialidade']  # Demanda Pacientes/Hora
                mu_servidor = 1 / row['Tempo Médio de Serviço (h)']
                c = max(1, math.ceil(row['Headcount']))
                capacidade_total = c * mu_servidor
                rho = lambda_ / capacidade_total
                if rho < 1:
                    if c == 1:
                        Wq_horas = calc_Wq_MM1(lambda_, mu_servidor)
                        Lq = lambda_ * Wq_horas if Wq_horas != np.inf else np.inf
                    else:
                        Wq_horas = calc_Wq_MMc(lambda_, mu_servidor, c)
                        Lq = lambda_ * Wq_horas if Wq_horas != np.inf else np.inf
                    TE = Wq_horas * 60  # Converter para minutos
                else:
                    TE = np.inf
                    Lq = np.inf
                return pd.Series({'Número de Clientes na Fila (Lq)': Lq, 'Tempo de Espera (TE) (min)': TE})
    
            df_metrica_fila = df_especialidades_grouped.apply(calcular_metrica_fila_especialidade, axis=1)
    
            # Combinar os resultados
            df_especialidades_grouped = pd.concat([df_especialidades_grouped, df_metrica_fila], axis=1)
    
            # Formatação final do DataFrame para exibição
            df_especialidades_display = df_especialidades_grouped[[
                'Especialidade',
                'Headcount',
                COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["PERCENTUAL_ATENDIMENTO_DIA"],
                'Pctes/dia',
                'Pacientes_Especialidade',  # Demanda Pacientes/Hora
                'Tempo Médio de Serviço (min)',
                'Tempo Médio de Serviço (h)',
                'TAF - Taxa de Atendimento Pctes/h (mu_total)',
                'Fator de Utilização % (rho)',
                'Número de Clientes na Fila (Lq)',
                'Tempo de Espera (TE) (min)'
            ]].copy()
    
            # Renomear colunas para melhor apresentação
            df_especialidades_display.rename(columns={
                COLUNAS["MEDIA_MEDICOS_ESPECIALIDADE"]["PERCENTUAL_ATENDIMENTO_DIA"]: 'Taxa de Atendimento (%)',
                'Pacientes_Especialidade': 'Demanda (Pacientes/Hora)',
            }, inplace=True)
    
            # Converter 'Taxa de Atendimento (%)' para percentual
            df_especialidades_display['Taxa de Atendimento (%)'] = df_especialidades_display['Taxa de Atendimento (%)'] * 100
    
            # Formatar os valores para exibição
            df_especialidades_display['Fator de Utilização % (rho)'] = df_especialidades_display['Fator de Utilização % (rho)'].apply(lambda x: f"{x:.2%}")
            df_especialidades_display['Número de Clientes na Fila (Lq)'] = df_especialidades_display['Número de Clientes na Fila (Lq)'].apply(
                lambda x: f"{x:.2f}" if not np.isinf(x) else 'Infinito'
            )
            df_especialidades_display['Tempo de Espera (TE) (min)'] = df_especialidades_display['Tempo de Espera (TE) (min)'].apply(
                lambda x: f"{x:.2f}" if not np.isinf(x) else 'Infinito'
            )
    
            # Exibir a tabela no Streamlit
            st.dataframe(df_especialidades_display.style.format({
                'Headcount': '{:.2f}',
                'Taxa de Atendimento (%)': '{:.2f}%',
                'Pctes/dia': '{:.0f}',
                'Demanda (Pacientes/Hora)': '{:.2f}',
                'Tempo Médio de Serviço (min)': '{:.2f}',
                'Tempo Médio de Serviço (h)': '{:.4f}',
                'TAF - Taxa de Atendimento Pctes/h (mu_total)': '{:.2f}',
            }))
    
            st.markdown("---")
    
            # ==========================
            # Visualizações Gráficas para Especialidades
            # ==========================
    
            st.markdown("#### 📈 Visualizações Gráficas")
    
            # Gráfico 1: VSM adaptado - Fluxo por Especialidade
            st.markdown("##### 1️⃣ VSM adaptado - Fluxo por Especialidade")
    
            dot = gv.Digraph()
            dot.attr(rankdir='LR', size='12,12')
    
            # Nó de Consultório
            dot.node('start', 'Consultório',
                     shape='circle',
                     style='filled',
                     fillcolor='lightblue',
                     width='0.7',
                     height='0.7',
                     fixedsize='true',
                     fontsize='8')
    
            # Identificar o gargalo (menor TAF)
            gargalo_idx = df_especialidades_display['TAF - Taxa de Atendimento Pctes/h (mu_total)'].idxmin()
            gargalo_esp = df_especialidades_display.loc[gargalo_idx, 'Especialidade']
    
            # Criar escala de cores para TE
            te_values = df_especialidades_display['Tempo de Espera (TE) (min)'].replace('Infinito', np.inf).astype(float)
            te_values_finite = te_values[te_values != np.inf]
            vmin, vmax = te_values_finite.min(), te_values_finite.max()
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = mcolors.LinearSegmentedColormap.from_list("", ["#FFCCCC", "#FF0000"])
    
            # Criar nós para cada especialidade
            for idx, row in df_especialidades_display.iterrows():
                esp = row['Especialidade']
                te = row['Tempo de Espera (TE) (min)']
                ts = row['Tempo Médio de Serviço (min)']
    
                # Nó de TE
                if te == 'Infinito':
                    color = '#FF0000'  # Vermelho escuro para TE infinito
                else:
                    color = mcolors.to_hex(cmap(norm(float(te))))
                dot.node(f'te_{esp}', f'TE: {te} min', shape='note', style='filled', fillcolor=color)
    
                # Nó da especialidade
                color = 'purple' if esp == gargalo_esp else 'lightblue'
                label = f"{esp}\nTS: {ts:.2f} min"
                if esp == gargalo_esp:
                    label += "\nGARGALO"
                dot.node(esp, label, shape='box', style='filled', fillcolor=color)
    
                # Conexões
                dot.edge('start', f'te_{esp}')
                dot.edge(f'te_{esp}', esp)
    
            st.graphviz_chart(dot)
    
            # Gráficos de Análise
    
            st.markdown("##### 2️⃣ Análise de Capacidade vs Demanda")
    
            # Gráfico de Capacidade vs Demanda
            fig_cap_dem = go.Figure()
            fig_cap_dem.add_trace(go.Bar(
                x=df_especialidades_display['Especialidade'],
                y=df_especialidades_display['TAF - Taxa de Atendimento Pctes/h (mu_total)'],
                name='Capacidade (TAF)',
                text=df_especialidades_display['TAF - Taxa de Atendimento Pctes/h (mu_total)'].round(2),
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='black')
            ))
            fig_cap_dem.add_trace(go.Scatter(
                x=df_especialidades_display['Especialidade'],
                y=df_especialidades_display['Demanda (Pacientes/Hora)'],
                mode='lines+markers+text',
                name='Demanda',
                text=df_especialidades_display['Demanda (Pacientes/Hora)'].round(2),
                textposition='top center'
            ))
            fig_cap_dem.update_layout(
                title='Capacidade vs Demanda',
                xaxis_title='Especialidade',
                yaxis_title='Pacientes/Hora',
                barmode='group'
            )
            st.plotly_chart(fig_cap_dem, use_container_width=True)
    
            st.markdown("##### 3️⃣ Análise de TS vs Takt Time vs TAF")
    
            # Gráfico de TS vs Takt Time vs TAF
            takt_time = 60 / df_especialidades_display['Demanda (Pacientes/Hora)'].mean()
    
            fig_times = go.Figure()
            fig_times.add_trace(go.Bar(
                x=df_especialidades_display['Especialidade'],
                y=df_especialidades_display['Tempo Médio de Serviço (min)'],
                name='Tempo de Serviço (TS)',
                text=df_especialidades_display['Tempo Médio de Serviço (min)'].round(2),
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='black')
            ))
            fig_times.add_trace(go.Scatter(
                x=df_especialidades_display['Especialidade'],
                y=[takt_time] * len(df_especialidades_display),
                mode='lines',
                name='Takt Time',
                line=dict(color='red', dash='dash')
            ))
            fig_times.add_trace(go.Scatter(
                x=df_especialidades_display['Especialidade'],
                y=60 / df_especialidades_display['TAF - Taxa de Atendimento Pctes/h (mu_total)'],
                mode='markers',
                name='TAF (min/paciente)',
                marker=dict(color='green', size=10, symbol='diamond')
            ))
            fig_times.update_layout(
                title='TS vs Takt Time vs TAF',
                xaxis_title='Especialidade',
                yaxis_title='Tempo (minutos)'
            )
            st.plotly_chart(fig_times, use_container_width=True)
    
            st.markdown("##### 4️⃣ Fator de Utilização por Especialidade")
    
            # Gráfico de Fator de Utilização
            fig_util = go.Figure()
            fig_util.add_trace(go.Bar(
                x=df_especialidades_display['Especialidade'],
                y=df_especialidades_display['Fator de Utilização % (rho)'].str.rstrip('%').astype(float),
                marker_color=df_especialidades_display['Fator de Utilização % (rho)'].str.rstrip('%').astype(float),
                marker_colorscale='Blues',
                text=df_especialidades_display['Fator de Utilização % (rho)'],
                textposition='inside',
                insidetextanchor='middle',
                textfont=dict(color='black')
            ))
            fig_util.add_hline(y=85, line_dash="dash", line_color="red", annotation_text="85% (ideal)")
            fig_util.update_layout(
                title='Fator de Utilização por Especialidade',
                xaxis_title='Especialidade',
                yaxis_title='Fator de Utilização (%)',
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_util, use_container_width=True)
    
            # Observações Analíticas
            st.markdown("#### Observações Analíticas:")
    
            st.markdown(f"""
            ⚠️ **Nota Importante:** Estas análises são baseadas em um Tempo de Serviço (TS) padrão de 
            **{tempo_medio_consultorio:.2f} minutos** para todas as especialidades. Este valor é derivado da 
            etapa 'ATENDIMENTO MÉDICO' nos dados de tempo de consulta. Para uma análise mais precisa, 
            seria necessário considerar tempos específicos para cada especialidade.
            """)
    
            potenciais_gargalos = df_especialidades_display[df_especialidades_display['Fator de Utilização % (rho)'].str.rstrip('%').astype(float) > 85]['Especialidade'].tolist()
            if potenciais_gargalos:
                st.write("1. **Especialidades com Alta Utilização:** As seguintes especialidades apresentam um fator de utilização superior a 85%, o que pode indicar sobrecarga:")
                for esp in potenciais_gargalos:
                    st.write(f"   - {esp}")
    
            baixa_utilizacao = df_especialidades_display[df_especialidades_display['Fator de Utilização % (rho)'].str.rstrip('%').astype(float) < 50]['Especialidade'].tolist()
            if baixa_utilizacao:
                st.write("2. **Especialidades com Baixa Utilização:** As seguintes especialidades apresentam um fator de utilização inferior a 50%, o que pode sugerir capacidade ociosa:")
                for esp in baixa_utilizacao:
                    st.write(f"   - {esp}")
    
            ts_medio = df_especialidades_display['Tempo Médio de Serviço (min)'].mean()
            st.write(f"3. **Takt Time vs Tempo de Serviço:** O Takt Time global é de aproximadamente {takt_time:.2f} minutos, enquanto o TS médio é de {ts_medio:.2f} minutos.")
    
            if takt_time < ts_medio:
                st.write("   - *O que isso pode indicar?* O sistema pode estar sob pressão, pois o Takt Time é menor que o TS médio.")
            else:
                st.write("   - *O que isso pode indicar?* O sistema parece ter capacidade para atender à demanda atual, pois o Takt Time é maior que o TS médio.")
    
            st.markdown("---")
            

        # ==========================
        # Sub-aba 3: Setor
        # ==========================
        
        with subtab3:
            st.markdown("### Setor")
            st.markdown("#### 📊 Métricas do Processo - Demanda/Setor")
    
            # Função aprimorada para calcular métricas de fila com tratamento para grandes valores de 'c'
            def calcular_metricas_fila(lambda_, mu, c):
                rho = lambda_ / (c * mu)
                if rho >= 1 or c == 0:
                    return np.inf, np.inf  # Sistema instável ou inválido
                
                if c > 100:
                    # Aproximação para grandes valores de 'c'
                    Lq = rho / (1 - rho )
                else:
                    # Cálculo exato para c <= 100
                    lambda_mu = lambda_ / mu
                    c_int = int(c)
                    try:
                        log_sum_terms = [n * math.log(lambda_mu) - special.gammaln(n +1) for n in range(c_int)]
                        # Evita overflow ao calcular log_sum_exp
                        max_log = max(log_sum_terms)
                        sum_exp = sum( math.exp(x - max_log) for x in log_sum_terms )
                        log_sum = max_log + math.log(sum_exp)
                        log_last_term = c * math.log(lambda_mu) - special.gammaln(c +1) - math.log(1 - rho )
                        # P0 = 1 / (sum_{n=0}^{c-1} term_n + term_c )
                        # Utilizando log_sum_exp de forma simplificada
                        sum_total = math.exp(log_sum - log_sum) + math.exp(log_last_term - log_sum)
                        P0 = 1 / ( math.exp(log_sum) * (1 + math.exp(log_last_term - log_sum)) )
                        # Calcular Lq
                        Lq = ( (lambda_mu **c) * rho * P0 ) / ( special.gammaln(c +1) * (1 - rho )**2 )
                    except (OverflowError, ZeroDivisionError, ValueError):
                        return np.inf, np.inf  # Retorna infinito em caso de erro
                # Cálculo de Wq usando Lq
                Wq = Lq / lambda_ if lambda_ >0 else 0
                
                return Lq, Wq
        
            # Assumindo que df_passagem_setores e df_internacao_demanda são DataFrames já carregados do Excel
            passagem_setores = df_passagem_setores.copy()
            internacao_demanda = df_internacao_demanda.copy()
            
            # Mapeamento de setores
            setor_mapping = {
                "Geral": "Leitos Geral",
                "P.A. (ENF.)": "Leitos Enfermaria",
                "P.A. (UTI)": "Leitos UTI",
                "P.A. (CIRÚRGICOS)": "Leitos Cirúrgicos",
                "P.A. (CLÍNICOS)": "Leitos para Enfermaria (Origem P.A.)"
            }
            
            # Criar DataFrame final
            df_final = passagem_setores.copy()
            df_final['Capacidade (Leitos/Dia)'] = df_final['quantidade_leitos'] / df_final['tempo_medio_permanencia_dias']
            
            # Adicionar demanda
            demanda_dict = dict(zip(internacao_demanda['solicitacoes_leito'], internacao_demanda['media_solicitacoes_dia']))
            df_final['Demanda (Média Solicitações/Dia)'] = df_final['setores'].map(setor_mapping).map(demanda_dict)
            
            # Tratar possíveis valores NaN na Demanda
            df_final['Demanda (Média Solicitações/Dia)'] = df_final['Demanda (Média Solicitações/Dia)'].fillna(0)
            
            # Calcular Fator de Utilização
            df_final['Fator de Utilização (%)'] = (df_final['Demanda (Média Solicitações/Dia)'] / 
                                                   df_final['Capacidade (Leitos/Dia)']) * 100
            
            # Calcular métricas de fila
            df_final['Lq (Pacientes na Fila)'] = 0.0
            df_final['Wq (Tempo de Espera em Dias)'] = 0.0
            df_final['Wq (Tempo de Espera em Horas)'] = 0.0  # Nova coluna para Wq em horas
            
            for index, row in df_final.iterrows():
                lambda_ = row['Demanda (Média Solicitações/Dia)']
                mu = 1 / row['tempo_medio_permanencia_dias'] if row['tempo_medio_permanencia_dias'] > 0 else np.inf
                c = row['quantidade_leitos']
                try:
                    lq, wq = calcular_metricas_fila(lambda_, mu, c)
                    df_final.at[index, 'Lq (Pacientes na Fila)'] = lq
                    df_final.at[index, 'Wq (Tempo de Espera em Dias)'] = wq
                    df_final.at[index, 'Wq (Tempo de Espera em Horas)'] = wq * 24  # Converter dias em horas
                except (OverflowError, ZeroDivisionError, ValueError):
                    df_final.at[index, 'Lq (Pacientes na Fila)'] = np.inf
                    df_final.at[index, 'Wq (Tempo de Espera em Dias)'] = np.inf
                    df_final.at[index, 'Wq (Tempo de Espera em Horas)'] = np.inf
        
            # Formatar o DataFrame final
            df_final = df_final.rename(columns={
                'setores': 'Setores',
                'quantidade_leitos': 'Quantidade de Leitos',
                'tempo_medio_permanencia_dias': 'TMP (Dias)'
            })
        
            df_final = df_final[[
                'Setores', 'Quantidade de Leitos', 'TMP (Dias)', 'Capacidade (Leitos/Dia)',
                'Demanda (Média Solicitações/Dia)', 'Fator de Utilização (%)',
                'Lq (Pacientes na Fila)', 'Wq (Tempo de Espera em Dias)', 'Wq (Tempo de Espera em Horas)'
            ]]
        
            # Arredondar valores numéricos
            df_final = df_final.round({
                'Capacidade (Leitos/Dia)': 2,
                'Demanda (Média Solicitações/Dia)': 2,
                'Fator de Utilização (%)': 2,
                'Lq (Pacientes na Fila)': 2,
                'Wq (Tempo de Espera em Dias)': 4,   # Mais casas decimais para maior precisão
                'Wq (Tempo de Espera em Horas)': 2
            })
        
            # Exibir o DataFrame final
            st.dataframe(df_final)
        
            # Análises adicionais refinadas
            st.markdown("### 📊 Observações Analíticas")
        
            # Iterar pelos setores para exibir as observações
            for index, row in df_final.iterrows():
                setor = row['Setores']
                utilizacao = row['Fator de Utilização (%)']
                tempo_espera_dias = row['Wq (Tempo de Espera em Dias)']
                tempo_espera_horas = row['Wq (Tempo de Espera em Horas)']
                lq = row['Lq (Pacientes na Fila)']
                
                # Formatar o tempo de espera em dias e horas
                if np.isfinite(tempo_espera_dias):
                    tempo_espera_str = f"Aproximadamente {tempo_espera_dias:.4f} dias ou {tempo_espera_horas:.2f} horas"
                else:
                    tempo_espera_str = "Valor indefinido (possível instabilidade no sistema)"
                
                st.write(f"**Setor {setor}:**")
                st.write(f"- Fator de Utilização: {utilizacao:.2f}%")
                st.write(f"- Tempo Médio de Espera: {tempo_espera_str}")
                st.write(f"- Número Médio de Pacientes na Fila: {lq:.2f}")
                st.write("---")
        
            # Visualizações aprimoradas
            st.markdown("### 📈 Visualizações")
        
            # Gráfico de barras para Fator de Utilização
            st.subheader("Fator de Utilização por Setor")
            fig_utilizacao = px.bar(df_final, x='Setores', y='Fator de Utilização (%)',
                                    text='Fator de Utilização (%)',
                                    color='Fator de Utilização (%)',
                                    color_continuous_scale=px.colors.sequential.Reds)
            fig_utilizacao.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_utilizacao.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Utilização")
            fig_utilizacao.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_utilizacao, use_container_width=True)
        
            # Gráfico de dispersão para Demanda vs Capacidade
            st.subheader("Demanda vs Capacidade por Setor")
            fig_demanda_capacidade = px.scatter(df_final, 
                                                x='Capacidade (Leitos/Dia)', 
                                                y='Demanda (Média Solicitações/Dia)',
                                                size='Quantidade de Leitos', 
                                                color='Setores', 
                                                hover_name='Setores',
                                                text='Setores')
        
            fig_demanda_capacidade.update_traces(textposition='top center')
        
            # Adicionar linha de equilíbrio
            max_valor = max(df_final['Capacidade (Leitos/Dia)'].max(), df_final['Demanda (Média Solicitações/Dia)'].max())
            fig_demanda_capacidade.add_trace(
                go.Scatter(x=[0, max_valor], 
                           y=[0, max_valor],
                           mode='lines', 
                           line=dict(dash='dash', color='gray'),
                           name="Linha de Equilíbrio")
            )
        
            # Atualizar layout
            fig_demanda_capacidade.update_layout(
                height=500,
                xaxis_title="Capacidade (Leitos/Dia)",
                yaxis_title="Demanda (Média Solicitações/Dia)",
                legend_title="Setores",
                title="Demanda vs Capacidade por Setor"
            )
        
            # Exibir o gráfico
            st.plotly_chart(fig_demanda_capacidade, use_container_width=True)
        
            # Adicionar explicação
            st.write("""
            Este gráfico mostra a relação entre a capacidade e a demanda para cada setor. 
            - Pontos acima da linha de equilíbrio indicam setores onde a demanda pode exceder a capacidade.
            - Pontos abaixo da linha sugerem setores com possível capacidade excedente.
            - O tamanho de cada ponto representa a quantidade de leitos no setor.
            """)
        
            # Gráfico de barras para Tempo Médio de Permanência
            st.subheader("Tempo Médio de Permanência por Setor")
            fig_tmp = px.bar(df_final, x='Setores', y='TMP (Dias)',
                             text='TMP (Dias)', color='TMP (Dias)',
                             color_continuous_scale=px.colors.sequential.Blues)
            fig_tmp.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_tmp.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig_tmp, use_container_width=True)
        
            # Gráfico adicional: Distribuição das Demandas e Capacidades
            st.subheader("Distribuição das Demandas e Capacidades")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Box(
                y=df_final['Demanda (Média Solicitações/Dia)'],
                name='Demanda',
                marker_color='orange'
            ))
            fig_dist.add_trace(go.Box(
                y=df_final['Capacidade (Leitos/Dia)'],
                name='Capacidade',
                marker_color='green'
            ))
            fig_dist.update_layout(
                yaxis_title='Leitos/Dia',
                height=500
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.write("Este gráfico mostra a distribuição das demandas e capacidades entre os setores, permitindo identificar variações e possíveis outliers.")
        
            # Considerações finais
            st.markdown("### ⚠️ Considerações Finais")
            st.write("""
            As análises e visualizações apresentadas oferecem uma perspectiva baseada em modelos sobre o funcionamento dos diferentes setores do hospital. Ao interpretar estes resultados, considere que:
        
            1. **Simplificações dos Modelos:** Os modelos utilizados são simplificações da realidade complexa de um ambiente hospitalar.
            2. **Pontos de Partida:** As sugestões fornecidas são pontos de partida para discussões mais aprofundadas e não devem ser tomadas como diretrizes definitivas.
            3. **Experiência Prática:** A experiência prática dos profissionais de saúde e gestores é fundamental para contextualizar e validar estes insights analíticos.
            4. **Fatores Qualitativos:** Fatores qualitativos, como a natureza das condições médicas tratadas em cada setor, políticas de saúde locais e recursos humanos disponíveis, são cruciais e não estão refletidos nestes modelos quantitativos.
        
            Utilize estas informações como um componente de um processo de tomada de decisão mais amplo, sempre priorizando a segurança e o bem-estar dos pacientes e da equipe de saúde.
            """)

        
