
# ================================================================================
# LIBRERÍAS
# ================================================================================

# Importación de librerías
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

import pandas as pd
import pandasql as ps
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

from openai import OpenAI
from google import genai
from groq import Groq
import ollama

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score

# Ignorar warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# ================================================================================
# INICIALIZACIÓN DEL ESTADO DE LA SESIÓN
# ================================================================================

# Sufijos:

# Chat analítico:                         _llm
# Editor de SQL:                          _sql_editor
# Generador de SQL:                       _sql_code
# Generador + ejecutador de SQL:          _sql_auto

# Editor de Python:                       _py_editor
# Generador de Python básico:             _py_code1
# Generador de Python avanzado:           _py_code2
# Generador de Python con capa semántica: _py_code3
# Generador + ejecutador de Python:       _py_auto


# Caché durante la ejecución de la app
if 'cache_llm' not in st.session_state:
    st.session_state.cache_llm = {}

if 'cache_sql_editor' not in st.session_state:
    st.session_state.cache_sql_editor = {}

if 'cache_sql_code' not in st.session_state:
    st.session_state.cache_sql_code = {}

if 'cache_sql_auto' not in st.session_state:
    st.session_state.cache_sql_auto = {}

if 'cache_py_editor' not in st.session_state:
    st.session_state.cache_py_editor = {}

if 'cache_py_code1' not in st.session_state:
    st.session_state.cache_py_code1 = {}

if 'cache_py_code2' not in st.session_state:
    st.session_state.cache_py_code2 = {}

if 'cache_py_code3' not in st.session_state:
    st.session_state.cache_py_code3 = {}

if 'cache_py_auto' not in st.session_state:
    st.session_state.cache_py_auto = {}

if 'cache_ml' not in st.session_state:
    st.session_state.cache_ml = {}

# ================================================================================
# CONFIGURACIÓN INICIAL
# ================================================================================

# Configuración inicial de la app de Streamlit
st.set_page_config(page_title = 'GenAI App', 
                   page_icon='https://www.isdi.education/es/wp-content/uploads/2024/02/cropped-Favicon.png',
                   layout='wide')

# Título principal de la app
st.title('Sistema de Analítica Conversacional con IA Generativa')
# Línea separadora
st.divider()

# %%
# ================================================================================
# 1 - SISTEMA DE DECISIÓN DESCRIPTIVO
# ================================================================================

# Encabezado de sección
st.header('📊 N1. Sistema de Decisión Descriptivo', divider='gray')

# %%
# ================================================================================
# SECCIÓN 1.1 - CARGA DE DATOS
# ================================================================================

# Subtítulo
st.subheader('Datos disponibles')

# Ruta del dataset
BASE_DIR = Path(__file__).parent
DATASET = BASE_DIR / 'data' / 'fitlife_members.csv'

# Función cacheada para leer el dataset
@st.cache_data
def read_data(path):

    df = pd.read_csv(path)
    
    # Extracción de mes numérico de registro
    df['month_num'] = pd.to_datetime(df['month'], format='%Y-%m').dt.month
    # Extracción de trimestre de registro
    df['quarter'] = pd.to_datetime(df['month'], format='%Y-%m').dt.quarter
    # Extracción de año de registro
    df['year'] = pd.to_datetime(df['month'], format='%Y-%m').dt.year
    
    return df

# Mostrar dataset interactivo básico
try:
    # Lectura cacheada de dataset
    df = read_data(DATASET)

    # Definición de columnas numéricas
    NUM_COL1 = 'price_paid'
    NUM_COL2 = 'competitor_lowcost_price'
    
    # Configuración de columna con barra de progreso
    COLUMN_CONFIG = {
        NUM_COL1: st.column_config.ProgressColumn(
            min_value=0,                  # Valor mínimo de columna
            max_value=df[NUM_COL1].max(), # Valor máximo de columna
            format='compact'              # Formato compacto de número
        ),
        
        NUM_COL2: st.column_config.ProgressColumn(
            min_value=0,                  # Valor mínimo de columna
            max_value=df[NUM_COL2].max(), # Valor máximo de columna
            format='compact'              # Formato compacto de número
        )
    }

    # Texto explicativo
    st.write('A continuación se muestran los datos cargados:')
    # Mostrar tabla de datos
    #st.dataframe(df, column_config=COLUMN_CONFIG)

except Exception as e:
    # Mensaje de error
    st.error(f'Error al encontrar los datos: {e}')

# %%
# ================================================================================
# SECCIÓN DE FILTROS
# ================================================================================

# Cálculo de último mes y año
month_last = df['month_num'].max()
year_last = df['year'].max()

# Creacion de barra lateral
with st.sidebar:

    # Subtítulo
    st.subheader('Filtros de datos')

    # Filtros en barra lateral

    # Filtro de año
    years = st.multiselect('Años', options=sorted(df['year'].unique(), reverse=True), default=year_last)
    #Filtro de mes
    months = st.slider('Rango de meses', min_value=int(df['month_num'].min()), max_value=int(df['month_num'].max()), value=(month_last-2, month_last))
    # Filtro de centros
    centers_ordered = df.drop_duplicates(subset='member_id')['center'].value_counts().index.tolist()
    centers = st.multiselect('Centros', options=centers_ordered, placeholder='Elige uno o varios centros')
    # Filtro de planes
    plans = st.multiselect('Planes de suscripción', options=sorted(df['plan'].unique()), default=df['plan'].unique())
    # Filtro de canales de adquisición
    channels_ordered = df.drop_duplicates(subset='member_id')['acquisition_channel'].value_counts().index.tolist()
    acq_channels = st.multiselect('Canales de adquisición', options=channels_ordered, placeholder='Elige uno o varios canales')


# Aplicación de filtros

# Copia de DataFrame original
df_filtered = df.copy()

# Filtrar DataFrame por año
if years:
    df_filtered = df_filtered[df_filtered['year'].isin(years)]

# Filtrar DataFrame por rango de meses
df_filtered = df_filtered[(df_filtered['month_num'] >= months[0]) & (df_filtered['month_num'] <= months[1])]

# Filtrar DataFrame por centros
if centers:
    df_filtered = df_filtered[df_filtered['center'].isin(centers)]

# Filtrar DataFrame por planes
if plans:
    df_filtered = df_filtered[df_filtered['plan'].isin(plans)]

# Filtrar DataFrame por canales de adquisición
if acq_channels:
    df_filtered = df_filtered[df_filtered['acquisition_channel'].isin(acq_channels)]

# %%
# ================================================================================
# SECCIÓN 1.2 - CARGA DE DATOS AVANZADA
# ================================================================================

# Todas las columnas excepto 'campaign_active'
disabled_columns = [col for col in df_filtered.columns if col != 'campaign_active']

# Mostrar dataset interactivo avanzado
try:
    df_edited = st.data_editor(
        
        df_filtered.reset_index(drop=True), # Dataset                       
        width='stretch',                    # Máximo ancho disponible          
        hide_index=True,                    # Ocultar el índice
        num_rows='dynamic',                 # Permitir agregar/eliminar filas
        disabled=disabled_columns           # Columnas no editables
    )

    # Actualización de DataFrame con los cambios realizados en el editor
    df_filtered = df_edited

except Exception as e:
    # Mensaje de error
    st.error(f'Error al mostrar los datos: {e}')

# %%
# ================================================================================
# SECCIÓN 1.3 - REPORTE DE PERFILADO
# ================================================================================

# Subtítulo
st.subheader('Perfilado de datos')

# Función cacheada para generar de informe de perfil
@st.cache_data 
def generate_profile(df):

    profile = ProfileReport(df, explorative=True)
    return profile

# Desplegable con el reporte de perfilado del DataFrame
with st.expander('Reporte de perfilado', expanded=False):
    try:
        # Generación cacheada de informe de perfil
        profile = generate_profile(df_filtered)
        
        # Mostrar informe HTML dentro de Streamlit
        components.html(profile.to_html(), height=1200, scrolling=True)
    
    except Exception as e:
        # Mensaje de error
        st.error(f'Error al generar el perfilado: {e}')

# %%
# ================================================================================
# 1.4 - DEFINICIÓN Y CÁLCULO DE KPIS
# ================================================================================

# Subtítulo
st.subheader('Indicadores de negocio')

# Función cacheada para filtrar por trimestre actual y anterior
@st.cache_data
def filter_quarters(df, months, years):

    month_actual = months[1]
    quarter_actual = ((month_actual - 1) // 3) + 1
    year_actual = max(years)

    # Cálculo de trimestre anterior
    if quarter_actual == 1:
        quarter_prev = 4
        year_prev = year_actual - 1
    else:
        quarter_prev = quarter_actual - 1
        year_prev = year_actual

    # Filtrar por trimestre actual y trimestre anterior
    df_actual = df[(df['year'] == year_actual) & (df['quarter'] == quarter_actual)]
    df_prev = df[(df['year'] == year_prev) & (df['quarter'] == quarter_prev)]

    return df_actual, df_prev, quarter_actual, year_actual

# Filtrado cacheado por el trimestre actual y anterior
try:
    df_actual, df_prev, quarter_actual, year_actual = filter_quarters(df, months, years)
    
except Exception as e:
    # Mensaje de error
    st.error(f'Error al filtrar trimestres: {e}')

# Función cacheada para calcular KPIs
@st.cache_data
def calculate_kpis(df_actual, df_prev):

    # ----------------------------------------------------------------------------
    # 1. Base de miembros
    # ----------------------------------------------------------------------------
    
    # Miembros totales
    members = df_actual['member_id'].nunique()
    members_prev = df_prev['member_id'].nunique()
    
    # 1.2. Cancelaciones
    churned_members = df_actual[df_actual['status'] == 'churned']['member_id'].nunique()
    churned_members_prev = df_prev[df_prev['status'] == 'churned']['member_id'].nunique()
    delta_churn = ((churned_members - churned_members_prev) / churned_members_prev) * 100

    # 1.1. Tasa de cancelación (calculado sin distinguir miembros nuevos)
    churn_rate = (churned_members / members) * 100
    churn_rate_prev = (churned_members_prev / members_prev) * 100
    delta_churn_rate = ((churn_rate - churn_rate_prev) / churn_rate_prev) * 100

    # 1.3. Nuevos miembros
    df_new_members = df_actual[df_actual['tenure_months'] == 1]
    new_members = df_new_members['member_id'].nunique()
    
    df_new_members_prev = df_prev[df_prev['tenure_months'] == 1]
    new_members_prev = df_new_members_prev['member_id'].nunique()
    
    delta_new = ((new_members - new_members_prev) / new_members_prev) * 100

    # 1.4. Miembros activos
    active_members = df_actual[df_actual['status'] == 'active']['member_id'].nunique()
    active_members_prev = df_prev[df_prev['status'] == 'active']['member_id'].nunique()
    delta_active = ((active_members - active_members_prev) / active_members_prev) * 100

    # ----------------------------------------------------------------------------
    # 2. Resultados económicos
    # ----------------------------------------------------------------------------
    
    # 2.1. Ingresos
    total_revenue = df_actual['price_paid'].sum()
    total_revenue_prev = df_prev['price_paid'].sum()
    delta_revenue = ((total_revenue - total_revenue_prev) / total_revenue_prev) * 100

    # 2.2. ARPU
    arpu = total_revenue / members
    arpu_prev = total_revenue_prev / members_prev
    delta_arpu = ((arpu - arpu_prev) / arpu_prev) * 100

    # 3.1. CLV (vida media de miembro)
    clv = df_actual['tenure_months'].mean()
    clv_prev = df_prev['tenure_months'].mean()
    delta_clv = ((clv - clv_prev) / clv_prev) * 100
    
    # 2.3. LTV
    ltv = arpu * clv
    ltv_prev = arpu_prev * clv_prev
    delta_ltv = ((ltv - ltv_prev) / ltv_prev) * 100

    # ----------------------------------------------------------------------------
    # 3. Fidelidad y experiencia
    # ----------------------------------------------------------------------------
    
    # 3.2. NPS
    df_survey_actual = df_actual[df_actual['satisfaction_survey'].notna()]
    promoters = (df_survey_actual['satisfaction_survey'] >= 9).sum()
    detractors = (df_survey_actual['satisfaction_survey'] <= 6).sum()
    total_responses = len(df_survey_actual)

    nps = ((promoters - detractors) / total_responses * 100)

    df_survey_prev = df_prev[df_prev['satisfaction_survey'].notna()]
    promoters_prev = (df_survey_prev['satisfaction_survey'] >= 9).sum()
    detractors_prev = (df_survey_prev['satisfaction_survey'] <= 6).sum()
    total_responses_prev = len(df_survey_prev)

    nps_prev = ((promoters_prev - detractors_prev) / total_responses_prev * 100)

    delta_nps = ((nps - nps_prev) / nps_prev) * 100
    
    # 3.3. Valor de visita
    visit_value = (total_revenue / df_actual['visits_this_month'].sum())
    visit_value_prev = (total_revenue / df_prev['visits_this_month'].sum())
    delta_visit_value = ((visit_value - visit_value_prev) / visit_value_prev) * 100

    # 3.4. Visitas por miembro
    visits_per_member = df_actual['visits_this_month'].sum() / df_actual['member_id'].nunique()
    visits_per_member_prev = df_prev['visits_this_month'].sum() / df_prev['member_id'].nunique()
    delta_visits = ((visits_per_member - visits_per_member_prev) / visits_per_member_prev) * 100

    # 3.5. Incidencias reportadas por miembro
    incidents_per_member = df_actual['incidents_reported'].sum() / df_actual['member_id'].nunique()
    incidents_per_member_prev = df_prev['incidents_reported'].sum() / df_prev['member_id'].nunique()
    delta_incidents = ((incidents_per_member - incidents_per_member_prev) / incidents_per_member_prev) * 100    

    # ----------------------------------------------------------------------------
    # 4. Competencias y palancas
    # ----------------------------------------------------------------------------
    
    # 4.1. Captación por campañas
    new_with = df_new_members[df_new_members['campaign_active'].notna()]['member_id'].nunique()
    new_without = df_new_members[df_new_members['campaign_active'].isna()]['member_id'].nunique()
    campaign_singups = (new_with / (new_with + new_without)) * 100
    
    new_with_prev = df_new_members_prev[df_new_members_prev['campaign_active'].notna()]['member_id'].nunique()
    new_without_prev = df_new_members_prev[df_new_members_prev['campaign_active'].isna()]['member_id'].nunique()
    campaign_singups_prev = (new_with_prev / (new_with_prev + new_without_prev)) * 100
    
    delta_campaign = ((campaign_singups - campaign_singups_prev) / campaign_singups_prev) * 100
    
    # 4.2. Gap de precio con competencia
    df_basic_price = df_actual[df_actual['plan'] == 'basic']
    price_gap = (df_basic_price['price_paid'] - df_basic_price['competitor_lowcost_price']).mean()

    df_basic_price_prev = df_prev[df_prev['plan'] == 'basic']
    price_gap_prev = (df_basic_price_prev['price_paid'] - df_basic_price_prev['competitor_lowcost_price']).mean()
    
    delta_gap = ((price_gap - price_gap_prev) / price_gap_prev) * 100

    # 4.3. Sensibilidad al precio
    avg_price = df_actual['price_paid'].mean()
    avg_price_prev = df_prev['price_paid'].mean()
    delta_price = ((avg_price - avg_price_prev) / avg_price_prev) * 100
    price_elasticity = delta_active / delta_price

    # 4.4. Miembros en riesgo
    risk_members = (df_actual[df_actual['visits_this_month'] <= 5]['member_id'].nunique() / active_members) * 100
    risk_members_prev = (df_prev[df_prev['visits_this_month'] <= 5]['member_id'].nunique() / active_members_prev) * 100
    delta_risk_members = ((risk_members - risk_members_prev) / risk_members_prev) * 100  

    return {'churned_members': churned_members,           'churned_members_prev': churned_members_prev,           'delta_churn': delta_churn, 
            'churn_rate': churn_rate,                     'churn_rate_prev': churn_rate_prev,                     'delta_churn_rate': delta_churn_rate,
            'new_members': new_members,                   'new_members_prev': new_members_prev,                   'delta_new': delta_new,
            'active_members': active_members,             'active_members_prev': active_members_prev,             'delta_active': delta_active,
            'total_revenue': total_revenue,               'total_revenue_prev': total_revenue_prev,               'delta_revenue': delta_revenue,
            'arpu': arpu,                                 'arpu_prev': arpu_prev,                                 'delta_arpu': delta_arpu,
            'clv': clv,                                   'clv_prev': clv_prev,                                   'delta_clv': delta_clv,
            'ltv': ltv,                                   'ltv_prev': ltv_prev,                                   'delta_ltv': delta_ltv,
            'nps': nps,                                   'nps_prev': nps_prev,                                   'delta_nps': delta_nps,
            'visit_value': visit_value,                   'visit_value_prev': visit_value_prev,                   'delta_visit_value': delta_visit_value,
            'visits_per_member': visits_per_member,       'visits_per_member_prev': visits_per_member_prev,       'delta_visits': delta_visits,
            'incidents_per_member': incidents_per_member, 'incidents_per_member_prev': incidents_per_member_prev, 'delta_incidents': delta_incidents,
            'campaign_singups': campaign_singups,         'campaign_singups_prev': campaign_singups_prev,         'delta_campaign': delta_campaign,
            'price_gap': price_gap,                       'price_gap_prev': price_gap_prev,                       'delta_gap': delta_gap,
            'price_elasticity': price_elasticity,
            'risk_members': risk_members,                 'risk_members_prev': risk_members_prev,                 'delta_risk_members': delta_risk_members
           }

try:
    # Cálculo cacheado de KPIs
    kpis = calculate_kpis(df_actual, df_prev)
    
except Exception as e:
    # Mensaje de error
    st.error(f'Error al calcular las métricas: {e}')

# Texto antes de las métricas
st.caption(f'**Datos trimestrales** (correspondientes al Q{quarter_actual} de {year_actual}).')

# Función para formatear millones y miles
def format_number(n):
    if n >= 1_000_000:
        return f'{n/1_000_000:.1f}M'
    elif n >= 1_000:
        return f'{n/1_000:.1f}K'
    else:
        return n if n == int(n) else round(n, 1)

# Mostrar métricas principales en columnas
try:
    met1, met2, met3, met4 = st.columns(4)

    met1.metric('👥 Tasa de Cancelación', f"{format_number(kpis['churn_rate'])}%", f"{kpis['delta_churn_rate']:.1f}%", delta_color='inverse')
    met2.metric('📈 Ingresos', f"{format_number(kpis['total_revenue'])} €", f"{kpis['delta_revenue']:.1f}%")
    met3.metric('⭐ CLV', f"{format_number(kpis['clv']):.0f} meses", f"{kpis['delta_clv']:.1f}%")
    met4.metric('⚡ Captación por Campañas', f"{format_number(kpis['campaign_singups']):.0f}%", f"{kpis['delta_campaign']:.1f}%")
    
except Exception as e:
    #Mensaje de error
    st.error(f'Error al visualizar los KPIs principales: {e}')

# Mostrar métricas secundarias en columnas
try:
    # Creación de pestañas por categorías de KPIs
    tab1, tab2, tab3, tab4 = st.tabs(['👥 Miembros', '📈 Finanzas', '⭐ Fidelidad y experiencia', '⚡ Estrategia'])
    
    with tab1:
        met5, met6, met7, met8 = st.columns(4)
        met5.metric('👥 Cancelaciones', format_number(kpis['churned_members']), f"{kpis['delta_churn']:.1f}%", delta_color='inverse')
        met6.metric('👥 Nuevos Miembros', format_number(kpis['new_members']), f"{kpis['delta_new']:.1f}%")
        met7.metric('👥 Total de Miembros Activos', format_number(kpis['active_members']), f"{kpis['delta_active']:.1f}%")
        
    with tab2:
        met9, met10, met11, met12 = st.columns(4)
        met9.metric('📈 ARPU (ingreso promedio por miembro)', f"{format_number(kpis['arpu'])} €", f"{kpis['delta_arpu']:.1f}%")
        met10.metric('📈 LTV (valor de vida de miembro)', f"{format_number(kpis['ltv'])} €", f"{kpis['delta_ltv']:.1f}%")

    with tab3:
        met13, met14, met15, met16 = st.columns(4)
        met13.metric('⭐ NPS', format_number(kpis['nps']), f"{kpis['delta_nps']:.1f}%")
        met14.metric('⭐ Valor de Visita (ratio de visitas por ingresos)', f"{format_number(kpis['visit_value'])} €", f"{kpis['delta_visit_value']:.1f}%")
        met15.metric('⭐ Visitas por Miembro', f"{format_number(kpis['visits_per_member']):.0f} visitas", f"{kpis['delta_visits']:.0f}%")
        met16.metric('⭐ Incidencias Reportadas por Miembro', f"{format_number(kpis['incidents_per_member'])} incidencias", f"{kpis['delta_incidents']:.1f}%", delta_color='inverse')

    with tab4:
        met17, met18, met19, met20 = st.columns(4)
        met17.metric('⚡ Gap de Precio con Competencia', f"{format_number(kpis['price_gap'])} €", f"{kpis['delta_gap']:.1f}%", delta_color='off')
        met18.metric('⚡ Sensibilidad al Precio', f"{kpis['price_elasticity']:.2f}")
        met19.metric('⚡ Miembros en Riesgo', f"{kpis['risk_members']:.0f}%", f"{kpis['delta_risk_members']:.1f}%", delta_color='inverse')

except Exception as e:
    st.error(f'Error al visualizar los KPIs secundarios: {e}')

# %%
# ================================================================================
# 1.5 - VISUALIZACIÓN DE DATOS
# ================================================================================

# Subtítulo
st.subheader('Visualización descriptiva')

# Creación de dos columnas
fig1, fig2 = st.columns(2)

# Encabezado
fig1.write('##### Evolución histórica de membresía')
fig1.caption('Este gráfico no está sujeto a los filtros aplicados.')

# Función cacheada para calcular los miembros activos por mes
@st.cache_data
def get_monthly_members(df):
    
    # Excluir cancelaciones
    df_active = df[df['status'] != 'churned']

    # Miembros activos por mes
    monthly_members = df_active.groupby('month').agg({'member_id':'nunique', 'competitor_lowcost_price':'mean'}).reset_index()    
    return monthly_members

try:
    # Cálculo cacheado de miembros activos por mes
    monthly_members = get_monthly_members(df)
    
    # Gráfico de líneas
    fig_line1 = px.line(monthly_members, x='month', y='member_id',
                         # Etiquetas a los ejes x e y
                         labels = {'month':'Mes', 'member_id':'Miembros activos'}) 
    
    # Mostrar gráfico
    fig1.plotly_chart(fig_line1, width='stretch')
    
except Exception as e:
    # Mensaje de error
    fig1.error(f'Error al visualizar: {e}')

# Encabezado
fig2.write('##### Comparativa mensual de precios: plan básico vs. competencia')
fig2.caption('Este gráfico no está sujeto a los filtros aplicados.')

# Función cacheada para calcular el precio medio de plan básico y competencia por mes
@st.cache_data
def get_monthly_price(df):
    
    # Filtrar registros de plan básico
    df_active = df[df['plan'] == 'basic']

    # Precio medio de plan básico y competencia por mes
    monthly_price = df_active.groupby('month').agg({'price_paid':'mean', 'competitor_lowcost_price':'mean'}).reset_index()   
    return monthly_price

try:
    # Cálculo cacheado de precio medio de plan básico y competencia por mes
    monthly_price = get_monthly_price(df)
    
    # Gráfico de líneas
    fig_line2 = px.line(monthly_price, x='month', y=['price_paid', 'competitor_lowcost_price'], 
                         
                         # Etiquetas a los ejes x e y, y leyenda
                         labels = {
                             'month':'Mes', 
                             'value':'Precio promedio (€)', 
                             'variable':'Tipo de precio', 
                             'price_paid':'Plan básico (FitLife)', 'competitor_lowcost_price':'Competencia'}, 
                         
                         # Colores de líneas
                         color_discrete_map = {
                             'price_paid':'royalblue', 
                             'competitor_lowcost_price':'orange'})

    # Renombrado de valores de leyenda
    price_names = {'price_paid':'Plan básico (FitLife)', 'competitor_lowcost_price':'Competencia'}
    fig_line2.for_each_trace(lambda t: t.update(name = price_names.get(t.name, t.name)))
    # Limitación de eje y
    fig_line2.update_yaxes(rangemode='tozero')

    # Mostrar gráfico
    fig2.plotly_chart(fig_line2, width='stretch')
    
except Exception as e:
    # Mensaje de error
    fig2.error(f'Error al visualizar: {e}')

# Creación de tres columnas
fig3, fig4, fig5 = st.columns(3)

# Encabezado
fig3.write('##### Ranking de gimnasios FitLife')

# Función cacheada para calcular el precio medio de plan básico y competencia por mes
@st.cache_data
def get_revenue_per_centre(df):
    
    # Ingresos por centro
    revenue_per_centre = df.groupby('center').agg({'price_paid':'sum'}).reset_index()  
    return revenue_per_centre

try:
    # Cálculo cacheado de ingresos por centro
    revenue_per_centre = get_revenue_per_centre(df_filtered)

    # Gráfico de barras
    fig_bar = px.bar(revenue_per_centre, x='price_paid', y='center',
                     # Etiquetas a los ejes x e y
                     labels={'price_paid':'Ingresos obtenidos (€)', 'center':'Centro'},
                     # Orden de barras
                     category_orders={'center': revenue_per_centre.sort_values('price_paid', ascending=False)['center'].tolist()})

    # Mostrar gráfico
    fig3.plotly_chart(fig_bar, width='stretch')
    
except Exception as e:
    # Mensaje de error
    fig3.error(f'Error al visualizar: {e}')

# Encabezado
fig4.write('##### Distribución de planes de suscripción')

# Función cacheada para calcular el número de miembros por plan
@st.cache_data
def get_plan_distribution(df):
    
    # Últimos registros por cada 'member_id'
    df_current_plan = df[['month', 'member_id', 'plan', 'status']].sort_values(['member_id', 'month']).drop_duplicates(subset='member_id', keep='last')
    # Filtrar por miembros activos
    df_current_plan = df_current_plan[df_current_plan['status'] == 'active']
    
    # Número de miembros por plan
    plan_distribution = df_current_plan['plan'].value_counts().reset_index()
    return plan_distribution

try:
    # Cálculo cacheado del número de miembros por plan
    plan_distribution = get_plan_distribution(df_filtered)

    # Gráfico de tarta
    fig_pie = px.pie(plan_distribution, names='plan', values='count')
    
    # Mostrar gráfico
    fig4.plotly_chart(fig_pie, width='stretch')
    
except Exception as e:
    # Mensaje de error
    fig4.error(f'Error al visualizar: {e}')

# Encabezado
fig5.write('##### Actividad mensual de miembros')
fig5.caption('Este gráfico no está sujeto a los filtros aplicados.')

# Función cacheada para calcular las visitas totales por año y mes
@st.cache_data
def get_monthly_visits_pivoted(df):
    
    # Visitas totales por año y mes
    monthly_visits = df.groupby(['year', 'month_num']).agg({'visits_this_month':'sum'}).reset_index()
    
    # Reestructuración de agregación a formato pivot
    monthly_visits_pivoted = monthly_visits.pivot(index='month_num', columns='year', values='visits_this_month')
    monthly_visits_pivoted = monthly_visits_pivoted.sort_index()
    
    return monthly_visits_pivoted

try:
    # Visitas totales por año y mes
    monthly_visits_pivoted = get_monthly_visits_pivoted(df)
    
    # Mapa de calor
    fig_heatmap = px.imshow(monthly_visits_pivoted, x=monthly_visits_pivoted.columns, y=monthly_visits_pivoted.index,                          
                            labels={'x':'Año', 'y':'Mes', 'color':'Visitas'},
                            color_continuous_scale='OrRd',
                            text_auto=True,
                            aspect='auto'
                           )
    
    # Mostrar todos los meses en las etiquetas
    month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    fig_heatmap.update_yaxes(tickmode='array', tickvals=monthly_visits_pivoted.index, ticktext=[month_names[i-1] for i in monthly_visits_pivoted.index])
    
    # Mostrar gráfico
    fig5.plotly_chart(fig_heatmap, width='stretch')
    
except Exception as e:
    # Mensaje de error
    fig5.error(f'Error al visualizar: {e}')

# %%
# ================================================================================
# 2 - SISTEMA PARAMETRIZABLE CONVERSACIONAL
# ================================================================================

# Encabezado de sección
st.header('🗨️ N2. Sistema Parametrizable Conversacional', divider='gray')

# Variables para LLM

# API Key
GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
GROQ_API_KEY = st.secrets['GROQ_API_KEY']

# Selección de modelo
#LLM_MODEL = 'gemini-2.5-flash'     # Gemini
LLM_MODEL = 'gpt-4.1-nano'         # OpenAI (alternativas: 'gpt-4.1-mini', 'o4-mini')
#LLM_MODEL = 'llama-3.1-8b-instant' # Groq
#LLM_MODEL = 'llama3'               # Ollama

# Variables para construir prompts

# Datos
data = df.to_string()

# Muestra de datos
data_sample = df[df['year'] == 2024].head(2).to_string()

# Estructura de DataFrame
schema = df.dtypes#.to_string()

# %%
# ================================================================================
# SECCIÓN 2.1 - EXPORTACIÓN DE DATOS FILTRADOS
# ================================================================================

with st.sidebar:
    
    # Línea separadora
    st.divider()
    # Subtítulo
    st.subheader('Exportar datos filtrados')

    # Botón de descarga
    st.download_button(label='📥 Descargar CSV',
                       data=df_filtered.to_csv(index=False), # Exportar DataFrame filtrado
                       file_name=f'data_exported.csv',       # Nombre de archivo
                       mime='text/csv'                       # Para exportar archivos de texto plano
                      )

# %%
# ================================================================================
# 2.2 - CONEXIÓN A LLM VÍA API
# ================================================================================

# Inicialización de LLM con la key directamente
try:
    # Gemini
    client = genai.Client(api_key=GEMINI_API_KEY)

    # OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Groq
    #client = Groq(api_key=GROQ_API_KEY)
    
except Exception as e:
    # Mensaje de error
    st.error(f'Error con la clave del LLM: {e}')

# Subtítulo
st.subheader('Chat analítico')
# Resumen de sección
st.markdown('Utiliza este chat para realizar preguntas sobre FitLife. '
            'El modelo responderá basándose directa y únicamente en la información contenida en el dataset.')

# Formulario con caja de texto y botón
with st.form(key='form_llm', border=False):
    # Pregunta de usuario
    question_llm = st.text_input(placeholder='Escribe una pregunta sobre los datos', key='input_llm', label='', label_visibility='collapsed')
    btn_llm = st.form_submit_button('Preguntar sobre los datos')

# Responder pregunta sobre los datos
if btn_llm and question_llm:
    if question_llm not in st.session_state.cache_llm:
        try:
            # Prompt
            prompt_llm = ('Actúa como un Senior Data Analyst experto en el sector de fitness y gestión de gimnasios.\n\n'

                          '## OBJETIVO:\n'
                          'Responder de forma precisa, técnica y analítica a la solicitud del usuario sobre los datos.\n\n'
                          
                          f'## DATOS DISPONIBLES:\n{data_sample}\n\n'
                          

                          f'## SOLICITUD DE USUARIO:\n{question_llm}\n\n'
                   
                          '## INSTRUCCIONES OBLIGATORIAS:\n'
                          '1. Basa tu respuesta únicamente en los datos proporcionados.\n'
                          '2. Evita suposiciones o datos inventados.\n\n'
                          
                          '## FORMATO DE RESPUESTA:\n'
                          '- Sé conciso y claro; máximo una línea de explicación intermedia si es necesaria.\n'
                          '- Devuelve sólo la información solicitada. Si es cálculo o resumen, muestra sólo el resultado principal.'
                         ) 

            with st.spinner(f'{LLM_MODEL} está procesando tu pregunta...'):
                
                # Llamada al LLM
                #response_llm = client.models.generate_content(model=LLM_MODEL, contents=prompt_llm)
                #st.session_state.cache_llm[question_llm] = response_llm.text

                # Opción con OpenAI
                response_llm = client.responses.create(model=LLM_MODEL, input=prompt_llm)
                st.session_state.cache_llm[question_llm] = response_llm.output_text
                
                # Opción con Groq
                #response_llm = client.chat.completions.create(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt_llm}])
                #response_llm.text = response_llm.choices[0].message.content

                # Opción con Ollama
                #response_llm = ollama.generate(model='llama3', prompt=prompt_llm, options={'temperature': 0.1})
                #st.session_state.cache_llm[question_llm] = response_llm['response']
        
        except Exception as e:
            # Mensaje de error
            st.error(f'Error al procesar la pregunta: {e}')

if question_llm in st.session_state.cache_llm:
    # Resultado del LLM
    st.success('Análisis completado')
    st.write(st.session_state.cache_llm[question_llm])

# %%
# ================================================================================
# 2.3 - EJECUCIÓN DE CONSULTAS CON SQL
# ================================================================================

# Subtítulo
st.subheader('Editor manual de consultas SQL')
# Resumen de sección
st.markdown('Escribe y ejecuta tus propias consultas SQL para explorar los datos de FitLife. '
            'Usa la sintaxis estándar de SQL y las columnas de **`members`**.\n\n'
            
            '**Ejemplo:**\n'
            '```sql\n'
            'SELECT member_id, SUM(visits_this_month) AS visits_last_month\n'
            'FROM members\n'
            'WHERE month = "2024-12"\n'
            'GROUP BY 1\n'
            'ORDER BY 2 DESC\n'
            'LIMIT 10\n'
            '```')

# Formulario con caja de texto y botón
with st.form(key='form_sql_editor', border=False):
    # Query de usuario
    query = st.text_area(placeholder='Escribe una query', height=30, key='input_sql_editor', label='', label_visibility='collapsed')
    btn_sql_editor = st.form_submit_button('Ejecutar query')

# Ejecutar consultas con SQL
if btn_sql_editor and query:
    try:
        # Ejecución de query
        result_sql_editor = ps.sqldf(query, {'members': df})
        st.session_state.cache_sql_editor[query] = result_sql_editor
        
    except Exception as e:
        # Mensaje de error
        st.error(f'Error al ejecutar la query: {e}')

if query in st.session_state.cache_sql_editor:
    # Resultado de query
    st.success('Query ejecutada correctamente')
    st.dataframe(st.session_state.cache_sql_editor[query])

# %%
# ================================================================================
# 2.4 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A SQL
# ================================================================================

# Subtítulo
st.subheader('Generador de consultas SQL')
# Resumen de sección
st.markdown('Convierte descripciones en lenguaje natural en consultas SQL listas para ejecutar. '
            'Escribe en palabras simples lo que necesitas y obtendrás la consulta preparada para copiar y ejecutar.')

# Formulario con caja de texto y botón
with st.form(key='form_sql_code', border=False):
    # Pregunta de usuario
    question_sql_code = st.text_input(placeholder='Escribe una consulta en lenguaje natural para convertirla en SQL', key='input_sql_code', label='', label_visibility='collapsed')
    btn_sql_code = st.form_submit_button('Generar consulta SQL')

# Convertir pregunta en consulta SQL
if btn_sql_code and question_sql_code:
        if question_sql_code not in st.session_state.cache_sql_code:
            try:
                # Prompt
                prompt_sql_code = ('Actúa como un experto en consultas SQL.\n\n'

                                   '## OBJETIVO:\n'
                                   'Convertir la solicitud del usuario en una consulta SQL válida.\n\n'

                                   f'## ESTRUCTURA DE TABLA (COLUMNAS Y TIPOS DE DATOS):\n{schema}\n\n'

                                   f'## SOLICITUD DE USUARIO:\n{question_sql_code}\n\n'
                                                                      
                                   '## INSTRUCCIONES OBLIGATORIAS:\n'
                                   '1. La tabla se llama exactamente members.\n'
                                   '2. Todas las variables (columnas) deben ir entre comillas dobles "".\n'
                                   '3. Usa comillas simples para valores de texto.\n'
                                   '4. Usa únicamente las columnas proporcionadas.\n\n'
                                   
                                   '## FORMATO DE RESPUESTA:\n'
                                   '- Devuelve sólo el código SQL, sin texto adicional.'
                                  ) 
            
                with st.spinner(f'{LLM_MODEL} está procesando tu consulta...'):
                    # Llamada al LLM
                    response_sql_code = client.models.generate_content(model=LLM_MODEL, contents=prompt_sql_code)
                    st.session_state.cache_sql_code[question_sql_code] = response_sql_code.text
    
            except Exception as e:
                # Mensaje de error
                st.error(f'Error al generar la query: {e}')

if question_sql_code in st.session_state.cache_sql_code:    
    # Resultado del LLM
    st.success('Consulta generada')
    st.write(st.session_state.cache_sql_code[question_sql_code])

# %%
# ================================================================================
# 2.5 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A SQL + EJECUCIÓN
# ================================================================================

# Subtítulo
st.subheader('Motor de consultas SQL con lenguaje natural')
# Resumen de sección
st.markdown('Transforma tus preguntas en resultados estructurados al instante. '
            'El motor genera automáticamente la consulta SQL, la ejecuta y muestra la información correspondiente.')

# Formulario con caja de texto y botón
with st.form(key='form_sql_auto', border=False):  
    # Pregunta de usuario
    question_sql_auto = st.text_input(placeholder='Escribe una pregunta en el motor SQL automático', key='input_sql_auto', label='', label_visibility='collapsed')
    btn_sql_auto = st.form_submit_button('Generar y ejecutar consulta SQL')

# Convertir pregunta en consulta SQL y ejecutarla
if btn_sql_auto and question_sql_auto:
        if question_sql_auto not in st.session_state.cache_sql_auto:
            try:
                # Prompt
                prompt_sql_auto = ('Actúa como un experto en consultas SQL.\n\n'
                       
                                   '## OBJETIVO:\n'
                                   'Convertir la petición del usuario en una consulta SQL válida.\n\n'
                                   
                                   f'## ESTRUCTURA DE TABLA (COLUMNAS Y TIPOS DE DATOS):\n{schema}\n\n'

                                   f'## SOLICITUD DE USUARIO:\n{question_sql_auto}\n\n'
                                   
                                   '## INSTRUCCIONES OBLIGATORIAS:\n'
                                   '1. La tabla se llama exactamente members.\n'
                                   '2. Todas las variables (columnas) deben ir entre comillas dobles "".\n'
                                   '3. Usa comillas simples para valores de texto.\n'
                                   '4. Usa únicamente las columnas proporcionadas.\n\n'

                                   '## FORMATO DE RESPUESTA:\n'
                                   '- Devuelve sólo el código SQL, sin texto adicional.'
                                  )

                with st.spinner(f'{LLM_MODEL} está procesando tu consulta...'):
                    # Llamada al LLM
                    response_sql_auto = client.models.generate_content(model=LLM_MODEL, contents=prompt_sql_auto)
        
                    # Conversión de consulta en NL a SQL
                    sql_generated = response_sql_auto.text                                         # Obtención del texto de la respuesta
                    sql_generated = sql_generated.strip()                                          # Eliminación de espacios en blanco al principio y al final
                    sql_generated = sql_generated.replace('```sql', '').replace('```', '').strip() # Eliminación de bloques de código Markdown
            
                    try:    
                        # Ejecución de query
                        result_sql_auto = ps.sqldf(sql_generated, {'members': df})
                        st.session_state.cache_sql_auto[question_sql_auto] = {'result_sql_auto': result_sql_auto, 'sql_generated': sql_generated}
        
                    except Exception as e_exec:
                        # Mensaje de error de ejecución
                        st.error(f'Error al ejecutar la query generada: {e_exec}')

            except Exception as e_gen:
                # Mensaje de error de generación
                st.error(f'Error al generar la query: {e_gen}')

if question_sql_auto in st.session_state.cache_sql_auto:
    # Mostrar resultados
    st.success('Consulta ejecutada correctamente')
    st.dataframe(st.session_state.cache_sql_auto[question_sql_auto]['result_sql_auto'])
    
    # Mostrar query
    st.info('Consulta ejecutada:')
    st.code(st.session_state.cache_sql_auto[question_sql_auto]['sql_generated'], language='sql') 

# %%
# ================================================================================
# 3 - SISTEMA DE ANALÍTIA DE NEGOCIO INTELIGENTE
# ================================================================================

# Encabezado de sección
st.header('🧠 N3. Sistema de Analítica de Negocio Inteligente', divider='gray')

# Variables para construir prompts

# Muestra de datos
data_sample = df.sample(10).to_markdown(index=False)

# Estructura de DataFrame
schema = df.dtypes.reset_index().rename(columns={'index': 'column', 0: 'dtype'}).to_markdown(index=False)

# Valores únicos de variables categóricas
str_stats = df.select_dtypes(exclude='number').drop(columns=['month', 'signup_date']).describe().T.drop(columns='count')
str_stats['values'] = [df[col].unique()[:5].tolist() for col in str_stats.index]
str_stats['nulls'] = df[str_stats.index].isnull().sum()
str_stats = str_stats.reset_index().rename(columns={'index': 'column'}).to_markdown(index=False)

# Estadísticas básicas de variables numéricas
num_cols = df.select_dtypes(include='number').drop(columns=['month_num', 'quarter', 'year'], errors='ignore').columns
num_stats = df[num_cols].describe().T
num_stats['nulls'] = df[num_cols].isnull().sum()
num_stats = num_stats.reset_index().rename(columns={'index': 'column'}).to_markdown(index=False)

# Rangos de fechas
date_ranges = df[['month', 'signup_date']].agg(['min', 'max']).T
date_ranges['nulls'] = df[date_ranges.index].isnull().sum()
date_ranges = date_ranges.reset_index().rename(columns={'index': 'column'}).to_markdown(index=False)

# Ruta del glosario de negocio
BASE_DIR = Path(__file__).parent
GLOSSARY = BASE_DIR / 'data' / 'fitlife_glossary.csv'

# Función cacheada para leer el glosario de negocio
@st.cache_data
def read_glossary(path):

    glossary = pd.read_csv(path)
    return glossary

try:
    # Lectura cacheada de glosario de negocio
    data_glossary = read_glossary(GLOSSARY)

except Exception as e:
    # Mensaje de error
    st.error(f'Error al leer el glosario: {e}')

# Glosario de negocio
business_glossary = data_glossary.to_markdown(index=False)

# %%
# ================================================================================
# 3.1 - EJECUCIÓN DE CÓDIGO DE PYTHON
# ================================================================================

# Subtítulo
st.subheader('Editor manual de código con Python')
# Resumen de sección
st.markdown('Aquí puedes ejecutar código con Pandas para procesar el DataFrame **`df`**. '
            'El resultado final debe almacenarse en la variable **`result`** para poder mostrarlo o graficarlo.\n\n'
                    
            "**Ejemplo:**\n"
            "```python\n"
            "cond = df['month'] == df['month'].max()\n\n"
            "top_members = df[cond]\\\n"
            "    .groupby('member_id').agg({'visits_this_month':'sum'})\\\n"
            "    .sort_values('visits_this_month', ascending=False)\\\n"
            "    .head(10)\n\n"
            "result = top_members\n"
            "```")

# Formulario con caja de texto y botón
with st.form(key='form_py_editor', border=False):  
    code = st.text_area(placeholder='Escribe código con Python para ejecutar sobre el DataFrame', height=125, key='input_py_editor', label='', label_visibility='collapsed')
    btn_py_editor = st.form_submit_button('Ejecutar código')

# Ejecutar código de Python
if btn_py_editor and code:
    if code not in st.session_state.cache_py_editor:
        try:
            # Variables locales
            local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'px': px}
    
            # Ejecución de todo el código
            exec(code, {'__builtins__': __builtins__}, local_vars)
            # {} es el diccionario para que se no tenga acceso a las variables del script
            # local_vars es el diccionario donde se guardan todas las variables creadas o modificadas durante la ejecución
    
            # Recuperar variable 'result'
            result = local_vars.get('result', None)

            # Almacenar resultado
            if result is not None:
                st.session_state.cache_py_editor[code] = result
            
            # Mostrar que no se ha encontrado 'result'    
            else:
                st.warning("No se encontró la variable 'result'. Asegúrate de definirla.")
    
        except Exception as e:
            # Mensaje de error
            st.error(f'Error al ejecutar el código: {e}')

if code in st.session_state.cache_py_editor:
    # Resultado de código
    result = st.session_state.cache_py_editor[code]

    # Mostrar resultados númericos        
    if isinstance(result, (int, float, np.integer, np.floating)):
        st.success('Código ejecutado correctamente')
        st.write(result)

    # Mostrar DataFrame o Series
    elif isinstance(result, (pd.DataFrame, pd.Series)):
        st.success('Código ejecutado correctamente')
        st.dataframe(result)

    # Mostrar gráfico de Seaborn o Matplotlib
    elif hasattr(result, 'plot') or isinstance(result, plt.Axes):
        st.success('Gráfico generado correctamente')
        st.pyplot(result.figure if hasattr(result, 'figure') else plt.gcf())
    
    else:
        st.success('Código ejecutado correctamente')
        st.write(result)

# %%
# ================================================================================
# 3.2 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A CÓDIGO
# ================================================================================

# Subtítulo
st.subheader('Generador de código básico')
# Resumen de sección
st.markdown('Convierte tus instrucciones en lenguaje natural en código Python con Pandas que procesa el dataset. '
            'El código generado aplica filtrado de filas, agregaciones y ordenamiento según tu descripción.')

# Formulario con caja de texto y botón
with st.form(key='form_py_code1', border=False):
    # Pregunta de usuario
    question_py_code1 = st.text_input(placeholder='Escribe una consulta en lenguaje natural para convertirla en código', key='input_py_code1', label='', label_visibility='collapsed')
    btn_py_code1 = st.form_submit_button('Generar código Python')

# Convertir pregunta a código de Python
if btn_py_code1 and question_py_code1:
    if question_py_code1 not in st.session_state.cache_py_code1:
        try:
            # Prompt
            prompt_py_code1 = ('Actúa como un experto en Data Analysis especializado en Python y Pandas.\n\n'

                               '## OBJETIVO:\n'
                               'Generar código Python ejecutable para resolver la petición del usuario.\n\n'

                               f'## ESTRUCTURA DE TABLA:\n{schema}\n\n'

                               f'## SOLICITUD DE USUARIO:\n{question_py_code1}\n\n'
                               
                               '## INSTRUCCIONES OBLIGATORIAS:\n'
                               '1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.\n'
                               '2. Usa únicamente Pandas para manipulación. No incluyas visualizaciones ni gráficos.\n'
                               '3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n'
                               '4. El resultado definitivo debe asignarse a la variable \'result\'.\n\n'
                               
                               '## FORMATO DE RESPUESTA:\n'
                               '- Devuelve sólo el código Python, sin texto adicional.'
                              )

            with st.spinner(f'{LLM_MODEL} está procesando tu consulta...'):
                # Llamada al LLM
                response_py_code1 = client.models.generate_content(model=LLM_MODEL, contents=prompt_py_code1)
                st.session_state.cache_py_code1[question_py_code1] = response_py_code1.text
                
        except Exception as e:
            # Mensaje de error
            st.error(f'Error al generar el código: {e}')

if question_py_code1 in st.session_state.cache_py_code1:   
    # Respuesta del LLM
    st.success('Código generado')
    st.write(st.session_state.cache_py_code2[question_py_code1])

# %%
# ================================================================================
# 3.3 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A CÓDIGO (CON VISUALIZACIÓN)
# ================================================================================

# Subtítulo
st.subheader('Generador de código avanzado')
#Resumen de sección
st.markdown('Genera código Python más completo, incluyendo visualización de datos, para analizar el negocio de FitLife. '
            'El LLM cuenta con información sobre estadísticas de las columnas, valores permitidos por categoría y rangos de fechas, lo que permite crear un código preciso.')

# Formulario con caja de texto y botón
with st.form(key='form_py_code2', border=False):
    # Pregunta de usuario
    question_py_code2 = st.text_input(placeholder='Escribe una consulta en lenguaje natural para convertirla en código avanzado', key='input_py_code2', label='', label_visibility='collapsed')
    btn_py_code2 = st.form_submit_button('Generar código Python')

# Convertir pregunta a código de Python
if btn_py_code2 and question_py_code2:
    if question_py_code2 not in st.session_state.cache_py_code2:
        try:
            # Prompt
            prompt_py_code2 = ('Actúa como un experto en Data Science especializado en Python, Pandas y Seaborn.\n\n'

                               '## OBJETIVO:\n'
                               'Generar código Python ejecutable para resolver la petición del usuario basada en el contexto proporcionado.\n\n'

                               '## CONTEXTO ANALÍTICO:\n'
                               f'### ESTRUCTURA DE TABLAS:\n{schema}\n\n'
                               f'### ESTADÍSTICAS DE COLUMNAS CATEGÓRICAS:\n{str_stats}\n\n'
                               f'### ESTADÍSTICAS DE COLUMNAS NUMÉRICAS:\n{num_stats}\n\n'
                               f'### RANGO DE FECHAS:\n{date_ranges}\n\n'

                               f'## SOLICITUD DE USUARIO:\n{question_py_code2}\n\n'
                           
                               '## INSTRUCCIONES OBLIGATORIAS:\n'
                               '1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.\n'
                               '2. Usa únicamente Pandas para manipulación y Seaborn para visualización.\n'
                               '3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n'
                               '4. El resultado definitivo debe asignarse a la variable \'result\'.\n\n'

                               '## FORMATO DE RESPUESTA:\n'
                               '- Devuelve sólo el código Python, sin texto adicional.'
                              )

            with st.spinner(f'{LLM_MODEL} está procesando tu consulta...'):
                # Llamada al LLM
                response_py_code2 = client.models.generate_content(model=LLM_MODEL, contents=prompt_py_code2)
                st.session_state.cache_py_code2[question_py_code2] = response_py_code2.text
                
        except Exception as e:
            # Mensaje de error
            st.error(f'Error al generar el código: {e}')

if question_py_code2 in st.session_state.cache_py_code2:   
    # Respuesta del LLM
    st.success('Código generado')
    st.write(st.session_state.cache_py_code2[question_py_code2])

# %%
# ================================================================================
# 3.4 - CARGA DE GLOSARIO DE NEGOCIO
# ================================================================================

# Subtítulo
st.subheader('Generador de código con conocimiento de negocio')
# Resumen de sección
st.markdown('En esta sección lleva tu análisis al siguiente nivel integrando el conocimiento de negocio de FitLife. '
            'El modelo utiliza un glosario de términos específico de gimnasios y reglas de la cadena para generar código Python.')

# Formulario con caja de texto y botón
with st.form(key='form_py_code3', border=False):
    # Pregunta de usuario
    question_py_code3 = st.text_input(placeholder='Escribe una consulta en lenguaje natural para convertirla en código contextualizado con negocio', key='input_py_code3', label='', label_visibility='collapsed')
    btn_py_code3 = st.form_submit_button('Generar código Python')

# Convertir pregunta a código de Python
if btn_py_code3 and question_py_code3:
    if question_py_code3 not in st.session_state.cache_py_code3:
        try:
            # Prompt
            prompt_py_code3 = ('Actúa como un experto en Data Science especializado en Python, Pandas y Seaborn.\n\n'

                               '## OBJETIVO:\n'
                               'Generar código Python ejecutable para resolver la petición del usuario basada en el contexto proporcionado.\n\n'

                               '## CONTEXTO ANALÍTICO:\n'
                               f'### GLOSARIO DE NEGOCIO:\n{business_glossary}\n\n'
                               f'### ESTRUCTURA DE TABLAS:\n{schema}\n\n'
                               f'### ESTADÍSTICAS DE COLUMNAS CATEGÓRICAS:\n{str_stats}\n\n'
                               f'### ESTADÍSTICAS DE COLUMNAS NUMÉRICAS:\n{num_stats}\n\n'
                               f'### RANGO DE FECHAS:\n{date_ranges}\n\n'

                               f'## SOLICITUD DE USUARIO:\n{question_py_code3}\n\n'
                    
                               '## INSTRUCCIONES OBLIGATORIAS:\n'
                               '1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.\n'
                               '2. Usa únicamente Pandas para manipulación y Seaborn para visualización.\n'
                               '3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n'
                               '4. El resultado definitivo debe asignarse a la variable \'result\'.\n\n'

                               '## FORMATO DE RESPUESTA:\n'
                               '- Devuelve sólo el código Python, sin texto adicional.'
                              )

            with st.spinner(f'{LLM_MODEL} está procesando tu consulta...'):
                # Llamada al LLM
                response_py_code3 = client.models.generate_content(model=LLM_MODEL, contents=prompt_py_code3)
                st.session_state.cache_py_code3[question_py_code3] = response_py_code3.text
                
        except Exception as e:
            # Mensaje de error
            st.error(f'Error al generar el código: {e}')

if question_py_code3 in st.session_state.cache_py_code3:   
    # Respuesta del LLM
    st.success('Código generado')
    st.write(st.session_state.cache_py_code3[question_py_code3])

# %%
# ================================================================================
# 4 - SISTEMA AUTÓNOMO DE ACCIONABILIDAD
# ================================================================================

# Encabezado de sección
st.header('⚡ N4. Sistema Autónomo de Accionabilidad', divider='gray')

# %%
# ================================================================================
# 4.1 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A CÓDIGO + EJECUCIÓN
# ================================================================================

# Subtítulo
st.subheader('Motor de código con lenguaje natural')
# Resumen de sección
st.markdown('Pasa de la pregunta al resultado en un sólo paso. '
            'Escribe lo que quieres analizar, y el sistema se encarga de generar, ejecutar y mostrar/visualizar los resultados automáticamente.')
# Flujo de sección
#st.info('Pregunta \u00A0 ➔ \u00A0 Procesamiento de LLM \u00A0 ➔ \u00A0 Generación de código Python \u00A0 ➔ \u00A0 Ejecución de código \u00A0 ➔ \u00A0 Visualización de resultados',
#        icon='⚡')

# Formulario con caja de texto y botón
with st.form(key='form_py_auto', border=False):
    # Pregunta de usuario
    question_py_auto = st.text_input(placeholder='Escribe una pregunta en el motor de código automático', key='input_py_auto', label='', label_visibility='collapsed')
    btn_py_auto = st.form_submit_button('Generar y ejecutar código Python')

# Convertir pregunta en código y ejecutarlo
if btn_py_auto and question_py_auto:
    if question_py_auto not in st.session_state.cache_py_auto:
        try:
            # Prompt
            prompt_py_auto = ('Actúa como un experto en Data Science especializado en Python, Pandas y Plotly.\n\n'
            
                              '## OBJETIVO:\n'
                              'Generar código Python ejecutable para resolver la consulta del usuario basada en el contexto proporcionado.\n\n'
                              
                              '## CONTEXTO ANALÍTICO:\n'
                              f'### GLOSARIO DE NEGOCIO:\n{business_glossary}\n\n'
                              f'### ESTRUCTURA DE TABLAS:\n{schema}\n\n'
                              f'### ESTADÍSTICAS DE COLUMNAS CATEGÓRICAS:\n{str_stats}\n\n'
                              f'### ESTADÍSTICAS DE COLUMNAS NUMÉRICAS:\n{num_stats}\n\n'
                              f'### RANGO DE FECHAS:\n{date_ranges}\n\n'

                              f'## SOLICITUD DE USUARIO:\n{question_py_auto}\n\n'
                       
                              '## INSTRUCCIONES OBLIGATORIAS:\n'
                              '1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.\n'
                              '2. Usa únicamente Pandas para manipulación y plotly.express (importado como px) para visualización.\n'
                              '3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n'
                              '4. El DataFrame, Series o valor numérico resultante debe asignarse a la variable \'result\'.\n'
                              '5. Si la consulta implica comparación, tendencia o evolución, genera una gráfico. En caso contrario, no visualices.\n'
                              '6. Si se genera una visualización, el objeto Figure o Axes debe asignarse a la variable \'fig\'.\n\n'

                              '## FORMATO DE RESPUESTA:\n'
                              '- Devuelve sólo el código Python, sin incluir comentarios ni explicaciones adicionales.'
                             )
        
            with st.spinner(f'{LLM_MODEL} está procesando tu consulta...'):
                # Llamada al LLM
                response_py_auto = client.models.generate_content(model=LLM_MODEL, contents=prompt_py_auto)
                
                # Conversión de consulta en NL a código
                py_generated = response_py_auto.text                                            # Obtención del texto de la respuesta
                py_generated = py_generated.strip()                                             # Eliminación de espacios en blanco al principio y al final
                py_generated = py_generated.replace('```python', '').replace('```', '').strip() # Eliminación de bloques de código Markdown

                try: 
                    # Diccionario de variables locales para ejecutar el código
                    local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns, 'px': px}
                    
                    # Ejecución de código generado
                    exec(py_generated, {'__builtins__': __builtins__}, local_vars)

                    st.session_state.cache_py_auto[question_py_auto] = {'py_generated': py_generated,
                                                                        'result': local_vars.get('result'),
                                                                        'fig': local_vars.get('fig')}
                    
                except Exception as e_exec:
                    # Mensaje de error de ejecución
                    st.error(f'Error al ejecutar el código generado: {e_exec}') 

        except Exception as e_gen:
            # Mensaje de error de generación
            st.error(f'Error al generar el código: {e_gen}')

if question_py_auto in st.session_state.cache_py_auto:
    # Resultados del LLM
    cache_py_auto = st.session_state.cache_py_auto[question_py_auto]
    py_generated = cache_py_auto['py_generated']
    result_auto = cache_py_auto['result']
    fig_auto = cache_py_auto['fig']
    
    # Mostrar el resultado según su tipo
    if fig_auto is not None:
        # Hay visualización
        if hasattr(fig_auto, 'plot') or isinstance(fig_auto, plt.Axes):
            st.success('Visualización generada')
            st.pyplot(fig_auto.figure if hasattr(fig_auto, 'figure') else plt.gcf())
        
        elif 'plotly.graph_objs' in str(type(fig_auto)):
            st.success('Gráfico generado correctamente')
            st.plotly_chart(fig_auto, width='stretch')

    # Hay DataFrame, Series o valor numérico
    else:
        if result_auto is None:
            st.warning("No se encontró la variable 'result'")
        
        elif isinstance(result_auto, (int, float, np.integer, np.floating)):
            st.success('Código ejecutado correctamente')
            st.write(result_auto)
            
        elif isinstance(result_auto, (pd.DataFrame, pd.Series)):
            st.success('Código ejecutado correctamente')
            st.dataframe(result_auto)
            
        else:
            st.success('Código ejecutado correctamente')
            st.write(result_auto)

    # Mostrar código
    st.info('Código ejecutado:')
    st.code(py_generated, language='python')

# %%
# ================================================================================
# 4.2 - GENERACIÓN DE INTERPRETACIONES/CONCLUSIONES DEL RESULTADO
# ================================================================================

# Encabezado
st.markdown('##### Interpretador de resultados')

# Interpretar resultados analíticos generados
with st.expander('Interpretación de IA', expanded=False):
    if question_py_auto in st.session_state.cache_py_auto:
        
        # Recuperar DataFrame, Series o valor numérico generado
        result_interpret = st.session_state.cache_py_auto[question_py_auto]['result']
    
        if result_interpret is not None:
                try:
                    # Prompt
                    prompt_interpret = ('Actúa como un Senior Data Storyteller experto en el sector fitness y gestión de gimnasios.\n\n'
                                        
                                        '## OBJETIVO:\n'
                                        'Interpretar los resultados analíticos y detectar hallazgos en lenguaje claro, explicando qué significan para la salud del gimnasio (FitLife).\n\n'
                                        
                                        '## CONTEXTO ANALÍTICO:\n'
                                        f'### GLOSARIO DE NEGOCIO:\n{business_glossary}\n\n'
                                        f'### ESTADÍSTICAS HISTÓRICAS GENERALES:\n{num_stats}\n\n'

                                        f'## RESULTADOS A INTERPRETAR: {result_interpret}\n\n'

                                        '## INSTRUCCIONES OBLIGATORIAS:\n'
                                        '1. Basa tu análisis exclusivamente en los datos proporcionados y el glosario.\n'
                                        '2. No menciones ni describas elementos técnicos (código, DataFrame tabla o columnas).\n'
                                        '3. El tono debe ser profesional, ejecutivo y orientado a la toma de decisiones.\n'
                                        '4. Si los datos son insuficientes para una conclusión sólida, indícalo y sugiere qué métrica adicional deberíamos consultar.\n\n'
                                        
                                        '## FORMATO DE RESPUESTA:\n'
                                        '- Empieza directamente con el contenido, sin introducciones.\n'
                                        '- Usa exclusivamente Markdown enriquecido (negritas, listas).\n'
                                        '- Respeta la siguiente estructura de tres bloques:\n\n'
                                        
                                        '1. 🎯 **VISIÓN EJECUTIVA**\n'
                                        '   - Crea un titular breve en negrita que resuma el hallazgo principal.\n'
                                        '   - Si hay una/s cifra/s clave/s, menciónala aquí.\n\n'
                                        '2. 💡 **INSIGHT DE NEGOCIO**\n'
                                        '   - Explica el por qué (causas, hallazgos ocultos, cruce de conocimiento datos) de estos datos en base a los datos.\n'
                                        '   - Identifica si es una anomalía, una tendencia positiva o un riesgo.\n'
                                        '   - Compara los valores si el resultado es una lista o tabla.\n\n'
                                        '3. 📈 **IMPACTO Y ACCIÓN**\n'
                                        '   - Explica cómo afecta esto a la rentabilidad de FitLife y experiencia de los miembros?\n'
                                        '   - Propón una acción concreta e inmediata para el equipo del gimnasio.'
                                       )
                                        
                    with st.spinner(f'{LLM_MODEL} está interpretando el resultado...'):
                        # Llamada al LLM
                        response_interpret = client.models.generate_content(model=LLM_MODEL, contents=prompt_interpret)
                
                        # Resultado del LLM
                        st.markdown(response_interpret.text)
                        # Resultado
                        #st.write(result_interpret)

                except Exception as e:
                    # Mensaje de error
                    st.error(f'Error al interpretar el resultado: {e}')

    else:
        st.write('👉 Escribe una pregunta en la sección anterior para obtener un interpretración.')

# %%
# ================================================================================
# 4.3 - PANEL DE ALERTAS Y ESTRATEGIA PRESCRIPTIVA
# ================================================================================

# Encabezado
st.markdown('##### Panel de alertas y estrategia prescriptiva')

# Definir plan estratégico tras resultados analíticos
with st.expander('Ruta estratégica de IA', expanded=False):
    if question_py_auto in st.session_state.cache_py_auto:
        
        # Recuperar DataFrame, Series o valor numérico generado
        result_strategic = st.session_state.cache_py_auto[question_py_auto]['result']
    
        if result_strategic is not None:
                try:
                    # Prompt
                    prompt_strategic = ('Actúa como el Director de Operaciones (COO) de la cadena de gimnasios FitLife.\n\n'
                                        
                                        '## OBJETIVO:\n'
                                        'Generar un plan de acción claro, directo y ejecutable basado en los resultados actuales y el contexto proporcionado.\n\n'
                                        
                                        '## CONTEXTO ANALÍTICO:\n'
                                        f'### GLOSARIO DE NEGOCIO:\n{business_glossary}\n\n'
                                        f'### ESTADÍSTICAS CATEGÓRICAS:\n{str_stats}\n\n'
                                        f'### ESTADÍSTICAS HISTÓRICAS GENERALES::\n{num_stats}\n\n'
                                        f'### RANGO DE FECHAS:\n{date_ranges}\n\n'

                                        f'## RESULTADOS A ANALIZAR: {result_strategic}\n\n'
                                        
                                        '## INSTRUCCIONES OBLIGATORIAS:\n'
                                        '1. No expliques los datos, prescribe soluciones. El análisis ya se hizo, ahora toca ejecutar.\n'
                                        '2. Usa verbos de acción imperativos.\n'
                                        '3. No menciones ni describas elementos técnicos (código, DataFrame tabla o columnas).\n'
                                        '4. El tono debe ser profesional, ejecutivo y orientado a la toma de decisiones.\n'
                                        '5. Si los datos son insuficientes para una clara, sugiere qué métrica adicional deberíamos analizar.\n\n'

                                        '## FORMATO DE RESPUESTA:\n'
                                        '- Empieza directamente con el contenido, sin introducciones.\n'
                                        '- Usa exclusivamente Markdown enriquecido (negritas, listas).\n'
                                        '- Respeta la siguiente estructura de tres bloques:\n\n'

                                        '🚨 **ALERTAS DE RIESGO**: Identifica desviaciones críticas o peligros en los datos actuales.\n'
                                        '⚡ **ACCIONES PRIORITARIAS**: Enumera 2-4 pasos exactos que el staff debe ejecutar HOY.\n'
                                        '🚀 **ESTRATEGIA A LARGO PLAZO**: Indica una recomendación para prevenir este problema en el futuro o para escalar el éxito detectado.'
                                        )
                                        
                    with st.spinner(f'{LLM_MODEL} está interpretando el resultado...'):
                        # Llamada al LLM
                        response_strategic = client.models.generate_content(model=LLM_MODEL, contents=prompt_interpret)
                
                        # Resultado del LLM
                        st.markdown(response_strategic.text)

                except Exception as e:
                    # Mensaje de error
                    st.error(f'Error al trazar el plan estratégico: {e}')

    else:
        st.write('👉 Escribe una pregunta en la sección anterior para activar el plan estratégico.')

# %%
# ================================================================================
# 4.4 - MODELO DE ML PARA PREDECIR CHURN
# ================================================================================

# Subtítulo
st.subheader('Predictor de *churn*')

# Función cacheada para entrenar modelo de ML
@st.cache_resource
def train_model():
 
    df_model = df.copy()
    
    # Preparación de variables

    # Ordenar cronológicamente por miembro
    df_model = df_model.sort_values(['member_id', 'month'])
        
    # Gestión de nulos
    df_model['satisfaction_survey'] = df_model['satisfaction_survey'].fillna(df_model['satisfaction_survey'].median())
    df_model = df_model.fillna(0)

    # Creación de variable dependiente
    df_model['target_churn'] = df_model.groupby('member_id')['status'].shift(-1).apply(lambda x: 1 if x == 'churned' else 0)
    
    # Cálculo de diferencia de precio respecto a la competencia
    df_model['price_diff'] = df_model['price_paid'] - df_model['competitor_lowcost_price']

    # Cálculo de diferencia de visitas vs. hace 1 mes
    df_model['visits_diff_1m'] = df_model.groupby('member_id')['visits_this_month'].diff(periods=1)
    
    # Eliminación de la última fila de cada socio
    df_model = df_model.dropna(subset=['target_churn', 'visits_diff_1m'])
    
    # Lista de variables independientes
    features = [
        'age',
        'price_diff',
        'tenure_months',
        'visits_this_month',
        'visits_diff_1m',
        'group_classes_attended',
        'support_contacts',
        'satisfaction_survey',
        'service_incident'
    ]
    
    # Conversión de variables categóricas
    X = pd.get_dummies(df_model[features], drop_first=True)
    # Variable dependiente
    y = df_model['target_churn']
    
    # División estratificada para mantener el % de bajas en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Cálculo de ratio de desequilibrio para scale_pos_weight
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Modelo XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4, 
        learning_rate=0.05,
        scale_pos_weight=ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Entrenamiento
    model.fit(X_train, y_train)
    
    # Predicción
    y_probs = model.predict_proba(X_test)[:, 1]
    # Ajuste de umbral para mayor agresividad en la predicción
    y_pred = (y_probs >= 0.3).astype(int)
    
    # Métricas de rendimiento
    roc_auc = roc_auc_score(y_test, y_probs).round(2)
    recall = recall_score(y_test, y_pred).round(2)
    
    # Importancia de variables
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return df_model, features, model, importances

# Entrenamiento cacheado de modelo de ML
df_model, features, model, importances = train_model()

# Últimos registros de cada miembros
df_last_status = df.drop_duplicates(subset='member_id', keep='last')

# Formulario con botón
with st.form(key='form_ml', border=False):

    # Entradas de modelo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write('###### 👤 Perfil')
        age = st.slider('Edad', int(df_last_status['age'].min()), int(df_last_status['age'].max()), int(df_last_status['age'].mean()))
        tenure = st.slider('Antigüedad (meses)', 0, df_last_status['tenure_months'].max(), 12)
        price_paid = st.number_input('Precio pagado (€)', value=df_last_status[df_last_status['plan'] == 'basic']['price_paid'].max())
    
    with col2:
        st.write('###### 🏋️ Actividad')
        visits_actual = st.number_input('Visitas este mes', 0, 31, 7)
        visits_prev = st.number_input('Visitas mes anterior', 0, 31, 10)
        group_classes = st.number_input('Clases colectivas', 0, 30, int(df_last_status['group_classes_attended'].mean()))

    with col3:
        st.write('###### 🛠️ Servicio')
        satisfaction = st.slider('Satisfacción (1-10)', 1, 10, int(df_last_status['satisfaction_survey'].mean()))
        support_contacts = st.number_input('Contactos soporte', 0, df_last_status['support_contacts'].max(), 0)
        incidents = st.selectbox('Incidente detectado', ['Ninguno', 'app_outage', 'heating_failure', 'pool_maintenance'])
    
    # Botón para predecir
    btn_ml = st.form_submit_button('Predecir')
    
# Predecir probabilidad de cancelación a partir de entradas
if btn_ml:

    # Cálculo de entradas derivadas
    price_diff = price_paid - df_last_status['competitor_lowcost_price'].min()
    visits_diff_1m = visits_actual - visits_prev
    
    # Inputs de modelo
    inputs = {'age': age,
              'price_diff': price_diff,
              'tenure_months': tenure,
              'visits_this_month': visits_actual,
              'visits_diff_1m': visits_diff_1m,
              'group_classes_attended': group_classes,
              'support_contacts': support_contacts,
              'satisfaction_survey': satisfaction
             }

    # Conversión de inputs a DataFrame para procesarlo
    df_inputs = pd.DataFrame([inputs])

    # One-Hot Encoding manual para los incidentes
    df_inputs['service_incident_app_outage'] = 1 if incidents == 'app_outage' else 0
    df_inputs['service_incident_heating_failure'] = 1 if incidents == 'heating_failure' else 0
    df_inputs['service_incident_pool_maintenance'] = 1 if incidents == 'pool_maintenance' else 0

    # Alineación de orden y cantidad de variables al modelo
    model_columns = model.get_booster().feature_names
    df_inputs = df_inputs.reindex(columns=model_columns)

    # Predicción y resultado
    prob = model.predict_proba(df_inputs)[0, 1]
    st.session_state.cache_ml = {'probability': prob,
                                 'inputs_member': inputs,
                                 'incidents': incidents,
                                 'feature_importance': importances.head(5).to_dict()
                                }

    # Mostrar resultados según el nivel de riesgo
    if prob >= 0.33:
        st.error(f'###### Riesgo de abandono: {prob:.1%}', icon='⚠️')
    else:
        st.success(f'###### Riesgo de abandono: {prob:.1%}', icon='✅')

# %%
# ================================================================================
# 4.5 - GENERACIÓN DE INTERPRETACIONES DE LA PREDICCION
# ================================================================================

# Encabezado
st.markdown('##### Explicador de predicción')

# Interpretar predicción de cancelación
with st.expander('Explicatividad de modelo', expanded=False):
    if 'probability' in st.session_state.cache_ml:

        # Recuperar predicción y variables
        result_ml = st.session_state.cache_ml

        probability = result_ml.get('probability', 0)
        inputs_member = result_ml.get('inputs_member', 'No disponibles')
        incidents = result_ml.get('incidents', 'Sin incidentes recientes')
        feature_importance = result_ml.get('feature_importance', {})
        feature_importance = '\n'.join([f'- **{k}**: {v:.1%}' for k, v in feature_importance.items()])
        
        try:
            # Prompt
            prompt_ml = ('Actúa como un Senior Growth Analyst experto en retención y abandono en el sector fitness.\n\n'

                         '## OBJETIVO:\n'
                         'Traducir la predicción técnica en un diagnóstico estratégico que identifique la causalidad del riesgo y proporcione claridad operativa.\n\n'
                         
                         '## PREDICCIÓN DEL MODELO:\n'
                         f'### PROBABILIDAD DE ABANDONO:\n{probability:.1%}\n\n'
                         f'### PERFIL DE MIEMBRO:\n{inputs_member}\n\n'
                         f'### INCIDENTES RECIENTES:\n{incidents}\n\n'
                         f'### IMPORTANCIA DE VARIABLES:\n{feature_importance}\n\n'

                         '## INSTRUCCIONES OBLIGATORIAS:\n'
                         '1. Basa tu análisis exclusivamente en las variables proporcionadas en la predicción.\n'
                         '2. No sólo listes los factores, explica cómo el factor X está impulsando la probabilidad de abandono en este miembro.\n'
                         "3. Si una variable tiene un peso bajo en 'Feature Importance', no le des protagonismo aunque parezca importante intuitivamente.\n"
                         '4. El tono debe ser profesional, ejecutivo y orientado a la toma de decisiones.\n\n'
                         
                         '## FORMATO DE RESPUESTA:\n'
                         '- Empieza directamente con el contenido, sin introducciones.\n'
                         '- Usa exclusivamente Markdown enriquecido (negritas, listas).\n'
                         '- Respeta la siguiente estructura de tres bloques:\n\n'

                         '🎯 **DIAGNÓSTICO DE PREDICCIÓN**: Resume de manera por qué es o no es probable que el miembro cancele su suscripción.\n'
                         '⚖️ **FACTORES QUE INFLUYEN**: Analiza las 5 variables con mayor peso en la predicción y cómo afectan a este miembro en particular.\n'
                         '🔍 **PATRONES DE ABANDONO**: Identifica patrones de comportamiento o señales de alerta temprana que expliquen la trayectoria del usuario.'
                        )

            with st.spinner(f'{LLM_MODEL} está interpretando la predicción...'):
                # Llamada al LLM
                response_ml = client.models.generate_content(model=LLM_MODEL, contents=prompt_ml)
                
                # Resultado del LLM
                st.markdown(response_ml.text)

        except Exception as e:
            # Mensaje de error
            st.error(f'Error al diagnosticar la predicción: {e}')
    else:
        st.write('👉 Realiza la predicción para diagnosticar los resultados.')
