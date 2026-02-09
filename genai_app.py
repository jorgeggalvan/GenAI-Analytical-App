
# Importación de librerías
import streamlit as st
import pandas as pd
import pandasql as ps
import numpy as np
import plotly.express as px

from pathlib import Path
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

from google import genai
from groq import Groq

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
# 1 - SISTEMA DESCRIPTIVO ESTÁTICO
# ================================================================================

# Encabezado de sección
st.header('📊 N1. Sistema de Decisión Estático', divider='gray')

# %%
# ================================================================================
# SECCIÓN 1.1 - CARGA DE DATOS
# ================================================================================

# Subtítulo
st.subheader('Datos disponibles')

# Ruta del dataset
BASE_DIR = Path(__file__).parent
DATASET = BASE_DIR / 'data' / 'fitlife_members.csv'

# Mostrar dataset interactivo básico
try:
    # Lectura de dataset
    df = pd.read_csv(DATASET)

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
    st.error(f'Error al encontrar los datos: {e}') # Mostrar error si no existe el dataset

# %%
# ================================================================================
# MANIPULACIÓN DE DATOS
# ================================================================================

# Extracción de mes numérico de registro
df['month_num'] = pd.to_datetime(df['month'], format='%Y-%m').dt.month
# Extracción de trimestre de registro
df['quarter'] = pd.to_datetime(df['month'], format='%Y-%m').dt.quarter
# Extracción de año de registro
df['year'] = pd.to_datetime(df['month'], format='%Y-%m').dt.year

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
        
        df_filtered,              # Dataset                       
        use_container_width=True, # Máximo ancho disponible          
        hide_index=True,          # Ocultar el índice
        num_rows='dynamic',       # Permitir agregar/eliminar filas
        disabled=disabled_columns # Columnas no editables
    )

    # Actualización de DataFrame con los cambios realizados en el editor
    df_filtered = df_edited

except Exception as e:
    st.error(f'Error al mostrar los datos: {e}') # Mostrar mensaje de error

# %%
# ================================================================================
# SECCIÓN 1.3 - REPORTE DE PERFILADO
# ================================================================================

# Subtítulo
st.subheader('Perfilado de datos')

# Reporte de perfilado del DataFrame
# Desplegable
with st.expander('Reporte de perfilado', expanded=False):
    try:
        # Generación de informe de perfil
        profile = ProfileReport(df_filtered, explorative=True)
        #"""
        # Mostrar informe HTML dentro de Streamlit
        components.html(
            profile.to_html(),
            height=1200,
            scrolling=True
        )
        #"""
    
    except Exception as e:
        st.error(f'Error al generar el perfilado: {e}') # Mostrar mensaje de error

# %%
# ================================================================================
# 1.4 - DEFINICIÓN Y CÁLCULO DE KPIS
# ================================================================================

# Subtítulo
st.subheader('Indicadores de negocio')

# Filtrar por el trimestre actual y anterior
try:
    month_actual = months[1]
    quarter_actual = ((month_actual - 1) // 3) + 1
    year_actual = max(years)

    # Cálculo de trimestre y año anterior
    if quarter_actual == 1:
        quarter_prev = 4
        year_prev = year_actual - 1
    else:
        quarter_prev = quarter_actual - 1
        year_prev = year_actual
    
    # Filtrar por trimestre actual y trimestre anterior
    df_actual = df[(df['year'] == year_actual) & (df['quarter'] == quarter_actual)]
    df_prev = df[(df['year'] == year_prev) & (df['quarter'] == quarter_prev)]

except Exception as e:
    st.error(f'Error al filtrar datos del trimestre actual y anterior: {e}') # Mostrar mensaje de error

# Cálculo de KPI's
try:
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
    
except Exception as e:
    st.error(f'Error al calcular las métricas: {e}') # Mostrar mensaje de error

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

    met1.metric('👥 Tasa de Cancelación', f'{format_number(churn_rate)}%', f'{delta_churn_rate:.1f}%', delta_color='inverse')
    met2.metric('📈 Ingresos', f'{format_number(total_revenue)} €', f'{delta_revenue:.1f}%')
    met3.metric('⭐ CLV', f'{format_number(clv):.0f} meses', f'{delta_clv:.1f}%')
    met4.metric('⚡ Captación por Campañas', f'{format_number(campaign_singups):.0f}%', f'{delta_campaign:.1f}%')
    
except Exception as e:
    st.error(f'Error al visualizar los KPIs principales: {e}')  # Mostrar mensaje de error

# Mostrar métricas secundarias en columnas
try:
    # Creación de pestañas por categorías de KPIs
    tab1, tab2, tab3, tab4 = st.tabs(['👥 Miembros', '📈 Finanzas', '⭐ Fidelidad y experiencia', '⚡ Estrategia'])
    
    with tab1:
        met5, met6, met7, met8 = st.columns(4)
        met5.metric('👥 Cancelaciones', format_number(churned_members), f'{delta_churn:.1f}%', delta_color='inverse')
        met6.metric('👥 Nuevos Miembros', format_number(new_members), f'{delta_new:.1f}%')
        met7.metric('👥 Total de Miembros Activos', format_number(active_members), f'{delta_active:.1f}%')
        
    with tab2:
        met9, met10, met11, met12 = st.columns(4)
        met9.metric('📈 ARPU (ingreso promedio por miembro)', f'{format_number(arpu)} €', f'{delta_arpu:.1f}%')
        met10.metric('📈 LTV (valor de vida de miembro)', f'{format_number(ltv)} €', f'{delta_ltv:.1f}%')

    with tab3:
        met13, met14, met15, met16 = st.columns(4)
        met13.metric('⭐ NPS', format_number(nps), f'{delta_nps:.1f}%')
        met14.metric('⭐ Valor de Visita (ratio de visitas por ingresos)', f'{format_number(visit_value)} €', f'{delta_visit_value:.1f}%')
        met15.metric('⭐ Visitas por Miembro', f'{format_number(visits_per_member):.0f} visitas', f'{delta_visits:.0f}%')
        met16.metric('⭐ Incidencias Reportadas por Miembro', f'{format_number(incidents_per_member)} incidencias', f'{delta_incidents:.1f}%', delta_color='inverse')

    with tab4:
        met17, met18, met19, met20 = st.columns(4)
        met17.metric('⚡ Gap de Precio con Competencia', f'{format_number(price_gap)} €', f'{delta_gap:.1f}%', delta_color='off')
        met18.metric('⚡ Sensibilidad al Precio', f'{price_elasticity:.2f}')
        met19.metric('⚡ Miembros en Riesgo', f'{risk_members:.0f}%', f'{delta_risk_members:.1f}%', delta_color='inverse')

except Exception as e:
    st.error(f'Error al visualizar los KPIs secundarios: {e}')

# %%
# ================================================================================
# 1.5 - VISUALIZACIÓN DE DATOS
# ================================================================================

# Subtítulo
st.subheader('Visualización de datos')

# Creación de dos columnas
fig1, fig2 = st.columns(2)

# Encabezado
fig1.write('##### Evolución histórica de membresía')
fig1.caption('Este gráfico no está sujeto a los filtros aplicados.')

try:
    # Excluir cancelaciones
    df_active = df[df['status'] != 'churned']
    
    # Miembros activos por mes
    monthly_members = df_active.groupby('month').agg({'member_id':'nunique', 'competitor_lowcost_price':'mean'}).reset_index()
    
    # Gráfico de líneas
    fig_line1 = px.line(monthly_members, x='month', y='member_id',
                         # Etiquetas a los ejes x e y
                         labels = {'month':'Mes', 'member_id':'Miembros activos'}) 
    
    # Mostrar gráfico
    fig1.plotly_chart(fig_line1, use_container_width=True)
    
except Exception as e:
    fig1.error(f'Error al visualizar: {e}') # Mostrar mensaje de error

# Encabezado
fig2.write('##### Comparativa mensual de precios: plan básico vs. competencia')
fig2.caption('Este gráfico no está sujeto a los filtros aplicados.')

try:
    # Filtrar registros de plan básico
    df_active = df[df['plan'] == 'basic']
    
    # Precio medio de plan básico y competencia por mes
    monthly_price = df_active.groupby('month').agg({'price_paid':'mean', 'competitor_lowcost_price':'mean'}).reset_index()
    
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
    fig2.plotly_chart(fig_line2, use_container_width=True)
    
except Exception as e:
    fig2.error(f'Error al visualizar: {e}') # Mostrar mensaje de error

# Creación de tres columnas
fig3, fig4, fig5 = st.columns(3)

# Encabezado
fig3.write('##### Ranking de gimnasios FitLife')

try:
    # Ingresos por centro
    revenue_per_centre = df_filtered.groupby('center').agg({'price_paid':'sum'}).reset_index()

    # Gráfico de barras
    fig_bar = px.bar(revenue_per_centre, x='price_paid', y='center',
                     # Etiquetas a los ejes x e y
                     labels={'price_paid':'Ingresos obtenidos (€)', 'center':'Centro'},
                     # Orden de barras
                     category_orders={'center': revenue_per_centre.sort_values('price_paid', ascending=False)['center'].tolist()})

    # Mostrar gráfico
    fig3.plotly_chart(fig_bar, use_container_width=True)
    
except Exception as e:
    fig3.error(f'Error al visualizar: {e}') # Mostrar mensaje de error 

# Encabezado
fig4.write('##### Distribución de planes de suscripción')

try:
    # Últimos registros por cada 'member_id'
    df_current_plan = df_filtered[['month', 'member_id', 'plan', 'status']].sort_values(['member_id', 'month']).drop_duplicates(subset='member_id', keep='last')
    # Filtrar por miembros activos
    df_current_plan = df_current_plan[df_current_plan['status'] == 'active']
    
    # Número de miembros por plan
    plan_distribution = df_current_plan['plan'].value_counts().reset_index()

    # Gráfico de tarta
    fig_pie = px.pie(plan_distribution, names='plan', values='count')
    
    # Mostrar gráfico
    fig4.plotly_chart(fig_pie, use_container_width=True)
    
except Exception as e:
    fig4.error(f'Error al visualizar: {e}') # Mostrar mensaje de error

# Encabezado
fig5.write('##### Actividad mensual de miembros')
fig5.caption('Este gráfico no está sujeto a los filtros aplicados.')

try:
    # Visitas totales por año y mes
    monthly_visits = df.groupby(['year', 'month_num']).agg({'visits_this_month':'sum'}).reset_index()
    # Reestructuración de agregación a formato pivot
    monthly_visits_pivoted = monthly_visits.pivot(index='month_num', columns='year', values='visits_this_month')
    monthly_visits_pivoted = monthly_visits_pivoted.sort_index()
    
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
    fig5.plotly_chart(fig_heatmap, use_container_width=True)
    
except Exception as e:
    fig5.error(f'Error al visualizar: {e}') # Mostrar mensaje de error

# %%
# ================================================================================
# 2 - SISTEMA DINÁMICO Y CONVERSACIONAL
# ================================================================================

# Encabezado de sección
st.header('🗨️ N2. Sistema Dinámico y Conservacional', divider='gray')

# Variables para LLM
# --------------------------------------------------------------------------------
# API Key
GEMINI_API_KEY = st.secrets['GEMINI_API_KEY'] # Extraer clave desde https://aistudio.google.com/app/api-keys y pegarla en secrets.toml
GROQ_API_KEY = st.secrets['GROQ_API_KEY'] # Extraer clave desde https://console.groq.com/keys y pegarla en secrets.toml
# Selección de modelo
GEMINI_MODEL = 'gemini-2.5-flash'
GROQ_MODEL = 'llama-3.1-8b-instant'

# Variables para construir prompts
# --------------------------------------------------------------------------------
# Datos
data = df.to_string()

# Muestra de datos
data_sample = df.sample(10).to_string()

# Estructura de DataFrame
schema = df.dtypes.to_string()

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

# Subtítulo
st.subheader('Chat analítico')
# Resumen de sección
st.markdown('Utiliza este chat para realizar preguntas sobre FitLife. '
            'El modelo responderá basándose directa y únicamente en la información contenida en el dataset.')

# Inicialización de Gemini con la key directamente
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Opción con Groq
    #client = Groq(api_key=GROQ_API_KEY)
    
except Exception as e:
    st.error(f'Error con la clave de Gemini: {e}')

# Pregunta de usuario
question_llm = st.text_input(placeholder='Escribe una pregunta sobre los datos', label='', label_visibility='collapsed')

# Responder pregunta sobre los datos
try:
    if question_llm:
        # Prompt
        # Añadir enfoque con ejemplos few-shot
        prompt_llm = (f'Actúa como un experto en análisis de datos y en el sector de fitness.\n\n'
                      
                      f'DATOS DISPONIBLES:\n{data_sample}\n\n'
                      f'PREGUNTA:\n{question_llm}\n\n'
                      
                      'INSTRUCCIONES OBLIGATORIAS:\n'
                      '1. Basa tu respuesta únicamente en los datos proporcionados.\n'
                      '2. Sé conciso y claro; máximo una línea de explicación intermedia si es necesaria.\n'
                      '3. Devuelve sólo la información solicitada. Si es cálculo o resumen, muestra solo el resultado principal.\n'
                      '4. Evita suposiciones o datos inventados.'
                      )

        with st.spinner('Gemini está procesando tu pregunta...'):
            # Procesamiento del LLM
            response_llm = client.models.generate_content(model=GEMINI_MODEL,
                                                          #temperature=0.1,
                                                          contents=prompt_llm
                                                         )
            
            # Opción con Groq
            #response_llm = client.chat.completions.create(model=GROQ_MODEL, messages=[{'role': 'user', 'content': prompt_llm}])
            #response_llm.text = response_llm.choices[0].message.content
            
            # Respuesta del LLM
            st.success('Análisis completado')
            st.write(response_llm.text)
    
except Exception as e:
    st.error(f'Error al procesar la pregunta: {e}') # Mostrar mensaje de error

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

# Pregunta de usuario
querie = st.text_area(placeholder='Escribe una query', height=30, label='', label_visibility='collapsed')

# Botón para ejecutar querie
if st.button('Ejecutar query'):
    try:
        # Ejecución de querie
        result_sql = ps.sqldf(querie, {'members': df})
        
        st.success('Query ejecutada correctamente')
        st.dataframe(result_sql) # Mostrar resultados
        
    except Exception as e:
        st.error(f'Error al ejecutar la query: {e}') # Mostrar mensaje de error

# %%
# ================================================================================
# 2.4 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A SQL
# ================================================================================

# Subtítulo
st.subheader('Generador de consultas SQL')
# Resumen de sección
st.markdown('Convierte descripciones en lenguaje natural en consultas SQL listas para ejecutar. '
            'Escribe en palabras simples lo que necesitas y obtendrás la consulta preparada para copiar y ejecutar.')

# Pregunta de usuario
question_sql = st.text_input(placeholder='Escribe una consulta en lenguaje natural para convertirla en SQL', label='', label_visibility='collapsed')

# Convertir pregunta en consulta SQL
try:
    if question_sql:
        # Prompt para generar querie
        prompt_sql = (f'Convierte la siguiente descripción en una consulta SQL.\n\n' 
                      
                      f'DESCRIPCIÓN:\n{question_sql}\n\n'
                      f'ESTRUCTURA DE TABLA (COLUMNAS Y TIPOS DE DATOS):\n{schema}\n\n'
                      
                      'INSTRUCCIONES OBLIGATORIAS:\n'
                      '1. La tabla se llama exactamente \'members\'.\n'
                      '2. Todas las variables (columnas) deben ir entre comillas dobles \'\'.\n'
                      '3. Usa únicamente las columnas proporcionadas.\n'
                      '4. Devuelve sólo el código SQL, sin texto adicional.'
                     )

        with st.spinner('Gemini está procesando tu consulta...'):
            # Procesamiento del LLM
            response_sql = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_sql)
            
            # Respuesta del LLM
            st.success('Consulta generada')
            st.write(response_sql.text)

except Exception as e:
    st.error(f'Error al generar la query: {e}') # Mostrar mensaje de error

# %%
# ================================================================================
# 2.5 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A SQL + EJECUCIÓN
# ================================================================================

# Subtítulo
st.subheader('Motor de consultas SQL con lenguaje natural')
# Resumen de sección
st.markdown('Transforma tus preguntas en resultados estructurados al instante. '
            'El motor genera automáticamente la consulta SQL, la ejecuta y muestra la información correspondiente.')

# Pregunta de usuario
question_auto_sql = st.text_input(placeholder='Escribe una pregunta en el motor SQL automático', label='', label_visibility='collapsed') 
# El texto de '.text_input' debe ser siempre diferente a los anteriores

# Convertir pregunta en consulta SQL y ejecutarla
try:
    if question_auto_sql:
        # Prompt para generar y ejecutar querie
        prompt_auto_sql = (f'Convierte la siguiente descripción en una consulta SQL.\n\n'
                           
                           f'DESCRIPCIÓN:\n{question_auto_sql}\n\n'
                           f'ESTRUCTURA DE TABLA (COLUMNAS Y TIPOS DE DATOS):\n{schema}\n\n'
                           
                           'INSTRUCCIONES OBLIGATORIAS:\n'
                           '1. La tabla se llama exactamente \'members\'.\n'
                           '2. Todas las variables (columnas) deben ir entre comillas dobles \'\'.\n'
                           '3. Usa únicamente las columnas proporcionadas.\n'
                           '4. Devuelve sólo el código SQL, sin texto adicional.'
                          )

        with st.spinner('Gemini está procesando tu consulta...'):
            # Procesamiento del LLM
            response_auto_sql = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_auto_sql)

            # Conversión de consulta en NL a SQL
            sql_generated = response_auto_sql.text # Obtención del texto de la respuesta
            sql_generated = sql_generated.strip() # Eliminación de espacios en blanco al principio y al final
            sql_generated = sql_generated.replace('```sql', '').replace('```', '').strip() # Eliminación de bloques de código Markdown
        
            try:    
                # Ejecución de querie
                result_auto_sql = ps.sqldf(sql_generated, {'members': df})
    
                st.success('Consulta ejecutada correctamente')
                st.dataframe(result_auto_sql) # Mostrar resultados
    
                st.info('Consulta ejecutada:')
                st.code(sql_generated, language='sql') # Mostrar consulta

            except Exception as e_exec:
                st.error(f'Error al ejecutar la query generada: {e_exec}') # Mostrar mensaje de ejecución

except Exception as e_gen:
    st.error(f'Error al generar la query: {e_gen}') # Mostrar mensaje de error de generación

# %%
# ================================================================================
# 3 - SISTEMA MODELADO CON CONTEXTO DE NEGOCIO Y CONVERSACIONAL
# ================================================================================

# Encabezado de sección
st.header('🧠 N3. Sistema avanzado de analítica de negocio', divider='gray')

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

# Código
code = st.text_area(placeholder='Escribe código con Python para ejecutar sobre el DataFrame', height=125, label='', label_visibility='collapsed')

# Botón para ejecutar el código
if st.button('Ejecutar código'):
    try:
        # Variables locales
        local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns}

        # Ejecución de todo el código
        exec(code, {}, local_vars)
        # {} es el diccionario para que se no tenga acceso a las variables del script
        # local_vars es el diccionario donde se guardan todas las variables creadas o modificadas durante la ejecución

        # Mostrar el resultado (almacenado en una variable llamada 'result')
        result = local_vars.get('result', None)

        # Mostrar que no se ha encontrado 'result'        
        if result is None:
            st.warning("No se encontró la variable 'result'. Asegúrate de definirla.")

        # Mostrar resultados númericos        
        elif isinstance(result, (int, float, np.integer, np.floating)):
            st.success('Código ejecutado correctamente')
            st.write(result)

        # Mostrar DataFrame o Series
        elif isinstance(result, (pd.DataFrame, pd.Series)):
            st.success('Código ejecutado correctamente')
            st.dataframe(result)

        # Mostrar gráficos
        elif hasattr(result, 'plot') or isinstance(result, plt.Axes):
            st.success('Gráfico generado correctamente')
            st.pyplot(result.figure if hasattr(result, 'figure') else plt.gcf())
        
        else:
            st.success('Código ejecutado correctamente')
            st.write(result)

    except Exception as e:
        st.error(f'Error al ejecutar el código: {e}')

# %%
# ================================================================================
# 3.2 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A CÓDIGO
# ================================================================================

# Subtítulo
st.subheader('Generador de código básico')
# Resumen de sección
st.markdown('Convierte tus instrucciones en lenguaje natural en código Python con Pandas que procesa el dataset. '
            'El código generado aplica filtrado de filas, agregaciones y ordenamiento según tu descripción.')

# Pregunta de usuario
question_code_1 = st.text_input(placeholder='Escribe una consulta en lenguaje natural para convertirla en código', key='code_1', 
                                label='', label_visibility='collapsed')

# Convertir pregunta en código
try:
    if question_code_1:
        # Prompt para generar código
        prompt_code_1 = (f'Convierte la siguiente descripción en código Python usando Pandas.\n\n'
                         
                         f'DESCRIPCIÓN:\n{question_code_1}\n\n'
                         f'ESTRUCTURA DE TABLA:\n{schema}\n\n'
                                             
                         'INSTRUCCIONES OBLIGATORIAS:\n'
                         '1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.\n'
                         '2. Usa únicamente Pandas para manipulación. No incluyas visualizaciones ni gráficos.\n'
                         '3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n'
                         '4. El resultado definitivo debe asignarse a la variable \'result\'.\n'
                         '5. Devuelve sólo el código Python, sin texto adicional.'
                        )

        with st.spinner('Gemini está procesando tu consulta...'):
            # Procesamiento del LLM
            response_code_1 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_code_1)
            
            # Respuesta del LLM
            st.success('Código generado')
            st.write(response_code_1.text)

except Exception as e:
    st.error(f'Error al generar el código: {e}') # Mostrar mensaje de error

# %%
# ================================================================================
# 3.3 - CONVERSIÓN DE CONSULTAS EN LENGUAJE NATURAL A CÓDIGO (CON VISUALIZACIÓN)
# ================================================================================

# Subtítulo
st.subheader('Generador de código avanzado')
#Resumen de sección
st.markdown('Genera código Python más completo, incluyendo visualización de datos, para analizar el negocio de FitLife. '
            'El LLM cuenta con información sobre estadísticas de las columnas, valores permitidos por categoría y rangos de fechas, lo que permite crear un código preciso.')

# Pregunta de usuario
question_code_2 = st.text_input(placeholder='Escribe una consulta en lenguaje natural para convertirla en código avanzado', key='code_2', 
                                label='', label_visibility='collapsed')

# Convertir pregunta en código
try:
    if question_code_2:
        # Prompt para generar código
        prompt_code_2 = (f'Convierte la siguiente descripción en código Python usando Pandas.\n\n'
                         
                         f'DESCRIPCIÓN:\n{question_code_2}\n\n'
                         f'ESTRUCTURA DE TABLAS:\n{schema}\n\n'
                         f'ESTADÍSTICAS DE COLUMNAS CATEGÓRICAS:\n{str_stats}\n\n'
                         f'ESTADÍSTICAS DE COLUMNAS NUMÉRICAS:\n{num_stats}\n\n'
                         f'RANGO DE FECHAS:\n{date_ranges}\n\n'
                           
                         'INSTRUCCIONES OBLIGATORIAS:\n'
                         '1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.\n'
                         '2. Usa únicamente Pandas para manipulación y Seaborn para visualización.\n'
                         '3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n'
                         '4. El resultado definitivo (DataFrame, Series, valor numérico o Axes) debe asignarse a la variable \'result\'.\n'
                         '5. Devuelve sólo el código Python, sin texto adicional.'
                        )

        with st.spinner('Gemini está procesando tu consulta...'):
            # Procesamiento del LLM
            response_code_2 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_code_2)
            
            # Respuesta del LLM
            st.success('Código generado')
            st.write(response_code_2.text)

except Exception as e:
    st.error(f'Error al generar el código: {e}') # Mostrar mensaje de error

# %%
# ================================================================================
# 3.4 - CARGA DE GLOSARIO DE NEGOCIO
# ================================================================================

# Subtítulo
st.subheader('Generador de código con conocimiento de negocio')
# Resumen de sección
st.markdown('En esta sección lleva tu análisis al siguiente nivel integrando el conocimiento de negocio de FitLife. '
            'El modelo utiliza un glosario de términos específico de gimnasios y reglas de la cadena para generar código Python.')

# Pregunta de usuario
question_code_3 = st.text_input(placeholder='Escribe una consulta en lenguaje natural para convertirla en código contextualizado con negocio', key='code_3', 
                                label='', label_visibility='collapsed')

# Convertir pregunta en código
try:
    if question_code_3:
        # Prompt para generar código
        prompt_code_3 = (f'Convierte la siguiente descripción en código Python usando Pandas.\n\n'
                         
                         f'DESCRIPCIÓN:\n{question_code_3}\n\n'
                         f'ESTRUCTURA DE DATOS:\n{schema}\n\n'
                         f'ESTADÍSTICAS DE COLUMNAS CATEGÓRICAS:\n{str_stats}\n\n'
                         f'ESTADÍSTICAS DE COLUMNAS NUMÉRICAS:\n{num_stats}\n\n'
                         f'RANGO DE FECHAS:\n{date_ranges}\n\n'
                         f'GLOSARIO DE NEGOCIO:\n\n\n'
                           
                         'INSTRUCCIONES OBLIGATORIAS:\n'
                         '1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.\n'
                         '2. Usa únicamente Pandas para manipulación y Seaborn para visualización.\n'
                         '3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n'
                         '4. El resultado definitivo (DataFrame, Series, valor numérico o Axes) debe asignarse a la variable \'result\'.\n'
                         '5. Devuelve sólo el código Python, sin texto adicional.'
                        )

        with st.spinner('Gemini está procesando tu consulta...'):
            # Procesamiento del LLM
            response_code_3 = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_code_3)
            
            # Respuesta del LLM
            st.success('Código generado')
            st.write(response_code_3.text)

except Exception as e:
    st.error(f'Error al generar el código: {e}') # Mostrar mensaje de error

# %%
# ================================================================================
# 4 - SISTEMA DE DECISIÓN ACCIONABLE
# ================================================================================

# Encabezado de sección
st.header('⚡ N4. Sistema de decisión accionable', divider='gray')

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
st.info('Pregunta \u00A0 ➔ \u00A0 Procesamiento de LLM \u00A0 ➔ \u00A0 Generación de código Python \u00A0 ➔ \u00A0 Ejecución de código \u00A0 ➔ \u00A0 Visualización de resultados',
        icon='⚡')

# Pregunta de usuario
question_auto_code = st.text_input(placeholder='Escribe una pregunta en el motor de código automático',
                                   label='', label_visibility='collapsed')

# Convertir pregunta en código y ejecutarlo
try:
    if question_auto_code:
        # Prompt para generar y ejecutar código
        prompt_auto_code = (f'Convierte la siguiente descripción en código Python usando Pandas y Seaborn.\n\n'
                            
                            f'DESCRIPCIÓN:\n{question_auto_code}\n\n'
                            f'ESTRUCTURA DE TABLAS:\n{schema}\n\n'
                            f'ESTADÍSTICAS DE COLUMNAS CATEGÓRICAS:\n{str_stats}\n\n'
                            f'ESTADÍSTICAS DE COLUMNAS NUMÉRICAS:\n{num_stats}\n\n'
                            f'RANGO DE FECHAS:\n{date_ranges}\n\n'
                            
                            'INSTRUCCIONES OBLIGATORIAS:\n'
                            '1. El DataFrame se llama exactamente \'df\' y ya está cargado en memoria.\n'
                            '2. Usa únicamente Pandas para manipulación y Seaborn para visualización.\n'
                            '3. Aplica filtrado de filas, agregaciones con agg() y groupby() y ordenamiento con sort_values().\n'
                            '4. El DataFrame, Series o valor numérico resultante debe asignarse a la variable \'result\'.\n'
                            '5. Si la consulta implica comparación, tendencia o evolución, genera una gráfico. En caso contrario, no visualices.\n'
                            '6. Si se genera una visualización, el objeto Axes debe asignarse a la variable \'fig\'.\n'
                            '7. Devuelve sólo el código Python, sin incluir comentarios ni explicaciones adicionales.'
                           )
    
        with st.spinner('Gemini está procesando tu consulta...'):
            # Procesamiento del LLM
            response_auto_code = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_auto_code)
            
            # Conversión de consulta en NL a código
            code_generated = response_auto_code.text # Obtención del texto de la respuesta
            code_generated = code_generated.strip() # Eliminación de espacios en blanco al principio y al final
            code_generated = code_generated.replace('```python', '').replace('```', '').strip() # Eliminación de bloques de código Markdown
    
        # Diccionario de variables locales para ejecutar el código
        local_vars = {'df': df, 'pd': pd, 'np': np, 'plt': plt, 'sns': sns}
    
        try:
            # Ejecución de código generado
            exec(code_generated, {}, local_vars)
            
            # Mostrar el resultado (almacenado en una variable llamada 'result')        
            result_auto = local_vars.get('result', None)
            fig_auto = local_vars.get('fig', None)
            # Guardar en session state
            st.session_state['result_auto'] = result_auto
            st.session_state['fig_auto'] = fig_auto

            # Mostrar el resultado según su tipo
            if fig_auto is not None:
                # Hay visualización
                st.success('Gráfico generado correctamente')
                st.pyplot(fig_auto.figure)
            
            else:
                # Hay DataFrame, Series o valor numérico
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
            st.code(code_generated, language='python')
            
        except Exception as e_exec:
            st.error(f'Error al ejecutar el código generado: {e_exec}') # Mostrar mensaje de ejecución
    
except Exception as e_gen:
    st.error(f'Error al generar el código: {e_gen}') # Mostrar mensaje de error de generación

# %%
# ================================================================================
# 4.2 - GENERACIÓN DE INTERPRETACIONES/CONCLUSIONES DEL RESULTADO
# ================================================================================

# Subtítulo
st.subheader('Interpretador de resultados')

# Botón para ver la interpretación de la IA
with st.expander('Interpretación de IA', expanded=False):
    try:
        # Recuperar DataFrame, Series o valor numérico generado
        result_auto = st.session_state.get('result_auto', None)
        
        if result_auto is not None:
            
            with st.spinner('La IA está interpretando el resultado...'):
                
                # Prompt para interpretar resultados
                prompt_interpret = (f'Actúa como un experto en análisis de datos y en el sector fitness.\n\n'
                                    f'Interpreta el siguiente resultado y explica los hallazgos en lenguaje claro, orientado a negocio y toma de decisiones.\n\n'
                                    
                                    f'RESULTADO:{result_auto}\n\n'
                                    
                                    'INSTRUCCIONES OBLIGATORIAS:\n'
                                    '1. No describas el código.\n'
                                    '2. No repitas la tabla.\n'
                                    '3. Extrae conclusiones claras.')
        
                # Llamada al LLM
                response_interpret = client.models.generate_content(model=GEMINI_MODEL, contents=prompt_interpret)
        
                # Resultado del LLM
                st.markdown(response_interpret.text)
                # Resultado
                #st.write(result_auto)
                
    except Exception as e:
        st.error(f'Error al interpretar el resultado: {e}') # Mostrar mensaje de error
