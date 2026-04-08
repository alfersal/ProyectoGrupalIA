import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px

# Inicialización de estado de sesión
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'Oscuro'
if 'sel_year' not in st.session_state:
    st.session_state.sel_year = '2023'
if 'sel_territory' not in st.session_state:
    st.session_state.sel_territory = 'Todas las CCAA'

# Configuración de la página
st.set_page_config(
    page_title="DEPORTEData | Reto A",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados dinámicos para Accesibilidad Total
if st.session_state.theme == 'Oscuro':
    bg_color = "#0E1117"
    text_color = "#FFFFFF"
    accent_color = "#00E676"  # Verde neón
    sidebar_bg = "#262730"
    plotly_template = "plotly_dark"
    table_border = "#444444"
    chart_axis_color = "#CCCCCC"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    accent_color = "#0066CC"  # Azul cobalto
    sidebar_bg = "#F0F2F6"
    plotly_template = "plotly_white"
    table_border = "#DDDDDD"
    chart_axis_color = "#333333"

# Inyectar CSS Dinámico
st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    header, [data-testid="stHeader"] {{
        background-color: {bg_color} !important;
    }}
    footer {{
        visibility: hidden;
    }}
    .main .block-container {{
        padding-top: 1rem;
    }}
    h1, h2, h3, h4, h5, h6, .stMetric label {{
        color: {accent_color} !important;
    }}
    .stMarkdown, p, span, div, label, .stMetricValue {{
        color: {text_color} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {text_color} !important;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        color: {text_color} !important;
        background-color: {bg_color} !important;
    }}
    th {{
        background-color: {sidebar_bg} !important;
        color: {accent_color} !important;
        border: 1px solid {table_border} !important;
        padding: 8px;
        text-align: left;
    }}
    td {{
        border: 1px solid {table_border} !important;
        padding: 8px;
        color: {text_color} !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {bg_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {text_color};
    }}
    .stChatMessage {{
        background-color: {sidebar_bg} !important;
        border-radius: 10px;
        margin-bottom: 10px;
    }}

    /* Estilo para los selectores (filtros) - Hacerlos 'claritos' en modo claro */
    div[data-baseweb="select"] > div {{
        background-color: {"#FFFFFF" if st.session_state.theme == "Claro" else "#1A1C23"} !important;
        color: {text_color} !important;
        border: 1px solid {table_border} !important;
    }}
    
    /* Asegurar color de texto en las etiquetas de los selectbox */
    .stSelectbox label p {{
        color: {text_color} !important;
    }}
    div.stButton > button, [data-testid="stFormSubmitButton"] > button p {{
        color: {"#000000" if st.session_state.theme == "Oscuro" else "#FFFFFF"} !important;
    }}
    div.stButton > button, [data-testid="stFormSubmitButton"] > button {{
        background-color: {accent_color} !important;
        border: none !important;
        border-radius: 5px !important;
        font-weight: bold !important;
    }}
</style>
""", unsafe_allow_html=True)

# Helper para configurar gráficos de Plotly
def apply_plotly_style(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=text_color,
        template=plotly_template,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)", tickfont=dict(color=chart_axis_color), title=dict(font=dict(color=text_color))),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zerolinecolor="rgba(128,128,128,0.3)", tickfont=dict(color=chart_axis_color), title=dict(font=dict(color=text_color))),
        legend=dict(font=dict(color=text_color))
    )
    return fig

# Título y encabezado
st.title("DEPORTEData: Analítica e Inteligencia Deportiva")
st.markdown("### Evolución del gasto en deporte por hogar y su relación con la práctica deportiva federada por CCAA")

# Pantalla de login
if st.session_state.show_login:
    st.header("🔑 Inicio de Sesión - Administrador")
    with st.form("login_form"):
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Entrar")
        if submit:
            if username == "admin" and password == "1234":
                st.session_state.is_admin = True
                st.session_state.show_login = False
                st.rerun()
            else:
                st.error("Credenciales incorrectas")
    if st.button("Cancelar"):
        st.session_state.show_login = False
        st.rerun()
    st.stop()

# Definición de pestañas
tabs_list = ["📊 Inicio", "🤖 Asistente IA (RAG)"]
if st.session_state.is_admin:
    tabs_list.append("⚙️ Admin")
tabs = st.tabs(tabs_list)
tab1, tab2 = tabs[0], tabs[1]
if st.session_state.is_admin:
    tab_admin = tabs[2]

with tab1:
    st.header(f"Dashboard Analítico - {st.session_state.sel_year}")
    try:
        # Cargar datos base
        df_real = pd.read_parquet(f"data/processed/deporte_data/anio={st.session_state.sel_year}/hechos_indicadores.parquet")
        df_display = df_real.rename(columns={'Gasto_Promedio_Hogar_Eur': 'Gasto Promedio Hogar Eur', 'Licencias_Federadas': 'Licencias Federadas'})
        
        # Aplicar filtro de Territorio
        if st.session_state.sel_territory != "Todas las CCAA":
            df_filtered = df_display[df_display['CCAA'] == st.session_state.sel_territory]
        else:
            df_filtered = df_display
            
        # Métricas Dinámicas
        col1, col2, col3 = st.columns(3)
        if not df_filtered.empty:
            avg_gasto = df_filtered['Gasto Promedio Hogar Eur'].mean()
            total_licencias = df_filtered['Licencias Federadas'].sum()
            num_ccaa = len(df_filtered)
            
            col1.metric("Gasto Promedio/Hogar", f"€ {avg_gasto:.0f}", None)
            col2.metric("Licencias Federadas", f"{total_licencias/1e6:.1f}M" if total_licencias > 1e5 else f"{total_licencias:,}", None)
            col3.metric("Áreas Analizadas", str(num_ccaa), None)
        
        st.subheader("Evolución de Gasto vs Licencias")
        fig_scatter = px.scatter(df_filtered, x="Gasto Promedio Hogar Eur", y="Licencias Federadas", hover_name="CCAA", color_discrete_sequence=[accent_color])
        st.plotly_chart(apply_plotly_style(fig_scatter), use_container_width=True)
        
        st.subheader("Gasto Promedio por Hogar por CCAA")
        fig_bar = px.bar(df_filtered, x="CCAA", y="Gasto Promedio Hogar Eur", color_discrete_sequence=[accent_color])
        st.plotly_chart(apply_plotly_style(fig_bar), use_container_width=True)

        st.subheader("Tabla de Indicadores")
        df_table = df_filtered[['CCAA', 'Gasto Promedio Hogar Eur', 'Licencias Federadas']].copy()
        df_table.index = range(1, len(df_table) + 1)
        st.table(df_table)
        
    except Exception as e:
        st.error(f"No se encontraron datos para los filtros seleccionados o el archivo no existe.")

with tab2:
    st.header("Consulta Inteligente sobre DEPORTEData")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy el asistente IA de DEPORTEData. ¿En qué puedo ayudarte?"}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                df_rag = pd.read_parquet("data/processed/deporte_data/anio=2023/hechos_indicadores.parquet")
                df_rag = df_rag.rename(columns={'Gasto_Promedio_Hogar_Eur': 'Gasto Promedio Hogar Eur'})
                if "gasta más" in prompt.lower():
                    row = df_rag.loc[df_rag['Gasto Promedio Hogar Eur'].idxmax()]
                    assistant_response = f"🔍 La CCAA que más gasta es {row['CCAA']} con {row['Gasto Promedio Hogar Eur']} €."
                else:
                    assistant_response = "🧠 He analizado los datos actuales. ¿Deseas algún detalle específico?"
            except Exception:
                assistant_response = "⚠️ No hay datos disponibles para el análisis."
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if st.session_state.is_admin:
    with tab_admin:
        st.header("Panel de Administración")
        st.subheader("📊 Panel de Uso")
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.metric("Usuarios Activos (24h)", "142", None)
            fig_usage = px.line(pd.DataFrame(np.random.randn(20, 2), columns=['Consultas IA', 'Visitas Dashboard']), color_discrete_sequence=[accent_color, "#FF4B4B"])
            st.plotly_chart(apply_plotly_style(fig_usage), use_container_width=True)
        with col_u2:
            st.metric("Consultas Totales", "2,840", None)
            fig_total = px.bar(np.random.randint(10, 100, size=(7, 1)), color_discrete_sequence=[accent_color])
            st.plotly_chart(apply_plotly_style(fig_total), use_container_width=True)
        st.divider()
        st.json({"status": "operativo", "version": "Alpha V1.5"})

# Sidebar y personalización
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5144/5144083.png", width=100)
    st.markdown("## DEPORTEData")
    st.divider()
    
    st.markdown("### Preferencias")
    theme = st.radio("Modo Visualización", ["Oscuro", "Claro"], index=0 if st.session_state.theme == "Oscuro" else 1, horizontal=True)
    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()

    st.divider()
    
    st.markdown("### Seleccionar Filtros Genéricos")
    # Filtro de Año persistente
    year_options = ["2023", "2022", "2021", "2020"]
    st.selectbox(
        "Año de Datos", 
        year_options, 
        index=year_options.index(st.session_state.sel_year),
        key='sel_year'
    )
    
    # Filtro de Territorio persistente
    territory_options = [
        "Todas las CCAA", 
        "Andalucía", "Aragón", "Asturias, Principado de", "Balears, Illes", 
        "Canarias", "Cantabria", "Castilla y León", "Castilla - La Mancha", 
        "Cataluña", "Comunitat Valenciana", "Extremadura", "Galicia", 
        "Madrid, Comunidad de", "Murcia, Región de", "Navarra, Comunidad Foral de", 
        "País Vasco", "Rioja, La"
    ]
    st.selectbox(
        "Territorio", 
        territory_options, 
        index=territory_options.index(st.session_state.sel_territory),
        key='sel_territory'
    )

    st.divider()
    
    if not st.session_state.is_admin:
        if st.button("🔓 Login Admin"):
            st.session_state.show_login = True
            st.rerun()
    else:
        st.success("👤 Administrador")
        if st.button("🔒 Cerrar Sesión"):
            st.session_state.is_admin = False
            st.rerun()
