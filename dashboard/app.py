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
    accent_color = "#A8DCAB"  # Verde neón
    sidebar_bg = "#262730"
    plotly_template = "plotly_dark"
    table_border = "#444444"
    chart_axis_color = "#CCCCCC" # Gris claro para ejes en oscuro
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    accent_color = "#0066CC"  # Azul cobalto
    sidebar_bg = "#F0F2F6"
    plotly_template = "plotly_white"
    table_border = "#DDDDDD"
    chart_axis_color = "#333333" # Gris oscuro para ejes

# Inyectar CSS Dinámico
st.markdown(f"""
<style>
    /* Aplicar colores base a toda la App */
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}

    /* Eliminar la franja negra superior */
    header, [data-testid="stHeader"] {{
        background-color: {bg_color} !important;
    }}
    
    footer {{
        visibility: hidden;
    }}
    
    /* Contenedor principal */
    .main .block-container {{
        padding-top: 1rem;
    }}
    
    /* Títulos con alto contraste */
    h1, h2, h3, h4, h5, h6, .stMetric label {{
        color: {accent_color} !important;
    }}
    
    /* Forzar color de texto en Markdown y Widgets */
    .stMarkdown, p, span, div, label, .stMetricValue {{
        color: {text_color} !important;
    }}
    
    /* Barra lateral */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: {text_color} !important;
    }}

    /* Estilos para Tablas Estáticas (st.table) */
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
    
    /* Pestañas (Tabs) */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {bg_color};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {text_color};
    }}
    
    /* Chat bubbles */
    .stChatMessage {{
        background-color: {sidebar_bg} !important;
        border-radius: 10px;
        margin-bottom: 10px;
    }}

    /* Estilo dinámico para los botones de Streamlit - Forzar color en el texto interno (p) */
    div.stButton > button p, [data-testid="stFormSubmitButton"] > button p {{
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
        xaxis=dict(
            gridcolor="rgba(128,128,128,0.2)", 
            zerolinecolor="rgba(128,128,128,0.3)", 
            tickfont=dict(color=text_color),
            title=dict(font=dict(color=text_color))
        ),
        yaxis=dict(
            gridcolor="rgba(128,128,128,0.2)", 
            zerolinecolor="rgba(128,128,128,0.3)", 
            tickfont=dict(color=text_color),
            title=dict(font=dict(color=text_color))
        ),
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
    st.header("Dashboard Analítico")
    col1, col2, col3 = st.columns(3)
    col1.metric("Gasto Promedio/Hogar", "€ 342", None)
    col2.metric("Licencias Federadas", "3.9 Millones", None)
    col3.metric("Comunidades Analizadas", "17", None)
    
    st.subheader("Evolución de Gasto vs Licencias (DEPORTEData 2023)")
    try:
        df_real = pd.read_parquet("data/processed/deporte_data/anio=2023/hechos_indicadores.parquet")
        df_display = df_real.rename(columns={'Gasto_Promedio_Hogar_Eur': 'Gasto Promedio Hogar Eur', 'Licencias_Federadas': 'Licencias Federadas'})
        
        fig_scatter = px.scatter(df_display, x="Gasto Promedio Hogar Eur", y="Licencias Federadas", hover_name="CCAA", color_discrete_sequence=[accent_color])
        st.plotly_chart(apply_plotly_style(fig_scatter), use_container_width=True)
        
        st.subheader("Gasto Promedio por Hogar por CCAA")
        fig_bar = px.bar(df_display, x="CCAA", y="Gasto Promedio Hogar Eur", color_discrete_sequence=[accent_color])
        st.plotly_chart(apply_plotly_style(fig_bar), use_container_width=True)

        st.subheader("Tabla de Indicadores Completos")
        df_table = df_display[['CCAA', 'Gasto Promedio Hogar Eur', 'Licencias Federadas']].copy()
        df_table.index = range(1, len(df_table) + 1)
        st.table(df_table)
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")

with tab2:
    st.header("Consulta Inteligente sobre DEPORTEData")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy el asistente IA de DEPORTEData. Puedes preguntarme sobre los datasets de gasto por hogar o licencias. ¿En qué puedo ayudarte?"}]
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
            prompt_lower = prompt.lower()
            try:
                df_rag = pd.read_parquet("data/processed/deporte_data/anio=2023/hechos_indicadores.parquet")
                df_rag = df_rag.rename(columns={'Gasto_Promedio_Hogar_Eur': 'Gasto Promedio Hogar Eur'})
                if "gasta más" in prompt_lower or "mayor gasto" in prompt_lower:
                    row = df_rag.loc[df_rag['Gasto Promedio Hogar Eur'].idxmax()]
                    assistant_response = f"🔍 La Comunidad que más gasta es **{row['CCAA']}**, con **{row['Gasto Promedio Hogar Eur']} €**."
                else:
                    assistant_response = "🧠 He analizado tu consulta. Observamos una correlación entre el gasto y la práctica deportiva. ¿Deseas detalles de alguna CCAA específica?"
            except Exception:
                assistant_response = "⚠️ Error al acceder a los datos."
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
            chart_data_usage = pd.DataFrame(np.random.randn(20, 2), columns=['Consultas IA', 'Visitas Dashboard'])
            fig_usage = px.line(chart_data_usage, color_discrete_sequence=[accent_color, "#FF4B4B"])
            st.plotly_chart(apply_plotly_style(fig_usage), use_container_width=True)
        with col_u2:
            st.metric("Consultas Totales", "2,840", None)
            fig_total = px.bar(np.random.randint(10, 100, size=(7, 1)), color_discrete_sequence=[accent_color])
            st.plotly_chart(apply_plotly_style(fig_total), use_container_width=True)
        st.divider()
        st.json({"status": "operativo", "version": "Alpha V1.4"})

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
    if not st.session_state.is_admin:
        if st.button("🔓 Login Admin"):
            st.session_state.show_login = True
            st.rerun()
    else:
        st.success("👤 Administrador")
        if st.button("🔒 Cerrar Sesión"):
            st.session_state.is_admin = False
            st.rerun()
