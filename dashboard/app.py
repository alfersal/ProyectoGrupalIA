import streamlit as st
import time
import pandas as pd
import numpy as np

# Inicialización de estado de sesión
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = False

# Configuración de la página
st.set_page_config(
    page_title="DEPORTEData | Reto A",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados adicionales
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #00E676;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

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
                st.success("Acceso concedido")
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
tab1 = tabs[0]
tab2 = tabs[1]
if st.session_state.is_admin:
    tab_admin = tabs[2]

with tab1:
    st.header("Dashboard Analítico")
    st.info("💡 En esta sección se integran los cuadros de mando oficiales (Grafana / Metabase) o visualizaciones interactivas de Spark/Pandas.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Gasto Promedio/Hogar", "€ 342", "4%")
    col2.metric("Licencias Federadas", "3.9 Millones", "1.2%")
    col3.metric("Comunidades Analizadas", "17", "0")
    
    st.subheader("Evolución de Gasto vs Licencias (DEPORTEData 2023)")
    try:
        df_real = pd.read_parquet("data/processed/deporte_data/anio=2023/hechos_indicadores.parquet")
        st.scatter_chart(df_real, x="Gasto_Promedio_Hogar_Eur", y="Licencias_Federadas")
        st.dataframe(df_real[['CCAA', 'Gasto_Promedio_Hogar_Eur', 'Licencias_Federadas']].head(5))
    except Exception as e:
        st.error(f"No se pudo cargar el dataset procesado (Parquet). Ejecuta process_data.py primero.")


with tab2:
    st.header("Consulta Inteligente sobre DEPORTEData")
    st.write("Consulta al asistente virtual (RAG) sobre los datasets de deporte y los indicadores transversales.")
    
    # Inicializar el historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "¡Hola! Soy el asistente IA de DEPORTEData. Puedes preguntarme sobre los datasets de gasto por hogar, indicadores de licencias por CCAA o nuestro análisis interactivo. ¿En qué puedo ayudarte?"}
        ]

    # Mostrar mensajes del chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ingreso de usuario
    if prompt := st.chat_input("Escribe tu pregunta aquí... (ej. ¿Qué CCAA gasta más en deporte?)"):
        # Mostrar y guardar lo que el usuario escribió
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Respuesta Inteligente del "RAG" Local
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Lógica básica basada en Pandas simulando un LLM/FAISS determinista
            prompt_lower = prompt.lower()
            try:
                df_rag = pd.read_parquet("data/processed/deporte_data/anio=2023/hechos_indicadores.parquet")
                
                if "gasta más" in prompt_lower or "mayor gasto" in prompt_lower or "más gasta" in prompt_lower:
                    row = df_rag.loc[df_rag['Gasto_Promedio_Hogar_Eur'].idxmax()]
                    assistant_response = f"🔍 He consultado el data warehouse. La Comunidad Autónoma que más gasta en deporte en promedio por hogar es **{row['CCAA']}**, con un gasto medio de **{row['Gasto_Promedio_Hogar_Eur']} €** al año."
                elif "gasta menos" in prompt_lower or "menor gasto" in prompt_lower or "menos gasta" in prompt_lower:
                    row = df_rag.loc[df_rag['Gasto_Promedio_Hogar_Eur'].idxmin()]
                    assistant_response = f"🔍 La Comunidad Autónoma que menos gasta en deporte en promedio por hogar es **{row['CCAA']}**, con un gasto medio de **{row['Gasto_Promedio_Hogar_Eur']} €** al año."
                elif "licencias" in prompt_lower and "más" in prompt_lower:
                    row = df_rag.loc[df_rag['Licencias_Federadas'].idxmax()]
                    assistant_response = f"🏆 Según los datos procesados, **{row['CCAA']}** tiene el mayor nivel de práctica federada con **{row['Licencias_Federadas']} licencias** activas."
                elif "dataset" in prompt_lower or "consiste" in prompt_lower or ("que" in prompt_lower and "datos" in prompt_lower):
                    assistant_response = "📂 El dataset contiene información transversal del 'Reto A'. Integra datos del INE y CSD con métricas como el 'Gasto Promedio por Hogar (Eur)' y el número de 'Licencias Federadas' de las 17 CCAA de España para el año 2023."
                else:
                    assistant_response = f"🧠 Como tu asistente IA local, he analizado tu consulta sobre '{prompt}'. Basado en los indicadores transversales, observamos una fuerte correlación directa entre gasto y nivel de práctica federada a nivel nacional. Si necesitas detalles de alguna CCAA (ej. '¿Quién gasta más?'), no dudes en preguntarme."
            except Exception as e:
                assistant_response = "⚠️ No he podido acceder a los archivos Parquet procesados. Asegúrate de generar los datos primero."
            
            # Simular efecto de máquina de escribir
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if st.session_state.is_admin:
    with tab_admin:
        st.header("Panel de Administración")
        st.write("Bienvenido, Administrador. Aquí puedes gestionar la configuración del sistema y los datos.")
        
        st.subheader("Subida de nuevos datasets")
        uploaded_file = st.file_uploader("Elige un archivo .parquet o .csv", type=["parquet", "csv"])
        if uploaded_file is not None:
            st.success("Archivo subido correctamente (simulación)")

        st.subheader("Estado del Sistema")
        st.json({"status": "operativo", "last_sync": "2023-11-20 14:00:23", "version": "Alpha V1.1"})

# Sidebar y personalización
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5144/5144083.png", width=100)
    st.markdown("## DEPORTEData")
    st.markdown("**Proyecto:** Analítica Deportiva\n**Reto:** A - Gasto vs Práctica\n**Estado:** Alpha V1")
    
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

    st.divider()
    st.markdown("### Seleccionar Filtros Genéricos")
    st.selectbox("Año de Datos", ["2023", "2022", "2021", "2020"])
    st.selectbox("Territorio", ["Todas las CCAA", "Madrid", "Cataluña", "Andalucía", "Galicia", "País Vasco", "Comunidad Valenciana"])
