# LosDelFondo

Proyecto con una web/dashboard en Streamlit para explorar indicadores de gasto deportivo y licencias federadas.

## Enlace de la web

Aplicación desplegada: https://losdelfondo-frontend.onrender.com/

## Cómo lanzar la web en local

1. Crea un entorno virtual:

```bash
python -m venv .venv
```

2. Activa el entorno virtual en Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

4. Genera o actualiza los datos procesados:

```bash
python scripts/process_data.py
```

5. Lanza la aplicación en local:

```bash
streamlit run dashboard/app.py
```

6. Abre en el navegador la URL que muestre Streamlit, normalmente:

```text
http://localhost:8501
```
