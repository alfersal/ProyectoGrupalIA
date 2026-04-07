# LosDelFondo

Proyecto con una web/dashboard en Streamlit para explorar indicadores de gasto deportivo y licencias federadas.

## Enlace de la web

Aplicacion desplegada: https://losdelfondo-frontend.onrender.com/

## CI antes de desplegar en Render

Se ha anadido el workflow de GitHub Actions en [`.github/workflows/ci.yml`](/d:/LosDelFondo/.github/workflows/ci.yml) para validar el proyecto antes del despliegue.

Checks que ejecuta:

- instala dependencias
- valida que los archivos Python compilan
- regenera el dataset procesado
- comprueba que el Parquet existe y tiene la estructura esperada

Para que Render espere a estos checks antes de publicar:

1. Abre el servicio en Render.
2. Ve a `Settings`.
3. En `Build & Deploy`, cambia `Auto-Deploy` a `After CI Checks Pass`.
4. Asegurate de que el repositorio conectado es este mismo repositorio de GitHub.

## Como lanzar la web en local

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

5. Lanza la aplicacion en local:

```bash
streamlit run dashboard/app.py
```

6. Abre en el navegador la URL que muestre Streamlit, normalmente:

```text
http://localhost:8501
```
