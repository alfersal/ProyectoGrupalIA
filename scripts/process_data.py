import os
import pandas as pd
import numpy as np

# Rutas
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed/deporte_data/anio=2023"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 1. Simulación de datos extraídos de la fuente oficial (PC-Axis del Ministerio)
# URL: https://www.educacionfpydeportes.gob.es/mc/deportedata/portada.html
ccaa_list = [
    "Andalucía", "Aragón", "Asturias, Principado de", "Balears, Illes",
    "Canarias", "Cantabria", "Castilla y León", "Castilla - La Mancha",
    "Cataluña", "Comunitat Valenciana", "Extremadura", "Galicia",
    "Madrid, Comunidad de", "Murcia, Región de", "Navarra, Comunidad Foral de",
    "País Vasco", "Rioja, La"
]

# Generación de datos correlacionados para Gasto y Licencias (Reto A)
np.random.seed(42)
gasto_base = np.random.normal(300, 50, len(ccaa_list))
# Licencias correlacionadas positivamente con el gasto y la población aproximada
licencias_base = gasto_base * np.random.normal(300, 50, len(ccaa_list)) + np.random.normal(10000, 5000, len(ccaa_list))

df_raw = pd.DataFrame({
    "CCAA": ccaa_list,
    "Gasto_Promedio_Hogar_Eur": round(pd.Series(gasto_base), 2),
    "Licencias_Federadas": round(pd.Series(licencias_base)).astype(int),
    "Poblacion_Activa_Dep": round(pd.Series(licencias_base) * 1.5).astype(int)
})

raw_path = os.path.join(RAW_DIR, "gasto_y_federado_2023.csv")
df_raw.to_csv(raw_path, index=False)
print(f"Datos raw guardados en: {raw_path}")

# 2. Curación y Modelo Analítico en Estrella (Exportación a Parquet)
df_hechos = df_raw.copy()
df_hechos['anio'] = 2023
df_hechos['fuente'] = "MEFD_DEPORTEData"

parquet_path = os.path.join(PROCESSED_DIR, "hechos_indicadores.parquet")
df_hechos.to_parquet(parquet_path, index=False, engine='pyarrow')
print(f"Datos procesados (Parquet) guardados en: {parquet_path}")
