"""
config.py
=========
Configuración central de DEPORTEData.

Para activar los modelos de IA reales tras el entrenamiento en EC2:
  1. Copia las carpetas entrenadas a models/
  2. Cambia USE_REAL_MODELS = True

Todo lo demás sigue funcionando igual.
"""

# ─── Modelos de IA ─────────────────────────────────────────────────────────
# Cambia a True cuando los modelos estén entrenados y disponibles en models/
USE_REAL_MODELS: bool = False

QWEN_MODEL_DIR:     str = "models/qwen2.5-7b-deporte"
TOXICITY_MODEL_DIR: str = "models/toxicity-classifier"

# Umbral de toxicidad (0.0-1.0). Más alto = menos falsos positivos.
TOXICITY_THRESHOLD: float = 0.7

# ─── Datos ─────────────────────────────────────────────────────────────────
PROCESSED_DIR: str = "data/processed"
FEDERADOS_PARQUET: str = f"{PROCESSED_DIR}/federados.parquet"
GASTO_PARQUET:     str = f"{PROCESSED_DIR}/gasto.parquet"
