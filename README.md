# 🏆 Proyecto DEPORTEData - Reto A

**Equipo:** LosDelFondo
**Área del Repositorio:** Ingeniería de Datos y Arquitectura IA (RAG)

## 📝 Descripción del Reto Elegido
[cite_start]Este proyecto resuelve el **Reto A (Transversal)** planteado en DEPORTEData: *"¿Cómo ha evolucionado el gasto en deporte por hogar y su relación con la práctica deportiva federada por CCAA?"*[cite: 122, 123]. [cite_start]El objetivo es analizar patrones territoriales y visualizar hallazgos comparativos[cite: 130, 132].

## 🗂️ Inventario de Fuentes y Trazabilidad
[cite_start]Cumpliendo con las restricciones del proyecto, no se ha utilizado scraping web automatizado[cite: 66]. [cite_start]Los datos provienen exclusivamente de fuentes oficiales y descargables con trazabilidad[cite: 3, 78]:
* **Fuente Principal:** Plataforma DEPORTEData (Ministerio de Educación, Formación Profesional y Deportes).
* **Dataset 1 (Práctica Federada):** Magnitudes sectoriales -> Deporte Federado -> Licencias federadas por federación, periodo y comunidad autónoma.
* **Dataset 2 (Gasto en Deporte):** Magnitudes transversales -> Gasto de los hogares vinculado al deporte -> Gasto en bienes y servicios vinculados al deporte por comunidad autónoma.
* [cite_start]*Nota de Extracción:* Debido a la naturaleza dinámica de INEbase (generación de tablas mediante JavaScript sin enlaces estáticos directos), la ingesta cruda (`raw`) se ha realizado mediante descarga manual documentada, asegurando la trazabilidad de la Capa 1 y Capa 2[cite: 182, 190, 191].

## ⚙️ Pipeline de Datos y Curación (ETL)
[cite_start]La limpieza y modelado analítico se han desarrollado utilizando **Apache Spark (PySpark)**[cite: 90, 209]. [cite_start]El pipeline (`01_ingesta_y_curacion.py`) realiza las siguientes tareas de la Capa 3[cite: 193]:
1.  [cite_start]Control de codificaciones (ISO-8859-1 a UTF-8)[cite: 325].
2.  [cite_start]Tratamiento de nulos y filtrado de notas metodológicas en el pie de los CSV[cite: 196].
3.  [cite_start]Homogeneización de nombres territoriales (minúsculas, eliminación de tildes y espacios)[cite: 195, 324].
4.  [cite_start]Casteo dinámico de tipos (eliminación de separadores de miles y conversión a Integer/Double)[cite: 320].
5.  [cite_start]**Persistencia Analítica:** Guardado final en formato **Parquet particionado** (`/anio=/ccaa=/`) preparado para su almacenamiento en S3 (Capa 5)[cite: 41, 88, 206, 207].

## 🤖 Arquitectura Inteligente (Motor RAG)
[cite_start]Se ha implementado una interfaz de consulta inteligente basada en RAG (Retrieval-Augmented Generation) para responder preguntas sobre los indicadores en lenguaje natural[cite: 10, 49, 111]. [cite_start]El stack tecnológico de IA (`02_motor_rag.py`) está compuesto por[cite: 229, 230, 231]:

1.  **Filtro de Toxicidad:** Un pipeline clasificador utilizando el modelo `JonatanGk/roberta-base-bne-finetuned-hate-speech-offensive-spanish` para asegurar interacciones respetuosas.
2.  [cite_start]**Vector Store (Retrieval):** Implementación de `FAISS` para la indexación y búsqueda en memoria[cite: 230].
3.  [cite_start]**Embeddings:** Uso de `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` para la vectorización del corpus[cite: 231].
4.  **Generación (LLM):** Integración con modelo Instruct (`Qwen2.5-0.5B-Instruct` / Adaptable a API externa) para redactar respuestas contextualizadas a partir de los datos de FAISS.

## 🚀 Instrucciones de Uso y Ejecución
1. Clonar el repositorio.
2. [cite_start]Instalar las dependencias de IA: `pip install sentence-transformers faiss-cpu transformers` (Desarrollado sin GPU [cite: 76]).
3. **Capa de Datos:** Ejecutar el script `src/01_ingesta_y_curacion.py` para leer de `data/raw/` y generar los archivos Parquet en `data/processed/`.
4. **Capa IA:** Ejecutar `src/02_motor_rag.py` para inicializar el RAG y testear consultas en lenguaje natural.
