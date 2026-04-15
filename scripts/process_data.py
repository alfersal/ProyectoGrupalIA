import os
import glob
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina BOM y espacios extra en las columnas."""
    # Eliminar el caracter especial invisible (BOM) que suele colarse al leer con latin1
    df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]
    return df

def unify_directory_csvs(directory_pattern: str, output_parquet: str) -> bool:
    """Busca CSVs por glob pattern, los unifica y genera un Parquet."""
    files = glob.glob(directory_pattern)
    if not files:
        print(f"No se encontraron archivos para el patrón: {directory_pattern}")
        return False

    dfs = []
    for f in files:
        try:
            # latin1 e iso-8859-1 suelen manejar mejor los acentos en datos del gobierno español
            df = pd.read_csv(f, sep=';', encoding='latin1')
            df = fix_columns(df)
            
            # Convertir todas las columnas a string para evitar conflictos de esquemas en Parquet
            # Esto soluciona 'Conversion failed for column Total with type object' (mezcla float/string)
            df = df.astype(str)
            
            # Añadir trazabilidad
            df['archivo_origen'] = os.path.basename(f)
            dfs.append(df)
            print(f"Leído exitosamente: {os.path.basename(f)}")
        except Exception as e:
            print(f"Error procesando {f}: {e}")

    if dfs:
        # Concatenar ignorando el índice. Si hay columas distintas, insertará NaNs (comportamiento deseado)
        df_unificado = pd.concat(dfs, ignore_index=True)
        
        # Guardar en parquet
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        output_path = os.path.join(PROCESSED_DIR, output_parquet)
        df_unificado.to_parquet(output_path, engine="pyarrow", index=False)
        print(f" Guardado exitoso: {output_path} con {len(df_unificado)} registros y {len(df_unificado.columns)} columnas.\n")
        return True
    return False

def persist_datasets() -> tuple[str, str]:
    """Carga los CSVs raw y devuelve las rutas de los procesados."""
    federados_pattern = os.path.join(RAW_DIR, "Deporte_Federado", "*.csv")
    unify_directory_csvs(federados_pattern, "federados.parquet")
    
    gasto_pattern = os.path.join(RAW_DIR, "Gasto_*", "*.csv")
    unify_directory_csvs(gasto_pattern, "gasto.parquet")

    return os.path.join(PROCESSED_DIR, "federados.parquet"), os.path.join(PROCESSED_DIR, "gasto.parquet")

def main() -> None:
    print("Iniciando procesamiento de datos...")
    federados_path, gasto_path = persist_datasets()
    print("¡Procesamiento finalizado!")

if __name__ == "__main__":
    main()
