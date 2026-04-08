import pandas as pd
import unicodedata
import random

def normalize(text):
    return "".join(c for c in unicodedata.normalize('NFD', text.lower()) if unicodedata.category(c) != 'Mn')

try:
    df = pd.read_parquet("data/processed/deporte_data/anio=2023/hechos_indicadores.parquet")
    df = df.rename(columns={'Gasto_Promedio_Hogar_Eur': 'Gasto Promedio Hogar Eur', 'Licencias_Federadas': 'Licencias Federadas'})

    prompts = ["CUANTO GASTA MADRID", "MADRID", "cuantos federados hay en valencia", "Y andalucia?", "gasta más"]

    aliases = {
        "andalucia": "Andalucía",
        "aragon": "Aragón",
        "asturias": "Asturias, Principado de",
        "baleares": "Balears, Illes",
        "balears": "Balears, Illes",
        "canarias": "Canarias",
        "cantabria": "Cantabria",
        "leon": "Castilla y León",
        "mancha": "Castilla - La Mancha",
        "cataluña": "Cataluña",
        "catalunya": "Cataluña",
        "catalonia": "Cataluña",
        "valencia": "Comunitat Valenciana",
        "valenciana": "Comunitat Valenciana",
        "extremadura": "Extremadura",
        "galicia": "Galicia",
        "madrid": "Madrid, Comunidad de",
        "murcia": "Murcia, Región de",
        "navarra": "Navarra, Comunidad Foral de",
        "vasco": "País Vasco",
        "rioja": "Rioja, La"
    }

    for prompt in prompts:
        print(f"\n--- Probando prompt: '{prompt}' ---")
        p_low = normalize(prompt)
        
        if any(x in p_low for x in ["gasta mas", "maximo gasto", "most spending", "highest spending", "mas dinero"]):
            row = df.loc[df['Gasto Promedio Hogar Eur'].idxmax()]
            print(f"Respuesta: Max spend - {row['CCAA']}")
        else:
            found = False
            for key, official_name in aliases.items():
                if key in p_low:
                    row = df[df['CCAA'] == official_name].iloc[0]
                    print(f"Respuesta: Region encontrada - {official_name} -> Gasto: {row['Gasto Promedio Hogar Eur']}")
                    found = True
                    break
            if not found:
                print("No encontrada, fallback")
                
except Exception as e:
    print(f"Error: {e}")
