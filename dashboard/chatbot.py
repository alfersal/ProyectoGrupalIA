"""Pure chatbot helpers used by the Streamlit app and tests."""

from __future__ import annotations

import unicodedata

import pandas as pd

ALIASES = {
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
    "rioja": "Rioja, La",
}

MAX_SPEND_PATTERNS = [
    "gasta mas",
    "maximo gasto",
    "most spending",
    "highest spending",
    "mas dinero",
]
MIN_SPEND_PATTERNS = [
    "gasta menos",
    "minimo gasto",
    "least spending",
    "lowest spending",
]
MAX_LICENSE_PATTERNS = [
    "mas licencias",
    "most licenses",
    "mas federados",
    "mas socios",
]


def normalize(text: str) -> str:
    """Return lowercase text without accents."""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(char) != "Mn"
    )


def prepare_assistant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names used by the chatbot."""
    return df.rename(
        columns={
            "Gasto_Promedio_Hogar_Eur": "Gasto Promedio Hogar Eur",
            "Licencias_Federadas": "Licencias Federadas",
        }
    )


def generate_chat_response(prompt: str, df: pd.DataFrame, labels: dict[str, str]) -> str:
    """Return a deterministic answer for the current prompt and dataset."""
    if df.empty:
        return labels["chat_error_data"]

    prompt_normalized = normalize(prompt)

    if any(pattern in prompt_normalized for pattern in MAX_SPEND_PATTERNS):
        row = df.loc[df["Gasto Promedio Hogar Eur"].idxmax()]
        return labels["chat_max_spend"].format(
            region=row["CCAA"],
            value=row["Gasto Promedio Hogar Eur"],
        )

    if any(pattern in prompt_normalized for pattern in MIN_SPEND_PATTERNS):
        row = df.loc[df["Gasto Promedio Hogar Eur"].idxmin()]
        return labels["chat_min_spend"].format(
            region=row["CCAA"],
            value=row["Gasto Promedio Hogar Eur"],
        )

    if any(pattern in prompt_normalized for pattern in MAX_LICENSE_PATTERNS):
        row = df.loc[df["Licencias Federadas"].idxmax()]
        return labels["chat_max_lic"].format(
            region=row["CCAA"],
            value=int(row["Licencias Federadas"]),
        )

    for alias, official_name in ALIASES.items():
        if alias in prompt_normalized:
            row = df[df["CCAA"] == official_name].iloc[0]
            return labels["chat_single_region"].format(
                region=official_name,
                gasto=row["Gasto Promedio Hogar Eur"],
                lic=int(row["Licencias Federadas"]),
            )

    fallback_row = df.sample(1, random_state=0).iloc[0]
    interesting_fact = labels["chat_single_region"].format(
        region=fallback_row["CCAA"],
        gasto=fallback_row["Gasto Promedio Hogar Eur"],
        lic=int(fallback_row["Licencias Federadas"]),
    )
    return f"{labels['chat_analyze']} {interesting_fact}"
