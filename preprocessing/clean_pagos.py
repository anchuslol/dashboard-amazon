import pandas as pd
import os
import unicodedata
import re

def limpiar_columna(col):
    # Quitar acentos
    col = unicodedata.normalize('NFKD', col).encode('ascii', errors='ignore').decode('utf-8')
    # Reemplazar caracteres problemáticos por _
    col = re.sub(r"[\s\-:()\t]+", "_", col)
    # Eliminar guiones bajos al principio y al final
    col = col.strip("_")
    # Minúsculas
    col = col.lower()
    return col

def limpiar_dataframe(df):
    df.columns = [limpiar_columna(col) for col in df.columns]

    # Identificar columnas de tipo 'object' que podrían ser numéricas
    columnas_objetivo = df.select_dtypes(include=['object']).columns
    for col in columnas_objetivo:
        df[col] = df[col].astype(str).str.replace('\\u202f', '')  # eliminar espacios finos
        df[col] = df[col].str.replace(' ', '')  # eliminar espacios normales
        df[col] = df[col].str.replace('€', '')
        df[col] = df[col].str.replace('%', '')
        df[col] = df[col].str.replace('>', '')
        df[col] = df[col].str.replace(',', '.')  # usar punto como separador decimal

        try:
            df[col] = df[col].astype(float)
        except ValueError:
            continue
    return df

def cargar_y_limpiar_csv(path):
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8')
        df = limpiar_dataframe(df)
        print(f"✅ Archivo cargado y limpiado: {path}")
        return df
    else:
        print(f"❌ Archivo no encontrado: {path}")
        return None
