#!/usr/bin/env python3

"""
 src/preprocess.py
 ------------------------------------------------------------
 Descripción:

 Módulo encargado del preprocesamiento de datos antes de aplicar 
 el modelo Bayesiano
"""

from __future__ import annotations
import pandas as pd

# Función para discretizar variables numéricas en intervalos o categorías
def discretize(df: pd.DataFrame, attrs: list[str], bins: int = 5, strategy: str = "quantile"):
    #Discretiza columnas numéricas de un DataFrame en categorías.
    df_copy = df.copy() # Se trabaja sobre una copia para no alterar el original
    for col in attrs:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            if strategy == "quantile":
                df_copy[col] = pd.qcut(df_copy[col], q=bins, duplicates="drop") # Discretización por cuantiles
            else:
                df_copy[col] = pd.cut(df_copy[col], bins=bins) # Discretización uniforme por rango
    return df_copy # Devuelve el DataFrame con las variables discretizadas
