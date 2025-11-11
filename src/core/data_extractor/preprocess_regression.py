#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/core/data_extractor/preprocess_regression.py
 ------------------------------------------------------------
 Módulo de preprocesamiento para modelos de regresión lineal numérica.

 Las rutinas incluidas garantizan que las columnas relevantes del
 dataset sean numéricas, eliminando aquellas filas con valores no
 convertibles o faltantes. Esto asegura que los cálculos matriciales
 posteriores (como la resolución de β o el cómputo de R²) operen sobre
 datos consistentes y válidos.

 Funcionalidades principales:
   - Conversión de columnas especificadas a tipo float.
   - Eliminación de filas con valores NaN resultantes de conversiones fallidas.
   - Conteo y retorno de las filas descartadas.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd


def ensure_numeric_subset(
    df: pd.DataFrame,
    used_cols: List[str]
) -> Tuple[pd.DataFrame, int]:
    """
    Convierte las columnas especificadas a tipo numérico y elimina las filas
    con valores no válidos o no convertibles.

    Esta función es utilizada antes del cálculo de regresión lineal
    para depurar el conjunto de datos y asegurar que todos los valores
    de entrada (X₁..Xₙ) y salida (Y) sean numéricos. Cualquier valor
    no interpretable como número es reemplazado por NaN y posteriormente
    eliminado del conjunto final.

    Parámetros
    ----------
    df : pd.DataFrame
        Conjunto de datos original, posiblemente con valores no numéricos
        en las columnas de interés.
    used_cols : List[str]
        Lista de nombres de columnas que deben convertirse a tipo float.
        Incluye tanto la variable dependiente (Y) como las independientes (X₁..Xₙ).

    Retorna
    -------
    Tuple[pd.DataFrame, int]
        - DataFrame limpio con las filas válidas conservadas.
        - Número entero correspondiente a las filas eliminadas.

    Flujo lógico
    ------------
    1. Se crea una copia del DataFrame original para evitar mutaciones.
    2. Cada columna listada en `used_cols` se convierte con `pd.to_numeric`,
       forzando valores no convertibles a `NaN`.
    3. Se eliminan todas las filas con `NaN` en cualquiera de las columnas
       usadas para el modelo.
    4. Se retorna el DataFrame depurado junto con la cantidad de filas
       descartadas durante el proceso.
    """
    # Copia del DataFrame para no alterar el original
    df2 = df.copy()

    # Conversión a numérico de todas las columnas relevantes
    for c in used_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # Conteo previo y posterior para determinar filas eliminadas
    before = len(df2)
    df2 = df2.dropna(subset=used_cols).reset_index(drop=True)
    dropped = before - len(df2)

    return df2, dropped
# ---------------------------------------------------------------------------

