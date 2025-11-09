#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/core/data_extractor/preprocess_regression.py
 ------------------------------------------------------------
 Descripción:

 Utilidades de preprocesamiento para regresión lineal numérica:
  - Conversión a float de columnas relevantes.
  - Eliminación de filas con NaN en columnas usadas.
  - Reporte de filas descartadas.

 (No se realizan discretizaciones ni one-hot; el caso categórico
 no es necesario según los requisitos.)
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
    Asegura que las columnas indicadas sean numéricas y elimina filas
    con valores no convertibles.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.
    used_cols : List[str]
        Columnas que deben ser numéricas (y + Xs).

    Retorna
    -------
    Tuple[pd.DataFrame, int]
        DataFrame filtrado y número de filas eliminadas.
    """
    df2 = df.copy()
    for c in used_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    before = len(df2)
    df2 = df2.dropna(subset=used_cols).reset_index(drop=True)
    dropped = before - len(df2)
    return df2, dropped
# ---------------------------------------------------------------------------

