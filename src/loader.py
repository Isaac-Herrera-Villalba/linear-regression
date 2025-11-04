#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/loader.py
 ------------------------------------------------------------
 Descripción:

 Módulo encargado de cargar datasets desde distintos formatos 
 (CSV, XLSX, ODS) y detectar automáticamente la tabla de datos 
 ignorando filas o columnas vacías.
 """

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

#Detecta automáticamente el bloque de datos (elimina filas/columnas vacías iniciales) y ajusta el encabezado a la primera fila no vacía.
def _detect_table(df: pd.DataFrame) -> pd.DataFrame:
    # Elimina columnas completamente vacías
    df = df.dropna(axis=1, how="all")

    # Encuentra la primera fila con al menos un valor no nulo
    first_valid_row = None
    for i, row in df.iterrows():
        if row.notna().any():
            first_valid_row = i
            break

    if first_valid_row is None:
        raise ValueError("No se detectaron datos válidos en el archivo.")

    # Usa esa fila como encabezado y recorta las filas anteriores
    new_header = df.iloc[first_valid_row].astype(str)
    df = df.iloc[first_valid_row + 1:].copy()
    df.columns = new_header

    # Elimina filas completamente vacías restantes
    df = df.dropna(how="all")

    # Convierte todas las columnas a string por consistencia
    df.columns = [str(c).strip() for c in df.columns]
    df = df.astype(str)

    return df.reset_index(drop=True)

# Carga un dataset desde un archivo CSV, XLSX o ODS, detectando automáticamente la región de la tabla (sin importar posición).
def load_dataset(path: str, sheet: str | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    ext = p.suffix.lower()

    # --- Carga según tipo de archivo ---
    if ext == ".csv":
        df = pd.read_csv(p, encoding="utf-8", header=None)

    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(p, sheet_name=sheet or 0, engine="openpyxl", header=None)

    elif ext == ".ods":
        df = pd.read_excel(p, sheet_name=sheet or 0, engine="odf", header=None)

    else:
        raise ValueError(f"Formato no soportado: {ext}")

    # --- Detección automática del bloque de datos ---
    df = _detect_table(df)
    return df
# ---------------------------------------------------------------------------------

