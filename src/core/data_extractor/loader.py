#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/core/data_extractor/loader.py
 ------------------------------------------------------------
 Módulo encargado de la carga de datasets desde archivos en formatos
 `.csv`, `.xlsx` y `.ods`, asegurando una detección robusta del bloque
 real de datos, incluso cuando el archivo contiene filas o columnas
 vacías iniciales o encabezados desplazados.

 Las funciones incluidas realizan:
   - Lectura uniforme independientemente del formato.
   - Detección automática del encabezado de la tabla.
   - Normalización de nombres de columnas.
   - Eliminación de filas y columnas vacías.

 Dependencias:
   - pandas
   - numpy
 ------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np


def _detect_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta automáticamente la región tabular válida dentro de un DataFrame.

    Este procedimiento elimina filas y columnas completamente vacías,
    localiza la primera fila con datos válidos y la utiliza como encabezado
    del conjunto. El resultado se normaliza para garantizar consistencia
    en los nombres de las columnas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame leído directamente desde un archivo plano o de hoja de cálculo
        sin encabezado definido (header=None).

    Retorna
    -------
    pd.DataFrame
        DataFrame limpio, con encabezados válidos y sin filas o columnas vacías.

    Excepciones
    -----------
    ValueError
        Si no se detectan filas con valores válidos.

    Lógica general
    --------------
    1. Se eliminan columnas completamente vacías.
    2. Se localiza la primera fila con al menos un valor no nulo.
    3. Se usa dicha fila como encabezado y se descartan las anteriores.
    4. Se eliminan filas residuales vacías.
    5. Se normalizan los nombres de columnas a texto simple y se fuerzan
       todos los datos a tipo `str` para evitar conflictos posteriores.
    """
    # Elimina columnas sin datos
    df = df.dropna(axis=1, how="all")

    # Busca la primera fila con algún valor no nulo
    first_valid_row = None
    for i, row in df.iterrows():
        if row.notna().any():
            first_valid_row = i
            break

    # Verificación de validez del bloque
    if first_valid_row is None:
        raise ValueError("No se detectaron datos válidos en el archivo.")

    # Redefine encabezado y recorta filas anteriores
    new_header = df.iloc[first_valid_row].astype(str)
    df = df.iloc[first_valid_row + 1:].copy()
    df.columns = new_header

    # Limpieza final
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.astype(str)

    return df.reset_index(drop=True)


def load_dataset(path: str, sheet: str | None = None) -> pd.DataFrame:
    """
    Carga un dataset desde archivo CSV, XLSX u ODS, aplicando detección
    automática del bloque de datos real.

    Esta función abstrae la lectura de distintos formatos de archivo
    mediante `pandas`, permitiendo al sistema de regresión lineal trabajar
    con un DataFrame uniforme independientemente del formato de entrada.

    Parámetros
    ----------
    path : str
        Ruta absoluta o relativa del archivo de datos.
    sheet : str | None, opcional
        Nombre o índice de la hoja (solo para archivos `.xlsx` o `.ods`).

    Retorna
    -------
    pd.DataFrame
        DataFrame con la tabla limpia, encabezado corregido y datos en formato texto.

    Excepciones
    -----------
    FileNotFoundError
        Si la ruta proporcionada no existe.
    ValueError
        Si el formato de archivo no está soportado o el bloque de datos no es válido.

    Flujo lógico
    ------------
    1. Valida la existencia del archivo.
    2. Determina el formato por extensión.
    3. Carga los datos sin encabezado.
    4. Invoca `_detect_table()` para aislar el bloque de datos.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")

    ext = p.suffix.lower()

    # Carga de archivo según su formato
    if ext == ".csv":
        df = pd.read_csv(p, encoding="utf-8", header=None)
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(p, sheet_name=sheet or 0, engine="openpyxl", header=None)
    elif ext == ".ods":
        df = pd.read_excel(p, sheet_name=sheet or 0, engine="odf", header=None)
    else:
        raise ValueError(f"Formato no soportado: {ext}")

    # Detección automática del bloque tabular
    df = _detect_table(df)
    return df

