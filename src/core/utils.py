#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/core/utils.py
 ------------------------------------------------------------
 Módulo de utilidades generales para operaciones auxiliares
 en el sistema de Regresión Lineal y otros módulos derivados
 (por ejemplo, proyectos previos de clasificación bayesiana).

 Incluye funciones de normalización de nombres de columnas,
 conversión segura de cadenas a valores booleanos y formateo
 textual de diccionarios en formato compacto.

 Estas funciones no dependen de librerías externas y se
 utilizan en distintas etapas del procesamiento de datos
 y configuración del sistema.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, Any


def normalize_colnames(cols) -> list[str]:
    """
    Devuelve una lista de nombres de columnas normalizados.

    En esta implementación, los nombres se devuelven sin
    alteración, pero la función se mantiene como punto de
    extensión para futuras normalizaciones (por ejemplo,
    eliminación de espacios, acentos o símbolos).

    Parámetros
    ----------
    cols : iterable
        Colección de nombres de columnas.

    Retorna
    -------
    list[str]
        Lista con los nombres de columna procesados.
    """
    return list(cols)


def parse_bool(s: str) -> bool:
    """
    Convierte una cadena de texto a su valor booleano equivalente.

    Se interpreta como `True` si la cadena coincide (ignorando
    mayúsculas y tildes) con alguno de los siguientes valores:
    `"1"`, `"true"`, `"yes"`, `"y"`, `"t"`, `"si"`, `"sí"`.

    Parámetros
    ----------
    s : str
        Cadena de entrada a evaluar.

    Retorna
    -------
    bool
        Valor booleano interpretado a partir de la cadena.
    """
    return str(s).strip().lower() in {"1", "true", "yes", "y", "t", "si", "sí"}


def dotted(d: Dict[str, Any]) -> str:
    """
    Devuelve una representación textual de un diccionario
    en formato `clave=valor`, separando los pares por comas.

    Parámetros
    ----------
    d : Dict[str, Any]
        Diccionario de entrada.

    Retorna
    -------
    str
        Cadena resultante con los pares formateados en
        la forma `k=v, k=v, ...`.
    """
    return ", ".join(f"{k}={v}" for k, v in d.items())
# -------------------------------------------------------

