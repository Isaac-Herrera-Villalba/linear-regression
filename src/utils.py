#!/usr/bin/env python3

"""
 src/utils.py
 ------------------------------------------------------------
 Descripción:

 Módulo de utilidades auxiliares para el sistema Bayesiano
"""

from __future__ import annotations
from typing import Dict, Any

# Normaliza los nombres de columnas
def normalize_colnames(cols):
    # Devuelve nombres tal cual (sin modificar).
    return list(cols)

# Convierte una cadena a valor booleano
def parse_bool(s: str) -> bool:
    # Interpreta como True si el texto equivale a "1", "true", "yes", "y", "t", "si" o "sí"
    return str(s).strip().lower() in {"1", "true", "yes", "y", "t", "si", "sí"}

# Formatea un diccionario en una cadena tipo 'clave=valor, clave=valor'
def dotted(d: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())
# ---------------------------------------------------------------------------------

