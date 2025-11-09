#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/core/config.py
 ------------------------------------------------------------
 Descripción:

 Módulo encargado de leer y procesar un archivo de configuración
 (input.txt) con soporte para pares clave=valor, instancias y
 comentarios de línea o bloque.

 Extensión para Regresión Lineal:
  - Se añaden alias DEPENDENT_VARIABLE (para Y)
    e INDEPENDENT_VARIABLES (para lista de X)
    para mantener compatibilidad y flexibilidad.
 ------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from src.core.utils import parse_bool

class Config:
    """
    Clase principal que administra la carga, interpretación y validación
    del archivo de configuración input.txt.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")

        self.kv: Dict[str, str] = {}
        self.instances: List[Dict[str, str]] = []
        self._parse()
        self._validate_duplicates()

    # ------------------------------------------------------------
    # === PARSEO DE ARCHIVO ===
    # ------------------------------------------------------------
    def _parse(self):
        """
        Lee input.txt e interpreta pares clave=valor e instancias.
        Soporta comentarios de bloque (/* ... */) y línea (# ...).
        """
        current_instance = None
        in_block_comment = False

        with self.path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()

                # Manejo de comentarios /* ... */
                if "/*" in line and "*/" in line:
                    line = line.split("/*", 1)[0] + line.split("*/", 1)[1]
                elif "/*" in line:
                    in_block_comment = True
                    line = line.split("/*", 1)[0]
                elif "*/" in line:
                    in_block_comment = False
                    line = line.split("*/", 1)[1]

                if in_block_comment:
                    continue

                # Ignorar líneas vacías o comentarios de línea
                if not line or line.startswith("#"):
                    continue

                # Inicia bloque INSTANCE:
                if line.endswith(":") and line[:-1].strip().upper().startswith("INSTANCE"):
                    if current_instance is not None:
                        self.instances.append(current_instance)
                    current_instance = {}
                    continue

                # Solo procesa líneas clave=valor
                if "=" not in line:
                    continue

                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()

                # Guarda clave-valor según contexto (global o instancia)
                if current_instance is not None:
                    current_instance[k] = v
                else:
                    if k in self.kv:
                        prev = self.kv[k]
                        if not isinstance(prev, list):
                            self.kv[k] = [prev, v]
                        else:
                            self.kv[k].append(v)
                    else:
                        self.kv[k] = v

        # Agrega la última instancia si existe
        if current_instance:
            self.instances.append(current_instance)

    # ------------------------------------------------------------
    # === VALIDACIÓN DE DUPLICADOS ===
    # ------------------------------------------------------------
    def _validate_duplicates(self):
        """
        Verifica si hay claves críticas duplicadas.
        """
        critical = ("DATASET", "TARGET_COLUMN", "DEPENDENT_VARIABLE")
        for key in critical:
            val = self.kv.get(key)
            if isinstance(val, list):
                msg = (
                    f"[ERROR] Se detectaron múltiples definiciones de '{key}' en {self.path.name}:\n"
                    f"         {val}\n"
                    f"         Mantén solo una definición válida."
                )
                raise ValueError(msg)

    # ------------------------------------------------------------
    # === PROPIEDADES PRINCIPALES ===
    # ------------------------------------------------------------
    @property
    def dataset(self) -> str:
        v = self.kv.get("DATASET", "")
        if isinstance(v, list):
            v = v[-1]
        return v.strip()

    @property
    def sheet(self) -> Optional[str]:
        """
        Devuelve el nombre de la hoja (sheet/hoja), detectando
        claves en inglés o español.
        """
        keys = {k.strip().upper(): v for k, v in self.kv.items()}
        posibles = ("SHEET", "HOJA", "SHEETS", "HOJAS", "SHEET_NAME", "PAGINA", "TAB")
        for key in posibles:
            if key in keys:
                val = keys[key]
                if isinstance(val, list):
                    val = val[-1]
                return val
        return None

    # ------------------------------------------------------------
    # === VARIABLES DEPENDIENTES / INDEPENDIENTES ===
    # ------------------------------------------------------------
    @property
    def target_column(self) -> Optional[str]:
        """
        Alias principal para columna objetivo.
        Compatible con TARGET_COLUMN (Bayes) y DEPENDENT_VARIABLE (Regresión).
        """
        v = self.kv.get("TARGET_COLUMN") or self.kv.get("DEPENDENT_VARIABLE")
        if isinstance(v, list):
            v = v[-1]
        return v

    @property
    def dependent_variable(self) -> Optional[str]:
        """
        Alias explícito para compatibilidad con input.txt de regresión lineal.
        """
        v = self.kv.get("DEPENDENT_VARIABLE")
        if isinstance(v, list):
            v = v[-1]
        return v

    @property
    def attributes(self) -> Optional[List[str]]:
        """
        Atributos independientes (X₁..Xₙ).
        Compatible con ATTRIBUTES (Bayes) e INDEPENDENT_VARIABLES (Regresión).
        """
        raw = self.kv.get("ATTRIBUTES") or self.kv.get("INDEPENDENT_VARIABLES")
        if not raw:
            return None
        if isinstance(raw, list):
            raw = raw[-1]
        return [c.strip() for c in str(raw).split(",") if c.strip()]

    @property
    def independent_variables(self) -> Optional[List[str]]:
        """
        Alias explícito para compatibilidad con input.txt de regresión lineal.
        """
        raw = self.kv.get("INDEPENDENT_VARIABLES")
        if not raw:
            return None
        if isinstance(raw, list):
            raw = raw[-1]
        return [c.strip() for c in str(raw).split(",") if c.strip()]

    # ------------------------------------------------------------
    # === OPCIONES ADICIONALES ===
    # ------------------------------------------------------------
    @property
    def use_all_attributes(self) -> bool:
        return parse_bool(self.kv.get("USE_ALL_ATTRIBUTES", "true"))

    @property
    def report_path(self) -> Optional[str]:
        v = self.kv.get("REPORT")
        if isinstance(v, list):
            v = v[-1]
        return v

    # Claves heredadas del proyecto bayesiano (mantienen compatibilidad)
    @property
    def laplace_alpha(self) -> float:
        try:
            v = self.kv.get("LAPLACE_ALPHA", "0")
            if isinstance(v, list):
                v = v[-1]
            return float(v)
        except Exception:
            return 0.0

    @property
    def numeric_mode(self) -> str:
        v = self.kv.get("NUMERIC_MODE", "raw")
        if isinstance(v, list):
            v = v[-1]
        return v

    @property
    def bins(self) -> int:
        try:
            v = self.kv.get("BINS", "5")
            if isinstance(v, list):
                v = v[-1]
            return int(v)
        except Exception:
            return 5

    @property
    def discretize_strategy(self) -> str:
        v = self.kv.get("DISCRETIZE_STRATEGY", "quantile")
        if isinstance(v, list):
            v = v[-1]
        return v
# ---------------------------------------------------------------------------

