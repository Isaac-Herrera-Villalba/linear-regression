#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/core/config.py
 ------------------------------------------------------------
 Módulo encargado de la lectura, interpretación y validación del
 archivo de configuración `input.txt` para el sistema de regresión lineal.

 Ahora soporta múltiples bloques `DATASET=` dentro del mismo archivo,
 cada uno con su propio conjunto de parámetros e instancias.
 ------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from src.core.utils import parse_bool


class Config:
    """
    Gestiona la lectura de un archivo `input.txt` que puede contener
    múltiples datasets, cada uno con su propio conjunto de instancias.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")

        # Lista de bloques de configuración completos
        self.blocks: List[Dict] = []

        # Proceso de carga
        self._parse()

    # ============================================================
    # === PARSEO DE BLOQUES ======================================
    # ============================================================
    def _parse(self):
        """
        Lee y separa múltiples bloques `DATASET=` dentro del mismo archivo.
        Cada bloque mantiene sus claves y sus instancias asociadas.
        """
        with self.path.open("r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]

        current_block = None
        current_instance = None
        in_block_comment = False

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            # Comentarios de bloque /* ... */
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

            # Ignora comentarios de línea
            if line.startswith("#"):
                continue

            # Nuevo bloque de DATASET
            if line.upper().startswith("DATASET"):
                # Si ya había un bloque activo, lo guardamos
                if current_block:
                    if current_instance:
                        current_block["INSTANCES"].append(current_instance)
                        current_instance = None
                    self.blocks.append(current_block)
                current_block = {"KV": {}, "INSTANCES": []}
                key, val = line.split("=", 1)
                current_block["KV"][key.strip()] = val.strip()
                continue

            # Nuevo bloque de instancia
            if line.endswith(":") and "INSTANCE" in line.upper():
                if current_instance:
                    current_block["INSTANCES"].append(current_instance)
                current_instance = {}
                continue

            # Línea clave=valor
            if "=" in line:
                key, val = [x.strip() for x in line.split("=", 1)]
                if current_instance is not None:
                    current_instance[key] = val
                else:
                    current_block["KV"][key] = val

        # Cierra últimos bloques abiertos
        if current_instance and current_block:
            current_block["INSTANCES"].append(current_instance)
        if current_block:
            self.blocks.append(current_block)

    # ============================================================
    # === UTILIDADES DE BLOQUES ==================================
    # ============================================================
    def get_blocks(self) -> List[Dict]:
        """Devuelve la lista completa de bloques (DATASET + INSTANCES)."""
        return self.blocks

    # ============================================================
    # === ACCESOR COMPATIBLE (modo simple) =======================
    # ============================================================
    @staticmethod
    def get_value(kv: Dict[str, str], key: str, default: str | None = None) -> Optional[str]:
        v = kv.get(key)
        if isinstance(v, list):
            v = v[-1]
        return v if v is not None else default

    @staticmethod
    def get_bool(kv: Dict[str, str], key: str, default: bool = False) -> bool:
        return parse_bool(kv.get(key, str(default)))

