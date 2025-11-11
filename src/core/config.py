#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/core/config.py
 ------------------------------------------------------------
 Módulo encargado de la lectura, interpretación y validación del
 archivo de configuración `input.txt`, utilizado por el sistema
 de Regresión Lineal.

 Implementa una clase de alto nivel (`Config`) que abstrae el
 manejo de claves globales, instancias y alias entre proyectos,
 permitiendo compatibilidad con configuraciones previas de otros
 módulos (por ejemplo, clasificación Bayesiana).

 Características principales:
   - Lectura estructurada de pares clave=valor.
   - Soporte para comentarios de bloque (/* ... */) y línea (# ...).
   - Soporte para múltiples instancias (bloques INSTANCE:).
   - Alias automáticos para variables dependientes e independientes.
   - Validación de duplicados críticos en la configuración.

 Dependencia:
   - src.core.utils.parse_bool
 ------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from src.core.utils import parse_bool


class Config:
    """
    Representa una configuración cargada desde un archivo `input.txt`.

    Esta clase administra la lectura y almacenamiento de pares clave=valor
    definidos en el archivo de configuración, además de manejar múltiples
    instancias (`INSTANCE:`) y realizar validaciones sobre duplicados de
    parámetros críticos (por ejemplo, `DATASET` o `DEPENDENT_VARIABLE`).

    Cada instancia del objeto contiene:
      - Un diccionario de claves globales (`self.kv`).
      - Una lista de instancias (`self.instances`), donde cada una
        corresponde a un conjunto de valores específicos para predicción.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de entrada: {path}")

        self.kv: Dict[str, str] = {}
        self.instances: List[Dict[str, str]] = []

        # Proceso de carga y validación
        self._parse()
        self._validate_duplicates()

    # ============================================================
    # === PARSEO DE ARCHIVO DE CONFIGURACIÓN ======================
    # ============================================================
    def _parse(self):
        """
        Lee y analiza un archivo de configuración `input.txt`.

        Reconoce:
          - Comentarios de bloque (/* ... */) y de línea (# ...).
          - Pares clave=valor.
          - Bloques de instancias (`INSTANCE:`) con sus atributos.

        Flujo lógico:
          1. El archivo se recorre línea por línea.
          2. Se eliminan o ignoran comentarios y líneas vacías.
          3. Cuando se detecta un encabezado `INSTANCE:`, se inicia
             un nuevo bloque de instancia.
          4. Las líneas dentro del bloque se almacenan como pares clave=valor.
          5. Los valores globales se almacenan en `self.kv`.
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

                # Ignora líneas vacías o comentarios de línea
                if not line or line.startswith("#"):
                    continue

                # Detección de bloque INSTANCE:
                if line.endswith(":") and line[:-1].strip().upper().startswith("INSTANCE"):
                    if current_instance is not None:
                        self.instances.append(current_instance)
                    current_instance = {}
                    continue

                # Procesa líneas tipo clave=valor
                if "=" not in line:
                    continue

                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()

                # Contexto de almacenamiento
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

        # Última instancia detectada
        if current_instance:
            self.instances.append(current_instance)

    # ============================================================
    # === VALIDACIÓN DE DUPLICADOS ===============================
    # ============================================================
    def _validate_duplicates(self):
        """
        Verifica la existencia de claves críticas duplicadas.

        Si alguna de las claves `DATASET`, `TARGET_COLUMN` o
        `DEPENDENT_VARIABLE` aparece más de una vez, se lanza
        una excepción indicando los valores repetidos.
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

    # ============================================================
    # === PROPIEDADES PRINCIPALES ================================
    # ============================================================
    @property
    def dataset(self) -> str:
        """Devuelve la ruta al archivo de dataset especificado en `input.txt`."""
        v = self.kv.get("DATASET", "")
        if isinstance(v, list):
            v = v[-1]
        return v.strip()

    @property
    def sheet(self) -> Optional[str]:
        """
        Devuelve el nombre de la hoja dentro del archivo de datos.
        Reconoce claves en inglés o español (SHEET, HOJA, SHEETS, etc.).
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

    # ============================================================
    # === VARIABLES DEPENDIENTES E INDEPENDIENTES ================
    # ============================================================
    @property
    def target_column(self) -> Optional[str]:
        """Devuelve el nombre de la variable dependiente (`Y`), compatible con configuraciones previas."""
        v = self.kv.get("TARGET_COLUMN") or self.kv.get("DEPENDENT_VARIABLE")
        if isinstance(v, list):
            v = v[-1]
        return v

    @property
    def dependent_variable(self) -> Optional[str]:
        """Alias explícito para la variable dependiente (`Y`)."""
        v = self.kv.get("DEPENDENT_VARIABLE")
        if isinstance(v, list):
            v = v[-1]
        return v

    @property
    def attributes(self) -> Optional[List[str]]:
        """Devuelve la lista de variables independientes, compatible con Bayes y Regresión Lineal."""
        raw = self.kv.get("ATTRIBUTES") or self.kv.get("INDEPENDENT_VARIABLES")
        if not raw:
            return None
        if isinstance(raw, list):
            raw = raw[-1]
        return [c.strip() for c in str(raw).split(",") if c.strip()]

    @property
    def independent_variables(self) -> Optional[List[str]]:
        """Devuelve explícitamente las variables independientes definidas en `input.txt`."""
        raw = self.kv.get("INDEPENDENT_VARIABLES")
        if not raw:
            return None
        if isinstance(raw, list):
            raw = raw[-1]
        return [c.strip() for c in str(raw).split(",") if c.strip()]

    # ============================================================
    # === OPCIONES ADICIONALES ==================================
    # ============================================================
    @property
    def use_all_attributes(self) -> bool:
        """Determina si se deben usar todas las columnas del dataset excepto la variable dependiente."""
        return parse_bool(self.kv.get("USE_ALL_ATTRIBUTES", "true"))

    @property
    def report_path(self) -> Optional[str]:
        """Ruta completa del archivo PDF de salida definido en la configuración."""
        v = self.kv.get("REPORT")
        if isinstance(v, list):
            v = v[-1]
        return v

    # ============================================================
    # === CLAVES HEREDADAS (COMPATIBILIDAD) ======================
    # ============================================================
    @property
    def laplace_alpha(self) -> float:
        """Valor de suavizado Laplaciano heredado del sistema bayesiano (no usado en regresión)."""
        try:
            v = self.kv.get("LAPLACE_ALPHA", "0")
            if isinstance(v, list):
                v = v[-1]
            return float(v)
        except Exception:
            return 0.0

    @property
    def numeric_mode(self) -> str:
        """Modo de interpretación numérica (por compatibilidad con Bayes)."""
        v = self.kv.get("NUMERIC_MODE", "raw")
        if isinstance(v, list):
            v = v[-1]
        return v

    @property
    def bins(self) -> int:
        """Número de intervalos discretos (solo relevante en clasificación Bayesiana)."""
        try:
            v = self.kv.get("BINS", "5")
            if isinstance(v, list):
                v = v[-1]
            return int(v)
        except Exception:
            return 5

    @property
    def discretize_strategy(self) -> str:
        """Estrategia de discretización para valores numéricos (solo compatibilidad)."""
        v = self.kv.get("DISCRETIZE_STRATEGY", "quantile")
        if isinstance(v, list):
            v = v[-1]
        return v
# ---------------------------------------------------------------------------

