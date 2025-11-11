#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/main.py
 ------------------------------------------------------------
 Punto de entrada del sistema de Regresión Lineal.

 Funcionalidad principal:
   - Lee y valida uno o varios bloques de configuración desde `input.txt`.
   - Carga datasets (.csv, .xlsx, .ods) y detecta encabezados/datos reales.
   - Selecciona columnas Y y X₁..Xₘ según la configuración.
   - Efectúa preprocesamiento numérico (conversión a float y eliminación de NaN).
   - Detecta automáticamente el caso de regresión simple (1 X) o múltiple (≥2 X).
   - Genera bloques LaTeX por instancia (una sección por instancia).
   - Compila un único PDF con todas las instancias procesadas.

 Notas de diseño:
   - La lógica de regresión reside en `src/regression/`.
   - La generación de LaTeX y compilación a PDF reside en `src/report/`.
   - Este módulo solo orquesta el flujo de datos y la selección de rutas.
 ------------------------------------------------------------
"""

from __future__ import annotations
import sys
import unicodedata
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd

from src.core.config import Config
from src.core.data_extractor.loader import load_dataset
from src.core.data_extractor.preprocess_regression import ensure_numeric_subset
from src.regression.linear_regression import run_linear_regression  # (usado para typing/consistencia)
from src.regression.simple_regression import run_simple_regression
from src.report.report_builder import build_full_report_block
from src.report.report_latex import render_all_instances_pdf


# ------------------------------------------------------------
# Utilidades internas
# ------------------------------------------------------------
def normalize_str(s: str) -> str:
    """
    Normaliza cadenas para búsqueda insensible a mayúsculas/acentos.

    Proceso:
      - Trim.
      - Minúsculas.
      - Normalización Unicode (NFD) y eliminación de marcas diacríticas.

    Ejemplos:
      "  X_Áreas  " -> "x_areas"
    """
    s = str(s).strip().lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def parse_multiple_datasets(path: str) -> List[Config]:
    """
    Divide `input.txt` en múltiples secciones de configuración, cada una
    comenzando con la clave `DATASET`, y construye un objeto `Config`
    por sección.

    Reglas:
      - Se considera el inicio de un bloque al encontrar una línea que
        comience con "DATASET" (ignorando espacios/tabs).
      - Solo se agrega un bloque si contiene al menos una línea DATASET.

    Retorna
    -------
    List[Config]
        Lista de configuraciones independientes (una por dataset).
    """
    sections: List[Path] = []
    content = Path(path).read_text(encoding="utf-8").splitlines()

    temp, idx = [], 1
    for line in content:
        if line.strip().upper().startswith("DATASET"):
            # Guarda bloque previo (si tenía DATASET)
            if temp and any("DATASET" in l.upper() for l in temp):
                subfile = Path(f"/tmp/config_part_{idx}.txt")
                subfile.write_text("\n".join(temp), encoding="utf-8")
                sections.append(subfile)
                temp, idx = [], idx + 1
        temp.append(line)

    # Último bloque (si aplica)
    if temp and any("DATASET" in l.upper() for l in temp):
        subfile = Path(f"/tmp/config_part_{idx}.txt")
        subfile.write_text("\n".join(temp), encoding="utf-8")
        sections.append(subfile)

    cfgs = [Config(str(p)) for p in sections]
    return cfgs


def select_columns(df: pd.DataFrame, cfg: Config) -> Tuple[List[str], str, Dict[str, str]]:
    """
    Determina las columnas Y (dependiente) y X (independientes) a partir
    del DataFrame y de la configuración.

    Estrategia:
      - Si no se especifica la dependiente, se usa la última columna.
      - Si `USE_ALL_ATTRIBUTES=true`, las X serán todas las columnas menos Y.
      - Si se especifican X explícitas y `USE_ALL_ATTRIBUTES=false`, se mapean
        por nombre normalizado (tolerante a tildes/cambios de mayúsculas).

    Validaciones:
      - Debe haber al menos 1 X y 1 Y.

    Retorna
    -------
    (x_cols, y, normalized_cols)
      x_cols : List[str]  -> columnas X reales en el DataFrame (orden final)
      y      : str        -> columna Y real
      normalized_cols : Dict[str, str] -> mapa nombre_normalizado -> nombre_real
    """
    cols = list(df.columns)
    normalized_cols = {normalize_str(c): c for c in cols}

    # Variable dependiente (Y)
    y_key = cfg.target_column or getattr(cfg, "dependent_variable", None)
    if not y_key:
        y = cols[-1]  # fallback: última columna
    else:
        nk = normalize_str(y_key)
        if nk not in normalized_cols:
            raise ValueError(
                f"La columna dependiente '{y_key}' no existe.\nColumnas disponibles: {cols}"
            )
        y = normalized_cols[nk]

    # Variables independientes (X)
    x_alias = cfg.attributes or getattr(cfg, "independent_variables", None)
    if x_alias and not cfg.use_all_attributes:
        x_cols = [
            normalized_cols[normalize_str(c)]
            for c in x_alias
            if normalize_str(c) in normalized_cols and normalized_cols[normalize_str(c)] != y
        ]
    else:
        x_cols = [c for c in cols if c != y]

    if len(x_cols) < 1:
        raise ValueError("Se requieren ≥ 2 columnas totales (al menos 1 X + 1 Y).")

    return x_cols, y, normalized_cols


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    """
    Orquesta el flujo completo:
      - Lee `input.txt` desde argv[1].
      - Procesa cada dataset y sus instancias.
      - Selecciona modo simple/múltiple.
      - Concatena bloques LaTeX y compila un PDF unificado.
    """
    if len(sys.argv) != 2:
        print("Uso: python -m src.main input.txt")
        sys.exit(1)

    input_path = sys.argv[1]

    # Parseo de múltiples datasets
    try:
        configs = parse_multiple_datasets(input_path)
    except Exception as e:
        print(f"[ERROR] No fue posible leer/parsing el archivo de entrada: {e}")
        sys.exit(1)

    if not configs:
        print("[ERROR] No se detectó ningún bloque de configuración con DATASET=")
        sys.exit(1)

    all_instances_blocks: List[str] = []
    instance_counter = 1

    for cfg in configs:
        print(f"=== Procesando dataset: {cfg.dataset} ===")

        # Validación de dataset
        if not cfg.dataset or not Path(cfg.dataset).exists():
            print(f"[WARN] Dataset inválido o no encontrado: {cfg.dataset}")
            continue

        # Carga de datos (como texto para detección de tabla/encabezados)
        df = load_dataset(cfg.dataset, cfg.sheet).astype(str)

        # Selección de columnas
        try:
            x_cols, y_col, normalized_cols = select_columns(df, cfg)
        except Exception as e:
            print(f"[WARN] Selección de columnas fallida: {e}")
            continue

        # Preprocesamiento numérico
        used_cols = [y_col] + x_cols
        df_num, dropped = ensure_numeric_subset(df, used_cols)
        if dropped > 0:
            print(f"[INFO] Filas descartadas por NaN/no-numéricas: {dropped}")

        # Normalización y validación de instancias
        norm_instances: List[Dict[str, str]] = []
        for inst in cfg.instances:
            inst_norm: Dict[str, str] = {}
            for k, v in inst.items():
                nk = normalize_str(k)
                if nk in normalized_cols:
                    true_col = normalized_cols[nk]
                    if true_col in x_cols:
                        inst_norm[true_col] = v
            if all(c in inst_norm for c in x_cols):
                norm_instances.append(inst_norm)
            else:
                print(f"[WARN] Instancia ignorada (faltan columnas X requeridas): {x_cols}")

        if not norm_instances:
            print("[WARN] No hay instancias válidas para este dataset.")
            continue

        # Detección de tipo de regresión
        if len(x_cols) == 1:
            print(f"[INFO] Usando modo regresión simple ({x_cols[0]})")
            block = run_simple_regression(df_num, y_col, x_cols[0], norm_instances, instance_counter)
        else:
            print(f"[INFO] Usando modo regresión múltiple ({len(x_cols)} variables)")
            block = build_full_report_block(norm_instances, df_num, y_col, x_cols)

        all_instances_blocks.append(block)
        instance_counter += len(norm_instances)

    # Generación del PDF (unificado)
    if not all_instances_blocks:
        print("[ERROR] No se generaron bloques de reporte.")
        sys.exit(1)

    all_text = "\n".join(all_instances_blocks)
    out_path = (configs[0].report_path or "output/reporte.pdf").replace(".pdf", "_all.pdf")

    try:
        render_all_instances_pdf(out_path, all_text)
        print(f"[OK] Reporte generado con todas las instancias: {out_path}")
    except Exception as e:
        print(f"[ERROR] Fallo en la generación del PDF: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

