#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/main.py
 ------------------------------------------------------------
 Descripción:

 Programa principal que ejecuta el flujo completo de Regresión Lineal:
   input.txt  →  carga dataset  →  preprocesa numéricos  →  OLS  →  PDF

 Ahora genera un único reporte PDF con todas las instancias válidas,
 mostrando todos los pasos (1–4) por cada una.
 ------------------------------------------------------------
"""

from __future__ import annotations
import sys
import unicodedata
from typing import Dict, List
import pandas as pd

from src.core.config import Config
from src.core.data_extractor.loader import load_dataset
from src.core.data_extractor.preprocess_regression import ensure_numeric_subset
from src.regression.linear_regression import run_linear_regression
from src.report.report_builder import build_full_report_block
from src.report.report_latex import render_all_instances_pdf

def normalize_str(s: str) -> str:
    """Normaliza cadenas para comparación (quita tildes, minúsculas, sin espacios extra)."""
    s = str(s).strip().lower()
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def select_columns(df: pd.DataFrame, cfg: Config) -> tuple[list[str], str, dict[str, str]]:
    """
    Selecciona variables dependiente e independientes según input.txt.
    Acepta alias:
      - Dependiente: TARGET_COLUMN, DEPENDENT_VARIABLE
      - Independientes: ATTRIBUTES, INDEPENDENT_VARIABLES
    Si no hay lista de X y USE_ALL_ATTRIBUTES=true, usa todas menos y.
    """
    cols = list(df.columns)
    normalized_cols = {normalize_str(c): c for c in cols}

    # Variable dependiente (y)
    y_key = cfg.target_column or getattr(cfg, "dependent_variable", None)
    if not y_key:
        y = cols[-1]
    else:
        nk = normalize_str(y_key)
        if nk not in normalized_cols:
            raise ValueError(
                f"La columna dependiente '{y_key}' no existe.\nColumnas: {cols}"
            )
        y = normalized_cols[nk]

    # Variables independientes (X)
    x_alias = cfg.attributes or getattr(cfg, "independent_variables", None)
    if x_alias and not cfg.use_all_attributes:
        x_cols = [normalized_cols[normalize_str(c)]
                  for c in x_alias if normalize_str(c) in normalized_cols and normalized_cols[normalize_str(c)] != y]
    else:
        x_cols = [c for c in cols if c != y]

    if len(x_cols) < 1:
        raise ValueError("Se requieren ≥ 2 columnas (al menos 1 X + 1 y).")

    return x_cols, y, normalized_cols


def main():
    if len(sys.argv) != 2:
        print("Uso: python -m src.main input.txt")
        sys.exit(1)

    # Carga configuración y dataset
    cfg = Config(sys.argv[1])
    df = load_dataset(cfg.dataset, cfg.sheet).astype(str)
    x_cols, y_col, normalized_cols = select_columns(df, cfg)

    # Normalización de nombres en instancias
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
            print(f"[WARN] Instancia ignorada por columnas faltantes. Se requieren: {x_cols}")

    # Verificación de instancias válidas
    if not norm_instances:
        print("[ERROR] No se detectaron instancias válidas en el archivo de configuración.")
        print(f"        Asegúrate de definir al menos un bloque INSTANCE: con todas las variables X requeridas: {x_cols}")
        sys.exit(1)

    # Preprocesamiento numérico
    used_cols = [y_col] + x_cols
    df_num, dropped = ensure_numeric_subset(df, used_cols)
    if dropped > 0:
        print(f"[INFO] Filas descartadas por NaN/no-numéricas en {used_cols}: {dropped}")

    # Genera bloque completo para todas las instancias
    print("=== Ejecutando regresión lineal ===")
    all_block = build_full_report_block(norm_instances, df_num, y_col, x_cols)

    # Generación del PDF único
    if cfg.report_path:
        out = cfg.report_path.replace(".pdf", "_all.pdf")
        render_all_instances_pdf(out, all_block)
        print(f"[OK] Reporte generado con todas las instancias: {out}")

    print("[OK] Ejecución completada.")


if __name__ == "__main__":
    main()

