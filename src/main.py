#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/main.py
 ------------------------------------------------------------
 Versión extendida con soporte para múltiples datasets definidos
 en un mismo input.txt, y detección automática del caso simple
 (1 variable X) o múltiple (≥2 variables).
 ------------------------------------------------------------
"""

from __future__ import annotations
import sys
import unicodedata
from typing import Dict, List
import pandas as pd
from pathlib import Path

from src.core.config import Config
from src.core.data_extractor.loader import load_dataset
from src.core.data_extractor.preprocess_regression import ensure_numeric_subset
from src.regression.linear_regression import run_linear_regression
from src.regression.simple_regression import run_simple_regression
from src.report.report_builder import build_full_report_block
from src.report.report_latex import render_all_instances_pdf


# ------------------------------------------------------------
def normalize_str(s: str) -> str:
    """Normaliza cadenas (minúsculas, sin tildes ni espacios extra)."""
    s = str(s).strip().lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# ------------------------------------------------------------
def parse_multiple_datasets(path: str) -> List[Config]:
    """
    Divide input.txt en múltiples secciones de dataset,
    retornando una lista de objetos Config independientes.
    Evita crear bloques vacíos o sin DATASET.
    """
    sections: List[Path] = []
    content = Path(path).read_text(encoding="utf-8").splitlines()

    temp, idx = [], 1
    for line in content:
        if line.strip().upper().startswith("DATASET"):
            # Guarda bloque anterior si tenía un DATASET válido
            if temp and any("DATASET" in l.upper() for l in temp):
                subfile = Path(f"/tmp/config_part_{idx}.txt")
                subfile.write_text("\n".join(temp), encoding="utf-8")
                sections.append(subfile)
                temp, idx = [], idx + 1
        temp.append(line)

    # Último bloque (solo si contiene un DATASET válido)
    if temp and any("DATASET" in l.upper() for l in temp):
        subfile = Path(f"/tmp/config_part_{idx}.txt")
        subfile.write_text("\n".join(temp), encoding="utf-8")
        sections.append(subfile)

    cfgs = [Config(str(p)) for p in sections]
    return cfgs


# ------------------------------------------------------------
def select_columns(df: pd.DataFrame, cfg: Config) -> tuple[list[str], str, dict[str, str]]:
    """Selecciona variables dependientes e independientes según input.txt."""
    cols = list(df.columns)
    normalized_cols = {normalize_str(c): c for c in cols}

    # Variable dependiente
    y_key = cfg.target_column or getattr(cfg, "dependent_variable", None)
    if not y_key:
        y = cols[-1]
    else:
        nk = normalize_str(y_key)
        if nk not in normalized_cols:
            raise ValueError(f"La columna dependiente '{y_key}' no existe.\nColumnas: {cols}")
        y = normalized_cols[nk]

    # Variables independientes
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
        raise ValueError("Se requieren ≥ 2 columnas (al menos 1 X + 1 Y).")

    return x_cols, y, normalized_cols


# ------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        print("Uso: python -m src.main input.txt")
        sys.exit(1)

    # === Parseo de múltiples datasets ===
    configs = parse_multiple_datasets(sys.argv[1])
    if not configs:
        print("[ERROR] No se detectó ningún bloque de configuración con DATASET=")
        sys.exit(1)

    all_instances_blocks: List[str] = []
    instance_counter = 1

    for cfg in configs:
        print(f"=== Procesando dataset: {cfg.dataset} ===")

        # Verifica que el dataset sea válido
        if not cfg.dataset or not Path(cfg.dataset).exists():
            print(f"[WARN] Dataset inválido o no encontrado: {cfg.dataset}")
            continue

        df = load_dataset(cfg.dataset, cfg.sheet).astype(str)
        x_cols, y_col, normalized_cols = select_columns(df, cfg)

        # Preprocesamiento numérico
        used_cols = [y_col] + x_cols
        df_num, dropped = ensure_numeric_subset(df, used_cols)
        if dropped > 0:
            print(f"[INFO] Filas descartadas por NaN/no-numéricas: {dropped}")

        # Normalización de instancias
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
                print(f"[WARN] Instancia ignorada (faltan columnas): {x_cols}")

        if not norm_instances:
            print("[WARN] No hay instancias válidas para este dataset.")
            continue

        # === Detección de tipo de regresión ===
        if len(x_cols) == 1:
            print(f"[INFO] Usando modo regresión simple ({x_cols[0]})")
            block = run_simple_regression(df_num, y_col, x_cols[0], norm_instances, instance_counter)
        else:
            print(f"[INFO] Usando modo regresión múltiple ({len(x_cols)} variables)")
            block = build_full_report_block(norm_instances, df_num, y_col, x_cols)

        all_instances_blocks.append(block)
        instance_counter += len(norm_instances)

    # === Generación del PDF ===
    if not all_instances_blocks:
        print("[ERROR] No se generaron bloques de reporte.")
        sys.exit(1)

    all_text = "\n".join(all_instances_blocks)
    out_path = configs[0].report_path or "output/reporte.pdf"
    out_path = out_path.replace(".pdf", "_all.pdf")

    render_all_instances_pdf(out_path, all_text)
    print(f"[OK] Reporte generado con todas las instancias: {out_path}")


if __name__ == "__main__":
    main()

