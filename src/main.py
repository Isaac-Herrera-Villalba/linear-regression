#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/main.py
 ------------------------------------------------------------
 Punto de entrada principal del sistema de Regresión Lineal.

 - Lee y procesa el archivo de configuración `input.txt`.
 - Carga múltiples datasets (uno por bloque `DATASET=`).
 - Ejecuta el análisis de regresión lineal simple o múltiple.
 - Genera un único reporte PDF unificado con todas las instancias.

 Estructura del flujo:
   1. Leer configuración global.
   2. Por cada bloque de dataset:
        a) Cargar datos y hoja especificada.
        b) Preprocesar columnas numéricas.
        c) Ejecutar regresión y generar bloque LaTeX.
   3. Concatenar todos los bloques y generar un único PDF.
 ------------------------------------------------------------
"""

from __future__ import annotations
import sys
from pathlib import Path

from src.core.config import Config
from src.core.data_extractor.loader import load_dataset
from src.core.data_extractor.preprocess_regression import ensure_numeric_subset
from src.report.report_builder import build_full_report_block
from src.report.report_latex import render_all_instances_pdf


# ============================================================
# === PUNTO DE ENTRADA PRINCIPAL ==============================
# ============================================================

def main():
    """
    Ejecuta el flujo completo del sistema de Regresión Lineal.
    """
    cfg_path = Path("input.txt")
    if not cfg_path.exists():
        print(f"[ERROR] No se encontró el archivo de configuración: {cfg_path}")
        sys.exit(1)

    # === Leer configuración ===
    config = Config(str(cfg_path))
    blocks = config.get_blocks()

    if not blocks:
        print("[ERROR] No se detectaron bloques válidos en el archivo de configuración.")
        sys.exit(1)

    # === Variables acumulativas ===
    all_latex_blocks = ""
    global_index = 1
    final_report_path = None

    print(f"[INFO] Se detectaron {len(blocks)} bloques de dataset en {cfg_path.name}")

    # === Procesar cada bloque (DATASET + INSTANCES) ===
    for i, block in enumerate(blocks, 1):
        kv = block["KV"]
        instances = block["INSTANCES"]

        dataset_path = kv.get("DATASET")
        sheet_name = kv.get("SHEET")
        y_col = kv.get("DEPENDENT_VARIABLE")
        x_cols_raw = kv.get("INDEPENDENT_VARIABLES")
        report_path = kv.get("REPORT", "output/reporte.pdf")
        final_report_path = report_path

        if not dataset_path or not y_col or not x_cols_raw:
            print(f"[WARN] Bloque {i}: configuración incompleta, se omite.")
            continue

        x_cols = [x.strip() for x in x_cols_raw.split(",") if x.strip()]

        print(f"\n=== Procesando dataset {i}/{len(blocks)} ===")
        print(f"[INFO] Archivo: {dataset_path}")
        print(f"[INFO] Hoja: {sheet_name}")
        print(f"[INFO] Variables: Y={y_col}, X={x_cols}")

        # === Cargar dataset ===
        try:
            df = load_dataset(dataset_path, sheet=sheet_name)
        except Exception as e:
            print(f"[ERROR] Fallo al cargar dataset '{dataset_path}': {e}")
            continue

        # === Preprocesamiento numérico ===
        used_cols = [y_col] + x_cols
        try:
            df_num, dropped = ensure_numeric_subset(df, used_cols)
        except Exception as e:
            print(f"[ERROR] Fallo al preprocesar datos: {e}")
            continue

        if df_num.empty:
            print(f"[WARN] Dataset vacío tras limpieza, se omite bloque {i}.")
            continue

        if dropped > 0:
            print(f"[INFO] Filas eliminadas durante preprocesamiento: {dropped}")

        # === Construcción del bloque LaTeX ===
        try:
            block_tex = build_full_report_block(
                instances=instances,
                df_num=df_num,
                y_col=y_col,
                x_cols=x_cols,
                start_index=global_index,  # contador global
            )
            all_latex_blocks += block_tex + "\n"
            global_index += len(instances)
            print(f"[OK] Bloque {i} procesado correctamente ({len(instances)} instancia(s)).")
        except Exception as e:
            print(f"[ERROR] Error generando bloque {i}: {e}")
            continue

    # === Compilar reporte unificado ===
    if all_latex_blocks.strip():
        print("\n=== Generando reporte PDF unificado ===")
        try:
            render_all_instances_pdf(final_report_path, all_latex_blocks)
        except Exception as e:
            print(f"[ERROR] No se pudo generar el PDF: {e}")
    else:
        print("[WARN] No se generaron bloques válidos; no hay contenido para el PDF final.")


# ============================================================
# === EJECUCIÓN ==============================================
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ejecución cancelada por el usuario.")
        sys.exit(130)

