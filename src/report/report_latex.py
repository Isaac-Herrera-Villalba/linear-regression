#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_latex.py
 ------------------------------------------------------------
 Módulo de utilidades para generar y compilar reportes en LaTeX
 a partir de los resultados de regresión lineal.

 Incluye funciones para:
   - Construir matrices NumPy en formato LaTeX (bmatrix).
   - Formatear números con punto decimal forzado.
   - Crear tablas de vista previa de datasets.
   - Renderizar el documento final en PDF.

 El sistema emplea `graphicx` + `\resizebox` para escalar matrices,
 y `\allowdisplaybreaks` para permitir saltos de página dentro de
 expresiones matemáticas extensas.
 ------------------------------------------------------------
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd


# ======================================================================
# === PLANTILLAS LaTeX BASE ============================================
# ======================================================================

LATEX_PREAMBLE_ALL = r"""
\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage{graphicx}            % Ajuste de tamaño de matrices
\usepackage{booktabs}            % Estilo de tablas
\usepackage{amsmath}             % Entornos matemáticos
\allowdisplaybreaks              % Permite cortes de página entre ecuaciones
\usepackage{breqn}               % Ecuaciones largas
\usepackage{microtype}           % Microajuste tipográfico
\usepackage{ragged2e}
\usepackage{siunitx}             % Formateo numérico
\sisetup{
  output-decimal-marker = {.},
  group-separator = {\,},
  detect-all,
  locale = US
}
\begin{document}
\RaggedRight
"""

LATEX_POSTAMBLE_ALL = r"""
\end{document}
"""


# ======================================================================
# === FORMATEADORES ====================================================
# ======================================================================

def _fmt_number(x: float) -> str:
    """
    Formatea un número a 6 decimales con punto decimal fijo.
    Convierte automáticamente a string en caso de error.
    """
    try:
        s = f"{float(x):.6f}"
    except Exception:
        s = str(x)
    return s.replace(",", ".").replace("−", "-")


def _matrix_to_latex(M: np.ndarray, max_rows: int = 12, max_cols: int = 12) -> str:
    """
    Convierte una matriz NumPy en código LaTeX (entorno bmatrix).

    Parámetros
    ----------
    M : np.ndarray
        Matriz a convertir.
    max_rows : int
        Número máximo de filas visibles.
    max_cols : int
        Número máximo de columnas visibles.

    Retorna
    -------
    str
        Cadena LaTeX con el entorno bmatrix.
    """
    r, c = M.shape
    rows = min(r, max_rows)
    cols = min(c, max_cols)

    lines = []
    for i in range(rows):
        vals = " & ".join(_fmt_number(M[i, j]) for j in range(cols))
        if c > cols:
            vals += " & \\cdots"
        lines.append(vals + r" \\")
    if r > rows:
        lines.append(r"\vdots")

    return "\\begin{bmatrix}\n" + "\n".join(lines) + "\n\\end{bmatrix}"


def dataset_preview_table(df: pd.DataFrame, max_rows: int = 15, max_cols: int = 8) -> str:
    """
    Genera una tabla de vista previa del dataset (truncada si excede los límites).

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset de entrada.
    max_rows : int
        Número máximo de filas visibles.
    max_cols : int
        Número máximo de columnas visibles.

    Retorna
    -------
    str
        Tabla LaTeX con encabezado, filas y nota de truncamiento si aplica.
    """
    rows, cols = df.shape
    truncated = rows > max_rows or cols > max_cols

    df_disp = df.iloc[:max_rows, :max_cols].copy()
    df_disp.columns = [str(c).replace("_", "\\_") for c in df_disp.columns]

    header_fmt = " ".join(["l"] * len(df_disp.columns))
    lines = [
        f"\\textit{{Dimensiones del dataset:}} ${rows}\\,\\text{{filas}} \\times {cols}\\,\\text{{columnas}}$\\\\[0.3em]",
        "\\begin{tabular}{" + header_fmt + "}",
        "\\toprule",
        " & ".join(df_disp.columns) + " \\\\",
        "\\midrule",
    ]

    for _, row in df_disp.iterrows():
        vals = [str(v).replace("_", "\\_") for v in row.values]
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    if truncated:
        lines.append(
            f"\\\\[0.3em]\\textit{{Nota: el dataset mostrado fue truncado a "
            f"{max_rows} filas y {max_cols} columnas. "
            f"El dataset completo contiene {rows} filas y {cols} columnas.}}"
            "\\\\[0.3em]\\textit{Consulte el archivo original para ver la tabla completa.}"
        )

    return "\n".join(lines)


# ======================================================================
# === RENDERIZACIÓN DE PDF =============================================
# ======================================================================

def render_all_instances_pdf(out_pdf: str, latex_block: str):
    """
    Genera un único documento PDF con todas las instancias concatenadas.

    Parámetros
    ----------
    out_pdf : str
        Ruta de salida del PDF final.
    latex_block : str
        Bloque LaTeX con el contenido de todas las secciones.

    Detalles
    ---------
    - Crea automáticamente el archivo .tex correspondiente.
    - Intenta compilar con `xelatex` y, en caso de fallo, con `pdflatex`.
    - Si ninguna compilación tiene éxito, se conserva el archivo .tex.
    """
    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = out_path.with_suffix(".tex")

    tex_content = LATEX_PREAMBLE_ALL + latex_block + LATEX_POSTAMBLE_ALL
    tex_path.write_text(tex_content, encoding="utf-8")

    for compiler in ("xelatex", "pdflatex"):
        try:
            subprocess.run(
                [compiler, "-interaction=nonstopmode", tex_path.name],
                cwd=tex_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(f"[OK] Reporte PDF generado con {compiler}: {out_pdf}")
            return
        except Exception:
            continue

    print(f"[WARN] No se pudo compilar el PDF; archivo TEX disponible en: {tex_path}")

