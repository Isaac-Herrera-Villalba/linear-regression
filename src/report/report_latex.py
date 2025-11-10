#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_latex.py
 ------------------------------------------------------------
 - Puntos decimales forzados.
 - Matrices con corchetes [ ] (bmatrix).
 - graphicx + \resizebox para que NO se corten matrices anchas.
 - \allowdisplaybreaks para permitir cortes entre fórmulas si es necesario.
 ------------------------------------------------------------
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

# === Plantilla LaTeX para reportes por secciones (usada por render_all_instances_pdf) ===
LATEX_PREAMBLE_ALL = r"""
\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage{graphicx}            % <-- para \resizebox
\usepackage{booktabs}
\usepackage{amsmath}
\allowdisplaybreaks              % <-- permite cortes de página entre ecuaciones
\usepackage{breqn}
\usepackage{microtype}
\usepackage{ragged2e}
\usepackage{siunitx}
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

# === Formateadores ===
def _fmt_number(x: float) -> str:
    """6 decimales, fuerza punto decimal."""
    try:
        s = f"{float(x):.6f}"
    except Exception:
        s = str(x)
    return s.replace(",", ".").replace("−", "-")

def _matrix_to_latex(M: np.ndarray, max_rows: int = 12, max_cols: int = 12) -> str:
    """
    Matriz NumPy → LaTeX (bmatrix) con filas separadas.
    Se deja completa; el ajuste de ancho lo hará \resizebox en el builder.
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
    """Tabla de vista previa (truncada) del dataset."""
    rows, cols = df.shape
    truncated = rows > max_rows or cols > max_cols
    df_disp = df.iloc[:max_rows, :max_cols].copy()
    df_disp.columns = [str(c).replace("_", "\\_") for c in df_disp.columns]
    header_fmt = " ".join(["l"] * len(df_disp.columns))
    lines = [
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
        lines.append("\\\\[0.3em]\\textit{Nota: el dataset es demasiado grande, consulte el archivo original.}")
    return "\n".join(lines)

def render_all_instances_pdf(out_pdf: str, latex_block: str):
    """
    Genera un único PDF con todas las instancias concatenadas.
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

