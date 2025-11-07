#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report_latex.py
 ------------------------------------------------------------
 Descripción:
 Genera un reporte en LaTeX para el proceso de REGRESIÓN LINEAL,
 mostrando explícitamente los 4 pasos del PDF:

  1) Modelo: y = β₀ + β₁ x₁ + ... + βₘ xₘ
  2) Función objetivo S(β) y condiciones de óptimo (derivadas = 0)
  3) Ecuaciones normales y solución cerrada: β = (XᵀX)⁻¹ Xᵀ y
  4) Sustitución de X en Y: ŷ = β₀ + β₁ x₁ + ... + βₘ xₘ

 Incluye:
  - Resumen de dataset.
  - Tablas (vista previa).
  - Matrices X, y, XᵀX, Xᵀy (con truncamiento si son grandes).
  - Coeficientes β con nombres de variables.
  - Predicciones de instancias (si se proporcionan en input.txt).
 ------------------------------------------------------------
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# === Plantilla LaTeX ===
latex_template = r"""
\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage[spanish]{babel}
\decimalpoint
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{breqn}
\usepackage{microtype}
\usepackage{ragged2e}
\usepackage{siunitx}
\sisetup{output-decimal-marker = {.}}

\begin{document}
\RaggedRight

\section*{Regresión Lineal (Mínimos Cuadrados)}

\subsection*{Resumen del dataset}
Filas: %(rows)d, Columnas: %(cols)d\\
Variable dependiente ($y$): \texttt{%(y_col)s}\\
Variables independientes ($\mathbf{x}$): \texttt{%(x_list)s}

\subsection*{Vista previa del dataset}
%(dataset_table)s

\section*{Paso 1: Modelo lineal}
Sea el modelo:
\[
y = \beta_0 + \beta_1 x_1 + \cdots + \beta_m x_m
\]
En forma matricial:
\[
\mathbf{y} = \mathbf{X}\,\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \quad
\mathbf{X} =
\begin{bmatrix}
1 & x_{11} & \cdots & x_{1m}\\
\vdots & \vdots & \ddots & \vdots\\
1 & x_{n1} & \cdots & x_{nm}
\end{bmatrix}, \ \
\boldsymbol{\beta} =
\begin{bmatrix}
\beta_0\\ \beta_1\\ \vdots\\ \beta_m
\end{bmatrix}
\]

\section*{Paso 2: Función objetivo}
Minimizamos la suma de cuadrados
\[
S(\boldsymbol{\beta}) = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^m \beta_j x_{ij}\right)\right)^2
\]
y las condiciones de primer orden (derivadas igual a cero):
\[
\frac{\partial S}{\partial \beta_j} = 0,\ \ \forall j = 0,\ldots,m.
\]

\section*{Paso 3: Ecuaciones normales y solución}
Las ecuaciones normales equivalen a:
\[
\mathbf{X}^\top \mathbf{X}\,\boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}
\]
Entonces, la solución cerrada es:
\[
\boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
\]
En la práctica, utilizamos pseudoinversa si $\mathbf{X}^\top\mathbf{X}$ es singular.

\subsection*{Matrices utilizadas}
\[
\mathbf{X} =
\begin{bmatrix}
%(X_matrix)s
\end{bmatrix}
,\quad
\mathbf{y} =
\begin{bmatrix}
%(y_vector)s
\end{bmatrix}
\]

\[
\mathbf{X}^\top\mathbf{X} =
\begin{bmatrix}
%(XtX)s
\end{bmatrix}
,\quad
\mathbf{X}^\top\mathbf{y} =
\begin{bmatrix}
%(Xty)s
\end{bmatrix}
\]

\subsection*{Coeficientes estimados}
\[
\boldsymbol{\beta} =
\begin{bmatrix}
\beta_0\\ \beta_1\\ \vdots\\ \beta_m
\end{bmatrix}
=
\begin{bmatrix}
%(beta_vector)s
\end{bmatrix}
\]
\begin{tabular}{l r}
\toprule
Parámetro & Valor \\
\midrule
%(beta_table)s
\bottomrule
\end{tabular}

\subsection*{Bondad de ajuste}
Coeficiente de determinación: $R^2 = %(r2)s$

\section*{Paso 4: Sustitución de $\mathbf{x}$ en $y$ (predicciones)}
%(predictions)s

\end{document}
"""

def _fmt_number(x: float) -> str:
    try:
        return f"{float(x):.6f}".replace(",", ".")
    except Exception:
        return str(x)

def _matrix_to_latex(M: np.ndarray, max_rows: int = 10, max_cols: int = 6) -> str:
    """
    Convierte una matriz en bloques ‘a_{ij}’ LaTeX con truncamiento
    si excede los límites para evitar páginas gigantes.
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
    return "\n".join(lines)

def dataset_preview_table(df: pd.DataFrame, max_rows: int = 15, max_cols: int = 8) -> str:
    """
    Genera una tabla compacta de vista previa del dataset.
    """
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
        lines.append("\\\\[0.3em]\\textit{Nota: tabla truncada para visualización.}")
    return "\n".join(lines)

def render_pdf(
    out_pdf: str,
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    beta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    XtX: np.ndarray,
    Xty: np.ndarray,
    r2: Optional[float],
    predictions_block: str
):
    """
    Genera el .tex y compila a PDF (xelatex/pdflatex).
    """
    out_pdf_path = Path(out_pdf)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = out_pdf_path.with_suffix(".tex")

    # Bloques formateados
    dataset_table = dataset_preview_table(df)
    X_block = _matrix_to_latex(X)
    y_block = _matrix_to_latex(y.reshape(-1, 1))
    XtX_block = _matrix_to_latex(XtX)
    Xty_block = _matrix_to_latex(Xty.reshape(-1, 1))
    beta_vec = " \\\\ ".join(_fmt_number(b) for b in beta)
    r2_str = _fmt_number(r2) if r2 is not None else "—"

    # Tabla parámetros nombrados
    beta_table_lines = [f"$\\beta_0$ & {_fmt_number(beta[0])} \\\\"]
    for j, name in enumerate(x_cols, start=1):
        beta_table_lines.append(f"$\\beta_{j}$ ({name}) & {_fmt_number(beta[j])} \\\\")
    beta_table = "\n".join(beta_table_lines)

    tex_content = latex_template % {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "y_col": y_col,
        "x_list": ", ".join(x_cols),
        "dataset_table": dataset_table,
        "X_matrix": X_block,
        "y_vector": y_block,
        "XtX": XtX_block,
        "Xty": Xty_block,
        "beta_vector": beta_vec,
        "beta_table": beta_table,
        "r2": r2_str,
        "predictions": predictions_block if predictions_block else r"\textit{No se proporcionaron instancias.}"
    }

    tex_path.write_text(tex_content, encoding="utf-8")

    for binname in ("xelatex", "pdflatex"):
        try:
            subprocess.run(
                [binname, "-interaction=nonstopmode", tex_path.name],
                cwd=tex_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return
        except Exception:
            continue

    print(f"[WARN] No se pudo compilar PDF, se dejó el archivo TEX en: {tex_path}")
# ---------------------------------------------------------------------------

def render_all_instances_pdf(out_pdf: str, latex_block: str):
    """
    Genera un único PDF con todas las instancias concatenadas,
    usando el bloque completo LaTeX ya construido.
    """
    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = out_path.with_suffix(".tex")

    tex_content = r"""
\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage[spanish]{babel}
\usepackage[utf8]{inputenc}
\usepackage{booktabs, amsmath, breqn, microtype, siunitx, ragged2e}
\sisetup{output-decimal-marker = {.}}
\begin{document}
\RaggedRight
""" + latex_block + "\n\\end{document}"

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

def render_all_instances_pdf(out_pdf: str, latex_block: str):
    """
    Genera un único PDF con todas las instancias concatenadas,
    usando el bloque completo LaTeX ya construido.
    """
    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = out_path.with_suffix(".tex")

    tex_content = r"""
\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage[spanish]{babel}
\usepackage[utf8]{inputenc}
\usepackage{booktabs, amsmath, breqn, microtype, siunitx, ragged2e}
\sisetup{output-decimal-marker = {.}}
\begin{document}
\RaggedRight
""" + latex_block + "\n\\end{document}"

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
            print(f"[OK] Reporte PDF generado: {out_pdf}")
            return
        except Exception:
            continue

    print(f"[WARN] No se pudo compilar el PDF; archivo TEX disponible en: {tex_path}")

