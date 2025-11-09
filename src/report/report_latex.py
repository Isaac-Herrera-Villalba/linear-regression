#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_latex.py
 ------------------------------------------------------------
 Genera un reporte en LaTeX para el proceso de REGRESIÓN LINEAL,
 mostrando los 4 pasos, con matrices correctamente formateadas:
  - Puntos decimales en vez de comas.
  - Saltos de línea entre filas.
  - Aviso si el dataset es demasiado grande.
 ------------------------------------------------------------
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

# === Plantilla LaTeX ===
latex_template = r"""
\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}

% --- Idioma: español, pero sin reemplazar los símbolos decimales ---
\usepackage[spanish,es-noshorthands]{babel}

% --- Codificación y tipografía ---
\usepackage[utf8]{inputenc}
\usepackage{microtype}
\usepackage{ragged2e}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{breqn}

% --- Configuración de siunitx (números y puntos decimales) ---
\usepackage{siunitx}
\sisetup{
  output-decimal-marker = {.},   % fuerza el uso del punto como separador decimal
  group-separator = {\,},        % separador de miles (espacio fino)
  group-minimum-digits = 4,      % agrupa cada 4 dígitos opcionalmente
  detect-all,
  locale = US                    % fuerza configuración numérica de EE.UU.
}

\begin{document}
\RaggedRight

\section*{Regresión Lineal (Mínimos Cuadrados Ordinarios)}

\textbf{Número de filas:} %(rows)d\\
\textbf{Número de columnas:} %(cols)d\\[0.3em]
%(dataset_notice)s

\textbf{Variable dependiente (Y):} \texttt{%(y_col)s}\\
\textbf{Variables independientes (X):} \texttt{%(x_list)s}\\

\subsection*{Vista previa del dataset}
%(dataset_table)s

\section*{Paso 1: Modelo lineal}
\[
y = \beta_0 + \beta_1 x_1 + \cdots + \beta_m x_m
\]

\section*{Paso 2: Función objetivo}
\[
S(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - (\beta_0 + \sum_{j=1}^m \beta_j x_{ij}))^2
\]

\section*{Paso 3: Ecuaciones normales}
\[
\boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
\]

\renewcommand{\arraystretch}{1.3}
\subsection*{Matrices utilizadas}
\[
\mathbf{X} =
\begin{bmatrix}
%(X_matrix)s
\end{bmatrix},\quad
\mathbf{y} =
\begin{bmatrix}
%(y_vector)s
\end{bmatrix}
\]

\[
\mathbf{X}^\top\mathbf{X} =
\begin{bmatrix}
%(XtX)s
\end{bmatrix},\quad
\mathbf{X}^\top\mathbf{y} =
\begin{bmatrix}
%(Xty)s
\end{bmatrix}
\]

\subsection*{Coeficientes β}
\[
\boldsymbol{\beta} =
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
$R^2 = %(r2)s$

\section*{Paso 4: Sustitución de X en Y (predicciones)}
%(predictions)s

\end{document}
"""

# === Formateadores ===
def _fmt_number(x: float) -> str:
    """
    Convierte cualquier número en cadena con 6 decimales y punto (.)
    forzado, sin depender de la configuración regional ni de LaTeX.
    """
    try:
        s = f"{float(x):.6f}"
    except Exception:
        s = str(x)
    # Fuerza el uso de punto decimal en cualquier circunstancia
    return s.replace(",", ".")

def _matrix_to_latex(M: np.ndarray, max_rows: int = 12, max_cols: int = 8) -> str:
    """
    Convierte una matriz NumPy a LaTeX con formato claro:
    - Puntos decimales.
    - Saltos entre filas.
    - Entorno bmatrix* para buena separación visual.
    """
    r, c = M.shape
    rows = min(r, max_rows)
    cols = min(c, max_cols)

    lines = []
    for i in range(rows):
        row_vals = " & ".join(_fmt_number(M[i, j]) for j in range(cols))
        if c > cols:
            row_vals += " & \\cdots"
        lines.append(row_vals + r" \\")
    if r > rows:
        lines.append(r"\\vdots")

    content = "\n".join(lines)
    return (
        "\\renewcommand{\\arraystretch}{1.25}%\n"
        "\\begin{bmatrix*}[r]\n"
        + content +
        "\n\\end{bmatrix*}"
    )

def dataset_preview_table(df: pd.DataFrame, max_rows: int = 15, max_cols: int = 8) -> str:
    """
    Genera una tabla de vista previa del dataset (truncada si es grande).
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
        lines.append("\\\\[0.3em]\\textit{Nota: el dataset es demasiado grande, consulte el archivo original.}")
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
    Genera y compila un PDF a partir de bloques LaTeX.
    """
    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = out_path.with_suffix(".tex")

    dataset_notice = ""
    if df.shape[0] > 20 or df.shape[1] > 8:
        dataset_notice = "\\textit{El dataset es demasiado grande, consulte el archivo original especificado en 'DATASET='}.\\\\[0.5em]"

    tex_content = latex_template % {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "dataset_notice": dataset_notice,
        "y_col": y_col,
        "x_list": ", ".join(x_cols),
        "dataset_table": dataset_preview_table(df),
        "X_matrix": _matrix_to_latex(X),
        "y_vector": _matrix_to_latex(y.reshape(-1, 1)),
        "XtX": _matrix_to_latex(XtX),
        "Xty": _matrix_to_latex(Xty.reshape(-1, 1)),
        "beta_vector": " \\\\ ".join(_fmt_number(b) for b in beta),
        "beta_table": "\n".join(
            [f"$\\beta_0$ & {_fmt_number(beta[0])} \\\\"] +
            [f"$\\beta_{j}$ ({name}) & {_fmt_number(beta[j])} \\\\" for j, name in enumerate(x_cols, start=1)]
        ),
        "r2": _fmt_number(r2) if r2 is not None else "—",
        "predictions": predictions_block if predictions_block else r"\textit{No se proporcionaron instancias.}"
    }

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

    print(f"[WARN] No se pudo compilar PDF; archivo TEX en: {tex_path}")


def render_all_instances_pdf(out_pdf: str, latex_block: str):
    """
    Genera un único PDF con todas las instancias concatenadas.
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

