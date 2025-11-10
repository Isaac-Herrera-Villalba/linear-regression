#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_builder.py
 ------------------------------------------------------------
 Versión final académica:
  - Matrices centradas y con sus dimensiones (n×m, m×m, etc.).
  - Tamaño ajustado automáticamente con \resizebox{0.9\linewidth}{!}.
  - Ecuaciones del Paso 4 homogéneas, con saltos automáticos.
  - Soporte para múltiples instancias y puntos decimales correctos.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import pandas as pd
from src.regression.linear_regression import run_linear_regression, predict
from src.report.report_latex import dataset_preview_table, _matrix_to_latex, _fmt_number


# ======================================================================
# Utilidad: división automática de ecuaciones largas
# ======================================================================
def _split_equation(parts: List[str], width: int = 90) -> str:
    """
    Divide ecuaciones largas en líneas para multline* en LaTeX.
    """
    lines, current, length = [], [], 0
    for p in parts:
        if length + len(p) > width:
            lines.append(" ".join(current))
            current, length = [p], len(p)
        else:
            current.append(p)
            length += len(p)
    if current:
        lines.append(" ".join(current))
    return " \\\\\n".join(lines)


# ======================================================================
# Construcción del bloque completo
# ======================================================================
def build_full_report_block(
    instances: List[Dict[str, float]],
    df_num: pd.DataFrame,
    y_col: str,
    x_cols: List[str]
) -> str:
    lines: List[str] = []

    for idx, inst in enumerate(instances, 1):
        res = run_linear_regression(df_num, y_col, x_cols)

        n, m = res.X.shape
        r2_str = _fmt_number(res.r2) if res.r2 is not None else "—"
        dataset_table = dataset_preview_table(df_num)

        # === MATRICES (formato LaTeX con dimensiones) ===
        X_block   = _matrix_to_latex(res.X)
        y_block   = _matrix_to_latex(res.y.reshape(-1, 1))
        XtX_block = _matrix_to_latex(res.XtX)
        Xty_block = _matrix_to_latex(res.Xty.reshape(-1, 1))

        beta_vec  = " \\\\ ".join(_fmt_number(b) for b in res.beta)

        # === TABLA DE COEFICIENTES ===
        beta_lines = [f"$\\beta_0$ & {_fmt_number(res.beta[0])} \\\\"]
        for j, name in enumerate(x_cols, start=1):
            beta_lines.append(f"$\\beta_{j}$ ({name}) & {_fmt_number(res.beta[j])} \\\\")
        beta_table = "\n".join(beta_lines)

        # === SUSTITUCIÓN DE VALORES ===
        xs = [float(inst[name]) for name in x_cols]
        parts_eq, parts_sub = [], []
        parts_eq.append(f"{_fmt_number(res.beta[0])}")
        parts_sub.append(f"{_fmt_number(res.beta[0])}")

        for j, name in enumerate(x_cols, start=1):
            val = float(inst[name])
            sign = "+" if res.beta[j] >= 0 else "-"
            parts_eq.append(f"{sign} {_fmt_number(abs(res.beta[j]))}\\,x_{{{j}}}")
            parts_sub.append(f"{sign} {_fmt_number(abs(res.beta[j]))}({_fmt_number(val)})")

        yhat = predict(res.beta, xs)
        eq_line  = _split_equation(parts_eq)
        sub_line = _split_equation(parts_sub)

        # === SECCIONES DEL REPORTE ===
        lines.append(f"\\section*{{Instancia {idx}}}")
        lines.append("\\subsection*{Resumen del dataset}")
        lines.append(dataset_table)

        # Paso 1
        lines.append(r"\subsection*{Paso 1: Modelo lineal}")
        lines.append(r"\[ y = \beta_0 + \beta_1 x_1 + \cdots + \beta_m x_m \]")

        # Paso 2
        lines.append(r"\subsection*{Paso 2: Función objetivo}")
        lines.append(r"\[ S(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - (\beta_0 + \sum_{j=1}^m \beta_j x_{ij}))^2 \]")

        # Paso 3
        lines.append(r"\subsection*{Paso 3: Ecuaciones normales}")
        lines.append(r"\[ \boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} \]")

        # Matrices una debajo de otra con sus dimensiones
        lines.append(r"\[ \textbf{Matriz } \mathbf{X} \in \mathbb{R}^{%d\times%d} \]" % (n, m))
        lines.append(r"\[ \resizebox{0.9\linewidth}{!}{$ \mathbf{X} = " + X_block + r"$} \]")

        lines.append(r"\[ \textbf{Vector } \mathbf{y} \in \mathbb{R}^{%d\times1} \]" % (n,))
        lines.append(r"\[ \resizebox{0.3\linewidth}{!}{$ \mathbf{y} = " + y_block + r"$} \]")

        lines.append(r"\[ \textbf{Matriz } \mathbf{X}^\top\mathbf{X} \in \mathbb{R}^{%d\times%d} \]" % (m, m))
        lines.append(r"\[ \resizebox{0.9\linewidth}{!}{$ \mathbf{X}^\top\mathbf{X} = " + XtX_block + r"$} \]")

        lines.append(r"\[ \textbf{Matriz } \mathbf{X}^\top\mathbf{y} \in \mathbb{R}^{%d\times1} \]" % (m,))
        lines.append(r"\[ \resizebox{0.45\linewidth}{!}{$ \mathbf{X}^\top\mathbf{y} = " + Xty_block + r"$} \]")

        # Vector β
        lines.append(r"\[ \boldsymbol{\beta} = \begin{bmatrix}" + beta_vec + r"\end{bmatrix} \]")

        # Tabla β
        lines.append(r"\begin{tabular}{l r}")
        lines.append(r"\toprule")
        lines.append(r"Parámetro & Valor \\")
        lines.append(r"\midrule")
        lines.append(beta_table)
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        # Paso 4
        lines.append(r"\subsection*{Paso 4: Sustitución de X en Y (predicción)}")
        lines.append("Atributos: " + ", ".join(f"{k}={v}" for k, v in inst.items()))

        # Ecuación simbólica
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + eq_line)
        lines.append(r"\end{multline*}")

        # Ecuación numérica (idéntico formato)
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + sub_line + f" = \\textbf{{{_fmt_number(yhat)}}}")
        lines.append(r"\end{multline*}")

        # R²
        lines.append(f"\\textbf{{Coeficiente de determinación:}} $R^2 = {r2_str}$")
        lines.append("\\newpage")

    return "\n".join(lines)

