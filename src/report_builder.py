#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report_builder.py
 ------------------------------------------------------------
 Descripción:
 Genera el bloque LaTeX completo con todos los pasos (1–4)
 de cada instancia válida en un único documento PDF.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import pandas as pd
from .linear_regression import run_linear_regression, predict
from .report_latex import dataset_preview_table, _matrix_to_latex, _fmt_number


def build_full_report_block(
    instances: List[Dict[str, float]],
    df_num: pd.DataFrame,
    y_col: str,
    x_cols: List[str]
) -> str:
    """
    Construye un bloque LaTeX que agrupa los pasos (1–4)
    de todas las instancias válidas.
    """
    lines: List[str] = []

    for idx, inst in enumerate(instances, 1):
        res = run_linear_regression(df_num, y_col, x_cols)

        # Bloques base
        X_block = _matrix_to_latex(res.X)
        y_block = _matrix_to_latex(res.y.reshape(-1, 1))
        XtX_block = _matrix_to_latex(res.XtX)
        Xty_block = _matrix_to_latex(res.Xty.reshape(-1, 1))
        beta_vec = " \\\\ ".join(_fmt_number(b) for b in res.beta)
        r2_str = _fmt_number(res.r2) if res.r2 is not None else "—"
        dataset_table = dataset_preview_table(df_num)

        # Tabla de coeficientes
        beta_lines = [f"$\\beta_0$ & {_fmt_number(res.beta[0])} \\\\"]
        for j, name in enumerate(x_cols, start=1):
            beta_lines.append(f"$\\beta_{j}$ ({name}) & {_fmt_number(res.beta[j])} \\\\")
        beta_table = "\n".join(beta_lines)

        # Sustitución de valores X → Y
        xs = [float(inst[name]) for name in x_cols]
        parts_eq = [f"{res.beta[0]:.6f}"]
        parts_sub = [f"{res.beta[0]:.6f}"]
        for j, name in enumerate(x_cols, start=1):
            val = float(inst[name])
            parts_eq.append(f"+ {res.beta[j]:.6f}\\,x_{{{j}}}")
            parts_sub.append(f"+ {res.beta[j]:.6f}\\cdot {val:.6f}")
        yhat = predict(res.beta, xs)

        # Construcción del bloque
        lines.append(f"\\section*{{Instancia {idx}}}")
        lines.append("\\subsection*{Resumen del dataset}")
        lines.append(dataset_table)
        lines.append(r"\subsection*{Paso 1: Modelo lineal}")
        lines.append(r"\[ y = \beta_0 + \beta_1 x_1 + \cdots + \beta_m x_m \]")
        lines.append(r"\subsection*{Paso 2: Función objetivo}")
        lines.append(r"\[ S(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - (\beta_0 + \sum_{j=1}^m \beta_j x_{ij}))^2 \]")
        lines.append(r"\subsection*{Paso 3: Ecuaciones normales}")
        lines.append(fr"""
\[
\boldsymbol{{\beta}} = (\mathbf{{X}}^\top \mathbf{{X}})^{{-1}}\mathbf{{X}}^\top \mathbf{{y}}
\]
\[
\mathbf{{X}} =
\begin{{bmatrix}}
{X_block}
\end{{bmatrix}},\quad
\mathbf{{y}} =
\begin{{bmatrix}}
{y_block}
\end{{bmatrix}}
\]
\[
\mathbf{{X}}^\top\mathbf{{X}} =
\begin{{bmatrix}}
{XtX_block}
\end{{bmatrix}},\quad
\mathbf{{X}}^\top\mathbf{{y}} =
\begin{{bmatrix}}
{Xty_block}
\end{{bmatrix}}
\]
\[
\boldsymbol{{\beta}} =
\begin{{bmatrix}}
{beta_vec}
\end{{bmatrix}}
\]
\begin{{tabular}}{{l r}}
\toprule
Parámetro & Valor \\
\midrule
{beta_table}
\bottomrule
\end{{tabular}}
""")

        lines.append(r"\subsection*{Paso 4: Sustitución de X en Y (predicción)}")
        lines.append("Atributos: " + ", ".join(f"{k}={v}" for k, v in inst.items()))
        lines.append(r"\[ \hat{y} = " + " ".join(parts_eq) + r" \]")
        lines.append(r"\[ \hat{y} = " + " ".join(parts_sub) + f" = \\textbf{{{yhat:.6f}}} \\]")
        lines.append(f"\\textbf{{Coeficiente de determinación:}} $R^2 = {r2_str}$")
        lines.append("\\newpage")

    return "\n".join(lines)

