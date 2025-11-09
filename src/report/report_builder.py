#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_builder.py
 ------------------------------------------------------------
 Genera el bloque LaTeX con todos los pasos (1–4) de cada instancia.
 Ajustes aplicados:
  - Matrices completas con saltos de línea (usa bmatrix* en report_latex).
  - Signos correctos (+ / -) en sustitución.
  - Puntos decimales forzados.
  - Paso 4 ahora usa entorno multline* para ecuaciones largas.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import pandas as pd
from src.regression.linear_regression import run_linear_regression, predict
from src.report.report_latex import dataset_preview_table, _matrix_to_latex, _fmt_number

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
        parts_eq = [f"{_fmt_number(res.beta[0])}"]
        parts_sub = [f"{_fmt_number(res.beta[0])}"]
        for j, name in enumerate(x_cols, start=1):
            val = float(inst[name])
            sign = "+" if res.beta[j] >= 0 else "-"
            parts_eq.append(f"{sign} {_fmt_number(abs(res.beta[j]))}\\,x_{{{j}}}")
            parts_sub.append(
                f"{sign} {_fmt_number(abs(res.beta[j]))}({_fmt_number(val)})"
            )
        yhat = predict(res.beta, xs)

        # --- Crea ecuaciones con saltos si son largas ---
        def format_multline(parts: list[str]) -> str:
            """
            Divide ecuaciones largas cada ~80 caracteres para hacer saltos en LaTeX.
            """
            chunks, current, length = [], [], 0
            for p in parts:
                if length + len(p) > 80:
                    chunks.append(" ".join(current))
                    current = [p]
                    length = len(p)
                else:
                    current.append(p)
                    length += len(p)
            if current:
                chunks.append(" ".join(current))
            return " \\\\\n".join(chunks)

        eq_line = format_multline(parts_eq)
        sub_line = format_multline(parts_sub)

        # --- Construcción del bloque ---
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
        lines.append(r"\renewcommand{\arraystretch}{1.25}")
        lines.append(r"\[ \boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} \]")
        lines.append(r"\[ \mathbf{X} = " + X_block + r",\quad \mathbf{y} = " + y_block + r" \]")
        lines.append(r"\[ \mathbf{X}^\top\mathbf{X} = " + XtX_block + r",\quad \mathbf{X}^\top\mathbf{y} = " + Xty_block + r" \]")
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
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + eq_line)
        lines.append(r"\end{multline*}")
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + sub_line + f" = \\textbf{{{yhat:.6f}}}")
        lines.append(r"\end{multline*}")
        lines.append(f"\\textbf{{Coeficiente de determinación:}} $R^2 = {r2_str}$")

        # Salto de página
        lines.append("\\newpage")

    return "\n".join(lines)

