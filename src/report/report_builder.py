#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_builder.py
 ------------------------------------------------------------
 Genera el bloque completo del reporte LaTeX para cada conjunto
 de instancias (dataset), integrando:
   - Datos del dataset (vista previa).
   - Desarrollo teórico (función objetivo, derivadas, ecuaciones normales).
   - Representación matricial (X, y, Xᵀ, XᵀX, Xᵀy, β).
   - Cálculo e interpretación del coeficiente de determinación (R²).
   - Sustitución de valores de entrada para la predicción de ŷ.

 Diferencia automáticamente entre:
   - Regresión simple (1 variable): usa run_simple_regression().
   - Regresión múltiple (2 o más variables): usa método matricial OLS.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from src.regression.linear_regression import run_linear_regression, predict
from src.report.report_latex import dataset_preview_table, _matrix_to_latex, _fmt_number
from src.report.derivations_latex import build_derivation_block


# ============================================================
# === UTILIDADES =============================================
# ============================================================

def split_equation(parts: List[str], width: int = 90) -> str:
    """Divide ecuaciones extensas en múltiples líneas."""
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


def compute_scale(matrix_block: str) -> tuple[float, float]:
    """
    Calcula factores de escala para ajustar el tamaño de matrices
    dentro del reporte sin afectar símbolos matemáticos.

    Parámetros
    ----------
    matrix_block : str
        Cadena con el bloque LaTeX de la matriz.

    Retorna
    -------
    tuple[float, float]
        (escala_horizontal, escala_vertical)
    """
    try:
        rows = matrix_block.count('\\\\') + 1
        cols = matrix_block.split(']')[0].count('&') + 1

        if cols == 1:
            width = 0.18
        elif cols <= 4:
            width = 0.55
        elif cols <= 6:
            width = 0.70
        elif cols <= 10:
            width = 0.75
        elif cols <= 15:
            width = 0.80
        else:
            width = 0.85

        return (round(width, 2), 1.0)
    except Exception:
        return (0.7, 1.0)


def _aligned_block(symbol: str, matrix_block: str, w: float, h: float) -> str:
    return rf"\[ {symbol} = \vcenter{{\hbox{{\resizebox{{{w}\linewidth}}{{!}}{{${matrix_block}$}}}}}} \]"


# ============================================================
# === CONSTRUCCIÓN DE BLOQUES =================================
# ============================================================

def build_full_report_block(
    instances: List[Dict[str, float]],
    df_num: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    start_index: int = 1,
) -> str:
    """
    Construye el bloque LaTeX completo del conjunto de instancias,
    seleccionando el método adecuado según el número de variables:
      - 1 variable  → usa run_simple_regression()
      - 2+ variables → usa run_linear_regression()
    """
    from src.regression.simple_regression import run_simple_regression

    lines: List[str] = []

    # === CASO 1: REGRESIÓN SIMPLE ======================================
    if len(x_cols) == 1:
        report = run_simple_regression(
            df=df_num,
            y_col=y_col,
            x_col=x_cols[0],
            instances=instances,
            instance_start=start_index
        )
        lines.append(report)
        return "\n".join(lines)

    # === CASO 2: REGRESIÓN MÚLTIPLE ====================================
    for idx, inst in enumerate(instances, start=start_index):
        res = run_linear_regression(df_num, y_col, x_cols)
        n, m_plus1 = res.X.shape
        dataset_table = dataset_preview_table(df_num)

        X_block = _matrix_to_latex(res.X)
        y_block = _matrix_to_latex(res.y.reshape(-1, 1))
        XtX_block = _matrix_to_latex(res.XtX)
        Xty_block = _matrix_to_latex(res.Xty.reshape(-1, 1))
        XtX_inv = np.linalg.pinv(res.XtX)
        XtX_inv_block = _matrix_to_latex(XtX_inv)

        beta_vec = " \\\\ ".join(_fmt_number(b) for b in res.beta)

        lines.append(f"\\section*{{Instancia {idx}}}")
        lines.append(r"\subsection*{Resumen del dataset}")
        lines.append(dataset_table)

        lines.append(r"\subsection*{Paso 1: Modelo lineal}")
        lines.append(r"\[ y = \beta_0 + \beta_1 x_1 + \cdots + \beta_m x_m \]")

        lines.append(r"\subsection*{Paso 2: Función objetivo y derivadas}")
        lines.append(build_derivation_block(df_num, y_col, x_cols))

        lines.append(r"\subsection*{Paso 3: Forma matricial y solución}")
        lines.append(_aligned_block(r"\mathbf{X}", X_block, *compute_scale(X_block)))
        lines.append(_aligned_block(r"\mathbf{y}", y_block, *compute_scale(y_block)))
        lines.append(_aligned_block(r"\mathbf{X}^\top\mathbf{X}", XtX_block, *compute_scale(XtX_block)))
        lines.append(_aligned_block(r"\mathbf{X}^\top\mathbf{y}", Xty_block, *compute_scale(Xty_block)))
        lines.append(_aligned_block(r"(\mathbf{X}^\top\mathbf{X})^{-1}", XtX_inv_block, *compute_scale(XtX_inv_block)))
        lines.append(_aligned_block(r"\boldsymbol{\beta}", rf"\begin{{bmatrix}}{beta_vec}\end{{bmatrix}}", 0.25, 1.0))

        # === Paso 4: Cálculo e interpretación del R² ====================
        y = res.y
        y_hat = res.X @ res.beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        pct = _fmt_number(r2 * 100)

        lines.append(r"\subsection*{Paso 4: Coeficiente de determinación $(R^2)$}")
        lines.append(
            r"El coeficiente de determinación mide qué tan bien el modelo explica "
            r"la variabilidad observada de $y$."
        )

        # --- Tabla interpretativa del R² ---
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tabular}{c l}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{R² (aprox.)} & \textbf{Interpretación} \\")
        lines.append(r"\midrule")
        lines.append(r"0.90 -- 1.00 & Ajuste excelente \\")
        lines.append(r"0.70 -- 0.89 & Ajuste bueno \\")
        lines.append(r"0.40 -- 0.69 & Ajuste regular \\")
        lines.append(r"0.10 -- 0.39 & Ajuste pobre \\")
        lines.append(r"0.00 -- 0.09 & Sin ajuste o nulo \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")

        # --- Sustitución numérica de R² ---
        lines.append(r"\[ SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = " + _fmt_number(ss_res) + r" \]")
        lines.append(r"\[ SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2 = " + _fmt_number(ss_tot) + r" \]")
        lines.append(
            r"\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{" +
            _fmt_number(ss_res) + "}{" + _fmt_number(ss_tot) + "} = " +
            _fmt_number(r2) + r" \]"
        )

        lines.append(
            r"\textbf{Interpretación:} un valor de $R^2$ cercano a 1 indica que el modelo explica "
            r"casi toda la variabilidad de los datos."
        )
        lines.append(
            "\\\\[0.5em]El modelo logra explicar aproximadamente el \\textbf{{{}}}\\% "
            "de la variabilidad total de $Y$.".format(_fmt_number(r2 * 100))
        )

        # === Paso 5: Sustitución y predicción ==========================
        xs = [float(inst[name]) for name in x_cols]
        yhat = predict(res.beta, xs)
        parts_eq = [f"{_fmt_number(res.beta[0])}"]
        for j, name in enumerate(x_cols, start=1):
            sign = "+" if res.beta[j] >= 0 else "-"
            parts_eq.append(f"{sign} {_fmt_number(abs(res.beta[j]))}\\,x_{{{j}}}")
        eq_line = split_equation(parts_eq)

        sub_parts = [f"{_fmt_number(res.beta[0])}"]
        for j, name in enumerate(x_cols, start=1):
            sign = "+" if res.beta[j] >= 0 else "-"
            sub_parts.append(f"{sign} {_fmt_number(abs(res.beta[j]))}({_fmt_number(float(inst[name]))})")
        sub_line = split_equation(sub_parts)

        lines.append(r"\subsection*{Paso 5: Sustitución y predicción}")
        lines.append(f"Atributos: {', '.join(f'{k}={v}' for k, v in inst.items())}")
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + eq_line + r" \\")
        lines.append(r"\hat{y} = " + sub_line + f" = \\textbf{{{_fmt_number(yhat)}}}")
        lines.append(r"\end{multline*}")
        lines.append("\\newpage")

    return "\n".join(lines)

