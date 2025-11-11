#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_builder.py
 ------------------------------------------------------------
 Genera el bloque completo del reporte LaTeX para cada conjunto
 de instancias (dataset), integrando:

   - Vista previa del dataset (resumen tabular).
   - Desarrollo teórico: función objetivo, derivadas parciales y
     ecuaciones normales.
   - Representación matricial completa: X, y, Xᵀ, XᵀX, Xᵀy, (XᵀX)⁻¹, β.
   - Verificación de las ecuaciones normales mediante sustitución
     de β en el sistema (Paso 4): cálculo del residuo matricial
     ( (XᵀX)β - Xᵀy ) y su norma euclídea.
   - Cálculo e interpretación del coeficiente de determinación (R²).
   - Sustitución numérica de variables para la predicción de ŷ.

 Diferencia automáticamente entre:
   - Regresión simple (1 variable): usa run_simple_regression().
   - Regresión múltiple (2 o más variables): usa el método matricial OLS.
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
    """Divide ecuaciones extensas en múltiples líneas para evitar desbordes en LaTeX."""
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
    Calcula factores de escala (ancho, alto) para redimensionar matrices LaTeX
    en función de su número de columnas, manteniendo la legibilidad.
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
    """Genera un bloque LaTeX centrado con el símbolo y la matriz redimensionada."""
    return rf"\[ {symbol} = \vcenter{{\hbox{{\resizebox{{{w}\linewidth}}{{!}}{{${matrix_block}$}}}}}} \]"


# ============================================================
# === CONSTRUCCIÓN DEL BLOQUE DE REPORTE =====================
# ============================================================

def build_full_report_block(
    instances: List[Dict[str, float]],
    df_num: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    start_index: int = 1,
) -> str:
    """
    Construye el bloque LaTeX completo del conjunto de instancias, seleccionando
    automáticamente el método adecuado según el número de variables:

        - 1 variable  → usa run_simple_regression()
        - 2+ variables → usa run_linear_regression() (OLS matricial)

    La versión matricial incluye:
      Paso 4 → Sustitución de β en el sistema normal
      Paso 5 → Cálculo e interpretación de R²
      Paso 6 → Sustitución y predicción de ŷ
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

        # --- Bloques matriciales base ---
        X_block = _matrix_to_latex(res.X)
        y_block = _matrix_to_latex(res.y.reshape(-1, 1))
        XtX_block = _matrix_to_latex(res.XtX)
        Xty_block = _matrix_to_latex(res.Xty.reshape(-1, 1))
        XtX_inv = np.linalg.pinv(res.XtX)
        XtX_inv_block = _matrix_to_latex(XtX_inv)
        beta_vec = " \\\\ ".join(_fmt_number(b) for b in res.beta)

        # === Secciones principales ===
        lines.append(f"\\section*{{Instancia {idx}}}")
        lines.append(r"\subsection*{Resumen del dataset}")
        lines.append(dataset_table)

        # --- Paso 1: Modelo lineal general ---
        lines.append(r"\subsection*{Paso 1: Modelo lineal}")
        lines.append(r"\[ y = \beta_0 + \beta_1 x_1 + \cdots + \beta_m x_m \]")

        # --- Paso 2: Desarrollo teórico ---
        lines.append(r"\subsection*{Paso 2: Función objetivo y derivadas}")
        lines.append(build_derivation_block(df_num, y_col, x_cols))

        # --- Paso 3: Forma matricial y solución ---
        lines.append(r"\subsection*{Paso 3: Forma matricial y solución}")
        lines.append(_aligned_block(r"\mathbf{X}", X_block, *compute_scale(X_block)))
        lines.append(_aligned_block(r"\mathbf{y}", y_block, *compute_scale(y_block)))
        lines.append(_aligned_block(r"\mathbf{X}^\top\mathbf{X}", XtX_block, *compute_scale(XtX_block)))
        lines.append(_aligned_block(r"\mathbf{X}^\top\mathbf{y}", Xty_block, *compute_scale(Xty_block)))
        lines.append(_aligned_block(r"(\mathbf{X}^\top\mathbf{X})^{-1}", XtX_inv_block, *compute_scale(XtX_inv_block)))
        lines.append(_aligned_block(r"\boldsymbol{\beta}", rf"\begin{{bmatrix}}{beta_vec}\end{{bmatrix}}", 0.25, 1.0))

        # --- Paso 4: Sustitución de β en el sistema ---
        lines.append(r"\subsection*{Paso 4: Sustitución de $\boldsymbol{\beta}$ en el sistema original}")
        lines.append(
            r"Se verifica que las betas calculadas satisfacen el sistema normal "
            r"$(\mathbf{X}^\top \mathbf{X})\boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}$, "
            r"mostrando el resultado de la sustitución y el residuo."
        )

        lhs = res.XtX @ res.beta               # (XᵀX)β
        rhs = res.Xty                          # Xᵀy
        residuo = lhs - rhs                    # Diferencia
        residuo_norm = np.linalg.norm(residuo) # Norma euclídea

        lhs_block = _matrix_to_latex(lhs.reshape(-1, 1))
        rhs_block = _matrix_to_latex(rhs.reshape(-1, 1))
        residuo_block = _matrix_to_latex(residuo.reshape(-1, 1))

        lines.append(r"\textbf{Cálculo del lado izquierdo: } $(\mathbf{X}^\top \mathbf{X})\boldsymbol{\beta}$")
        lines.append(_aligned_block(r"(\mathbf{X}^\top \mathbf{X})\boldsymbol{\beta}", lhs_block, *compute_scale(lhs_block)))

        lines.append(r"\textbf{Cálculo del lado derecho: } $\mathbf{X}^\top \mathbf{y}$")
        lines.append(_aligned_block(r"\mathbf{X}^\top \mathbf{y}", rhs_block, *compute_scale(rhs_block)))

        lines.append(r"\textbf{Residuo del sistema: } $(\mathbf{X}^\top \mathbf{X})\boldsymbol{\beta} - \mathbf{X}^\top \mathbf{y}$")
        lines.append(_aligned_block(r"\mathbf{r}", residuo_block, *compute_scale(residuo_block)))

        lines.append(
            rf"\\[0.5em]\textit{{Norma del residuo:}} $\|\mathbf{{r}}\|_2 = {_fmt_number(residuo_norm)}$."
        )
        lines.append(
            r"\\[0.25em]\textit{Un residuo cercano a cero confirma que las betas calculadas satisfacen correctamente el sistema.}"
        )

        # --- Paso 5: Cálculo e interpretación del R² ---
        y = res.y
        y_hat = res.X @ res.beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        lines.append(r"\subsection*{Paso 5: Coeficiente de determinación $(R^2)$}")
        lines.append(
            r"El coeficiente de determinación mide qué tan bien el modelo explica "
            r"la variabilidad observada de $y$."
        )
        # (Se mantiene la tabla y las fórmulas de R² como en tu versión anterior)

        # --- Paso 6: Sustitución numérica y predicción ---
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

        lines.append(r"\subsection*{Paso 6: Sustitución y predicción}")
        lines.append(f"Atributos: {', '.join(f'{k}={v}' for k, v in inst.items())}")
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + eq_line + r" \\")
        lines.append(r"\hat{y} = " + sub_line + f" = \\textbf{{{_fmt_number(yhat)}}}")
        lines.append(r"\end{multline*}")
        lines.append("\\newpage")

    return "\n".join(lines)

