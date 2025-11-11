#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_builder.py
 ------------------------------------------------------------
 Genera el bloque completo del reporte LaTeX para cada instancia.
 Corrige las proporciones tipográficas entre matrices y vectores:
 los símbolos (X, y, β, etc.) se mantienen en tamaño natural, y
 solo las matrices se escalan dinámicamente.

 Ahora incluye:
   - Matriz transpuesta X^T
   - Matriz inversa (X^T X)^{-1}  (con pseudoinversa si es necesario)
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from src.regression.linear_regression import run_linear_regression, predict
from src.report.report_latex import dataset_preview_table, _matrix_to_latex, _fmt_number
from src.regression.derivations import build_derivation_block


# ======================================================================
# Funciones auxiliares
# ======================================================================

def split_equation(parts: List[str], width: int = 90) -> str:
    """
    Divide ecuaciones largas en varias líneas si exceden la longitud 'width'.
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

def _aligned_block(symbol: str, matrix_block: str, w: float, h: float) -> str:
    """
    Genera un bloque matemático LaTeX alineado verticalmente:
    el símbolo (X, y, β, etc.) queda centrado con la matriz escalada.
    """
    return (
        r"\[ %s = \vcenter{\hbox{\resizebox{%.2f\linewidth}{!}{$ %s $}}} \]"
        % (symbol, w, matrix_block)
    )

def compute_scale(matrix_block: str) -> tuple[float, float]:
    """
    Calcula una escala proporcional (solo en ancho) según tamaño de la matriz.
    Mantiene proporciones visuales entre vectores, matrices medianas y grandes.
    """
    try:
        rows = matrix_block.count('\\\\') + 1
        cols = matrix_block.split(']')[0].count('&') + 1

        # --- Escalado horizontal más natural ---
        if cols == 1:
            width = 0.18  # vectores: más compactos
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

        # --- Altura siempre natural ---
        height = 1.0

        return (round(width, 2), height)
    except Exception:
        return (0.7, 1.0)

# ======================================================================
# Función principal del reporte
# ======================================================================

def build_full_report_block(
    instances: List[Dict[str, float]],
    df_num: pd.DataFrame,
    y_col: str,
    x_cols: List[str]
) -> str:
    """
    Construye el bloque completo del reporte LaTeX para todas las instancias.
    """
    lines: List[str] = []

    for idx, inst in enumerate(instances, 1):
        # --- Cálculos base ---
        res = run_linear_regression(df_num, y_col, x_cols)
        n, m_plus1 = res.X.shape          # m_plus1 = m + 1 (columna de 1's)
        m = m_plus1 - 1
        r2_str = _fmt_number(res.r2) if res.r2 is not None else "—"
        dataset_table = dataset_preview_table(df_num)

        # --- Conversiones de matrices ya disponibles y derivadas ---
        X_block      = _matrix_to_latex(res.X)                      # (n x (m+1))
        y_block      = _matrix_to_latex(res.y.reshape(-1, 1))       # (n x 1)
        Xt           = res.X.T                                      # ((m+1) x n)
        Xt_block     = _matrix_to_latex(Xt)
        XtX_block    = _matrix_to_latex(res.XtX)                    # ((m+1) x (m+1))
        Xty_block    = _matrix_to_latex(res.Xty.reshape(-1, 1))     # ((m+1) x 1)

        # Inversa (o pseudoinversa) de X^T X
        try:
            XtX_inv = np.linalg.inv(res.XtX)
            inv_note = ""
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(res.XtX)
            inv_note = r"\\[0.25em]\textit{Nota: se empleó pseudoinversa por singularidad o mal condicionado.}"
        XtX_inv_block = _matrix_to_latex(XtX_inv)

        beta_vec  = " \\\\ ".join(_fmt_number(b) for b in res.beta)

        # --- Tabla de parámetros ---
        beta_lines = [f"$\\beta_0$ & {_fmt_number(res.beta[0])} \\\\"]
        for j, name in enumerate(x_cols, start=1):
            beta_lines.append(f"$\\beta_{j}$ ({name}) & {_fmt_number(res.beta[j])} \\\\")
        beta_table = "\n".join(beta_lines)

        # --- Ecuaciones de sustitución ---
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
        eq_line  = split_equation(parts_eq)
        sub_line = split_equation(parts_sub)

        # === SECCIONES ===
        lines.append(f"\\section*{{Instancia {idx}}}")
        lines.append("\\subsection*{Resumen del dataset}")
        lines.append(dataset_table)

        # Paso 1
        lines.append(r"\subsection*{Paso 1: Modelo lineal}")
        lines.append(r"\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \cdots + \beta_m x_m \]")

        lines.append(r"\subsection*{Función objetivo y derivadas parciales}")
        lines.append(build_derivation_block(df_num, y_col, x_cols))

        # Paso 2
        lines.append(r"\subsection*{Paso 2: Forma matricial y solución}")
        lines.append(r"Despejando la matriz $\boldsymbol{\beta}$ queda de la siguiente forma:\\[0.75em]")
        lines.append(r"\begin{center}")
        lines.append(r"$\displaystyle \boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$")
        lines.append(r"\end{center}")

        # --- Escalado ---
        wX,   hX   = compute_scale(X_block)
        wy,   hy   = compute_scale(y_block)
        wXt,  hXt  = compute_scale(Xt_block)
        wXtX, hXtX = compute_scale(XtX_block)
        wInv, hInv = compute_scale(XtX_inv_block)
        wXty, hXty = compute_scale(Xty_block)
        wB,    hB  = compute_scale(beta_vec)

        # --- Bloques de matrices (alineación corregida con \vcenter) ---
        lines.append(r"\textbf{Matriz } $\mathbf{X} \in \mathbb{R}^{%d\times%d}$" % (n, m_plus1))
        lines.append(_aligned_block(r"\mathbf{X}", X_block, wX, hX))

        lines.append(r"\textbf{Vector } $\mathbf{y} \in \mathbb{R}^{%d\times1}$" % (n,))
        lines.append(_aligned_block(r"\mathbf{y}", y_block, wy, hy))

        lines.append(r"\textbf{Matriz transpuesta } $\mathbf{X}^\top \in \mathbb{R}^{%d\times%d}$" % (m_plus1, n))
        lines.append(_aligned_block(r"\mathbf{X}^\top", Xt_block, wXt, hXt))

        lines.append(r"\textbf{Matriz } $\mathbf{X}^\top\mathbf{X} \in \mathbb{R}^{%d\times%d}$" % (m_plus1, m_plus1))
        lines.append(_aligned_block(r"\mathbf{X}^\top\mathbf{X}", XtX_block, wXtX, hXtX))

        lines.append(r"\textbf{Matriz inversa } $(\mathbf{X}^\top\mathbf{X})^{-1} \in \mathbb{R}^{%d\times%d}$" % (m_plus1, m_plus1))
        lines.append(_aligned_block(r"(\mathbf{X}^\top\mathbf{X})^{-1}", XtX_inv_block, wInv, hInv))
        if inv_note:
            lines.append(inv_note)

        lines.append(r"\textbf{Matriz } $\mathbf{X}^\top\mathbf{y} \in \mathbb{R}^{%d\times1}$" % (m_plus1,))
        lines.append(_aligned_block(r"\mathbf{X}^\top\mathbf{y}", Xty_block, wXty, hXty))

        lines.append(_aligned_block(r"\boldsymbol{\beta}", r"\begin{bmatrix}%s\end{bmatrix}" % beta_vec, wB, hB))

        # --- Tabla de parámetros ---
        lines.append(r"\begin{tabular}{l r}")
        lines.append(r"\toprule")
        lines.append(r"Parámetro & Valor \\")
        lines.append(r"\midrule")
        lines.append(beta_table)
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        # Paso 3: Coeficiente de determinación
        lines.append(r"\subsection*{Paso 3: Coeficiente de determinación $(R^2)$}")
        lines.append(
            r"El \textbf{coeficiente de determinación} mide qué tan bien el modelo de regresión "
            r"explica la variabilidad observada de la variable dependiente $y$. "
            r"Se define como la proporción de la variabilidad total explicada por el modelo."
        )
        lines.append(r"\\[1em]")
        lines.append(r"\textbf{Definición general:}")
        lines.append(r"\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \]")
        lines.append(
            r"donde:"
            r"\\[0.25em]"
            r"\begin{itemize}"
            r"\item $SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ es la \textit{suma de cuadrados del residuo (error)}."
            r"\item $SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2$ es la \textit{suma total de cuadrados respecto a la media}."
            r"\item $\bar{y}$ es la media aritmética de los valores observados de $y$."
            r"\end{itemize}"
        )

        # === Cálculo numérico detallado ===
        y = res.y
        y_hat = res.X @ res.beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        lines.append(r"\textbf{Sustitución paso a paso:}")
        lines.append(r"\[ SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = " + _fmt_number(ss_res) + r" \]")
        lines.append(r"\[ SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2 = " + _fmt_number(ss_tot) + r" \]")
        lines.append(
            r"\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = "
            + "1 - \\frac{" + _fmt_number(ss_res) + "}{" + _fmt_number(ss_tot) + "} = "
            + _fmt_number(res.r2)
            + r" \]"
        )
        lines.append(r"\\[0.5em]")

        lines.append(
            r"En este modelo, al sustituir los valores obtenidos para cada término, "
            r"se obtiene el siguiente resultado numérico:"
        )
        lines.append(r"\[ R^2 = " + _fmt_number(res.r2) + r" \]")

        lines.append(
            r"\textbf{Interpretación:} un valor de $R^2$ cercano a 1 indica que el modelo explica "
            r"casi toda la variabilidad de los datos, es decir, que las predicciones ajustan muy "
            r"bien a los valores observados. En este caso:"
        )

        pct = _fmt_number(res.r2 * 100)
        lines.append(
            r"\\[0.5em]El modelo logra explicar aproximadamente el \textbf{" + pct + r"\%} "
            r"de la variabilidad total de $Y$, lo que representa un ajuste excelente."
        )

        # Paso 4
        lines.append(r"\subsection*{Paso 4: Sustitución de X en Y (predicción)}")
        lines.append("Atributos: " + ", ".join(f"{k}={v}" for k, v in inst.items()))
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + eq_line)
        lines.append(r"\end{multline*}")
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + sub_line + f" = \\textbf{{{_fmt_number(yhat)}}}")
        lines.append(r"\end{multline*}")

        lines.append("\\newpage")

    return "\n".join(lines)

