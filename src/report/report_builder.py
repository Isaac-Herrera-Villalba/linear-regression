#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/report_builder.py
 ------------------------------------------------------------
 Módulo encargado de construir el bloque LaTeX completo del
 reporte de regresión lineal múltiple, integrando:

   - Datos del dataset (vista previa).
   - Desarrollo teórico (función objetivo, derivadas, ecuaciones normales).
   - Representación matricial (X, y, Xᵀ, XᵀX, Xᵀy, β).
   - Cálculo y tabla interpretativa del coeficiente de determinación (R²).
   - Sustitución de valores de entrada para la predicción de ŷ.

 El formato se genera de manera compatible con LaTeX estándar,
 empleando escalado dinámico de matrices y disposición vertical
 alineada para las expresiones matemáticas.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from src.regression.linear_regression import run_linear_regression, predict
from src.report.report_latex import dataset_preview_table, _matrix_to_latex, _fmt_number
from src.report.derivations_latex import build_derivation_block


# ======================================================================
# === FUNCIONES AUXILIARES =============================================
# ======================================================================

def split_equation(parts: List[str], width: int = 90) -> str:
    """
    Divide ecuaciones extensas en varias líneas si superan la longitud
    de caracteres indicada por `width`, preservando el formato LaTeX.

    Parámetros
    ----------
    parts : List[str]
        Fragmentos de la ecuación a ensamblar.
    width : int
        Límite máximo de longitud antes de dividir.

    Retorna
    -------
    str
        Ecuación con saltos de línea insertados.
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
    Crea un bloque matemático alineado verticalmente, combinando
    un símbolo (por ejemplo, X o β) con su representación matricial.

    Parámetros
    ----------
    symbol : str
        Símbolo de la variable o matriz (por ejemplo, "\\mathbf{X}").
    matrix_block : str
        Representación LaTeX del contenido matricial.
    w : float
        Factor de escala horizontal relativo al ancho del documento.
    h : float
        Factor de escala vertical (normalmente fijo en 1.0).

    Retorna
    -------
    str
        Cadena LaTeX del bloque alineado.
    """
    return (
        r"\[ %s = \vcenter{\hbox{\resizebox{%.2f\linewidth}{!}{$ %s $}}} \]"
        % (symbol, w, matrix_block)
    )


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


# ======================================================================
# === FUNCIÓN PRINCIPAL ================================================
# ======================================================================

def build_full_report_block(
    instances: List[Dict[str, float]],
    df_num: pd.DataFrame,
    y_col: str,
    x_cols: List[str]
) -> str:
    """
    Construye la sección LaTeX completa correspondiente a todas las
    instancias de un dataset, abarcando todo el flujo del análisis
    de regresión lineal múltiple.

    Parámetros
    ----------
    instances : List[Dict[str, float]]
        Lista de instancias de entrada con valores para cada variable X.
    df_num : pd.DataFrame
        Dataset filtrado y convertido a numérico.
    y_col : str
        Nombre de la variable dependiente (Y).
    x_cols : List[str]
        Nombres de las variables independientes (X₁..Xₘ).

    Retorna
    -------
    str
        Bloque LaTeX concatenado con los resultados de todas las instancias.
    """
    lines: List[str] = []

    for idx, inst in enumerate(instances, 1):
        # === Cálculos principales ===
        res = run_linear_regression(df_num, y_col, x_cols)
        n, m_plus1 = res.X.shape
        m = m_plus1 - 1
        r2_str = _fmt_number(res.r2) if res.r2 is not None else "—"
        dataset_table = dataset_preview_table(df_num)

        # === Conversión de matrices a formato LaTeX ===
        X_block = _matrix_to_latex(res.X)
        y_block = _matrix_to_latex(res.y.reshape(-1, 1))
        Xt = res.X.T
        Xt_block = _matrix_to_latex(Xt)
        XtX_block = _matrix_to_latex(res.XtX)
        Xty_block = _matrix_to_latex(res.Xty.reshape(-1, 1))

        # Inversa o pseudoinversa según condición numérica
        try:
            XtX_inv = np.linalg.inv(res.XtX)
            inv_note = ""
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(res.XtX)
            inv_note = r"\\[0.25em]\textit{Nota: se empleó pseudoinversa por singularidad o mal condicionado.}"
        XtX_inv_block = _matrix_to_latex(XtX_inv)

        # Vector β
        beta_vec = " \\\\ ".join(_fmt_number(b) for b in res.beta)

        # === Tabla de parámetros (β₀, β₁..βₘ) ===
        beta_lines = [f"$\\beta_0$ & {_fmt_number(res.beta[0])} \\\\"]
        for j, name in enumerate(x_cols, start=1):
            beta_lines.append(f"$\\beta_{j}$ ({name}) & {_fmt_number(res.beta[j])} \\\\")
        beta_table = "\n".join(beta_lines)

        # === Construcción de ecuaciones ===
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
        eq_line = split_equation(parts_eq)
        sub_line = split_equation(parts_sub)

        # === Secciones del reporte ===
        lines.append(f"\\section*{{Instancia {idx}}}")
        lines.append(r"\subsection*{Resumen del dataset}")
        lines.append(dataset_table)

        # Paso 1: Modelo lineal general
        lines.append(r"\subsection*{Paso 1: Modelo lineal}")
        lines.append(r"\[ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_m x_m \]")

        # Paso 2: Desarrollo teórico
        lines.append(r"\subsection*{Paso 2: Función objetivo y derivadas parciales}")
        lines.append(build_derivation_block(df_num, y_col, x_cols))

        # Paso 3: Forma matricial y solución
        lines.append(r"\subsection*{Paso 3: Forma matricial y solución}")
        lines.append(r"Despejando la matriz $\boldsymbol{\beta}$ queda de la siguiente forma:\\[0.75em]")
        lines.append(r"\begin{center}")
        lines.append(r"$\displaystyle \boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$")
        lines.append(r"\end{center}")

        # Escalado de matrices
        wX, hX = compute_scale(X_block)
        wy, hy = compute_scale(y_block)
        wXt, hXt = compute_scale(Xt_block)
        wXtX, hXtX = compute_scale(XtX_block)
        wInv, hInv = compute_scale(XtX_inv_block)
        wXty, hXty = compute_scale(Xty_block)
        wB, hB = compute_scale(beta_vec)

        # --- Representaciones matriciales ---
        lines.append(r"\textbf{Matriz } $\mathbf{X} \in \mathbb{R}^{%d\times%d}$" % (n, m_plus1))
        lines.append(_aligned_block(r"\mathbf{X}", X_block, wX, hX))

        lines.append(r"\textbf{Vector } $\mathbf{y} \in \mathbb{R}^{%d\times1}$" % (n,))
        lines.append(_aligned_block(r"\mathbf{y}", y_block, wy, hy))

        lines.append(r"\textbf{Matriz transpuesta } $\mathbf{X}^\top \in \mathbb{R}^{%d\times%d}$" % (m_plus1, n))
        lines.append(_aligned_block(r"\mathbf{X}^\top", Xt_block, wXt, hXt))

        lines.append(r"\textbf{Matriz } $\mathbf{X}^\top\mathbf{X} \in \mathbb{R}^{%d\times%d}$" % (m_plus1, m_plus1))
        lines.append(_aligned_block(r"\mathbf{X}^\top\mathbf{X}", XtX_block, wXtX, hXtX))

        lines.append(r"\textbf{Matriz inversa } $(\mathbf{X}^\top\mathbf{X})^{-1}$")
        lines.append(_aligned_block(r"(\mathbf{X}^\top\mathbf{X})^{-1}", XtX_inv_block, wInv, hInv))
        if inv_note:
            lines.append(inv_note)

        lines.append(r"\textbf{Matriz } $\mathbf{X}^\top\mathbf{y}$")
        lines.append(_aligned_block(r"\mathbf{X}^\top\mathbf{y}", Xty_block, wXty, hXty))
        lines.append(_aligned_block(r"\boldsymbol{\beta}", r"\begin{bmatrix}%s\end{bmatrix}" % beta_vec, wB, hB))

        # Tabla de parámetros β
        lines.append(r"\begin{tabular}{l r}")
        lines.append(r"\toprule")
        lines.append(r"Parámetro & Valor \\")
        lines.append(r"\midrule")
        lines.append(beta_table)
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

        # Paso 4: Coeficiente de determinación
        lines.append(r"\subsection*{Paso 4: Coeficiente de determinación $(R^2)$}")
        lines.append(
            r"El \textbf{coeficiente de determinación} evalúa qué tan bien el modelo "
            r"explica la variabilidad observada en la variable dependiente $y$. "
            r"Se define como la proporción de la variabilidad total explicada por el modelo."
        )

        # Tabla interpretativa de R²
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

        # Cálculo numérico de R²
        y = res.y
        y_hat = res.X @ res.beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        lines.append(r"\textbf{Sustitución paso a paso:}")
        lines.append(r"\[ SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = " + _fmt_number(ss_res) + r" \]")
        lines.append(r"\[ SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2 = " + _fmt_number(ss_tot) + r" \]")
        lines.append(
            r"\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{" +
            _fmt_number(ss_res) + "}{" + _fmt_number(ss_tot) + "} = " +
            _fmt_number(res.r2) + r" \]"
        )
        lines.append(r"\\[0.5em]")

        # Interpretación final
        lines.append(
            r"\textbf{Interpretación:} un valor de $R^2$ cercano a 1 indica un modelo con alto "
            r"grado de ajuste entre las predicciones y los valores observados."
        )
        pct = _fmt_number(res.r2 * 100)
        lines.append(
            r"\\[0.5em]El modelo explica aproximadamente el \textbf{" + pct + r"\%} "
            r"de la variabilidad total de $Y$."
        )

        # Paso 5: Sustitución y predicción
        lines.append(r"\subsection*{Paso 5: Sustitución de X en Y (predicción)}")
        lines.append("Atributos: " + ", ".join(f"{k}={v}" for k, v in inst.items()))
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + eq_line)
        lines.append(r"\end{multline*}")
        lines.append(r"\begin{multline*}")
        lines.append(r"\hat{y} = " + sub_line + f" = \\textbf{{{_fmt_number(yhat)}}}")
        lines.append(r"\end{multline*}")
        lines.append("\\newpage")

    return "\n".join(lines)

