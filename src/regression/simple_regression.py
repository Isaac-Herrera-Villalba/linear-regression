#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/regression/simple_regression.py
 ------------------------------------------------------------
 Genera el desarrollo completo en LaTeX para una regresión lineal simple.
 Incluye:
   - Tabla de resumen del dataset (con dimensiones)
   - Cálculo paso a paso de β₁ (pendiente) y β₀ (intercepto)
   - Ecuación del modelo
   - Cálculo e interpretación del coeficiente de determinación (R²)
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd

from src.report.report_latex import _fmt_number, dataset_preview_table


def run_simple_regression(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    instances: List[Dict[str, float]],
    instance_start: int
) -> str:
    """Construye el bloque LaTeX completo para regresión lineal simple."""

    # === Datos base ===
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    n = len(x)

    # === Cálculos estadísticos ===
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # β₁ (pendiente) y β₀ (intercepto)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    beta1 = num / den
    beta0 = y_mean - beta1 * x_mean

    # R²
    y_pred = beta0 + beta1 * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None

    # === Construcción LaTeX ===
    lines: List[str] = []
    lines.append(f"\\section*{{Instancia {instance_start}}}")
    lines.append(
        rf"\textbf{{Regresión lineal simple}} sobre la variable ${x_col}$ con $n = {n}$ observaciones."
    )

    # --- Tabla del dataset ---
    rows, cols = df.shape
    lines.append(r"\subsection*{Resumen del dataset}")
    lines.append(dataset_preview_table(df))

    # === Paso 1: Modelo general ===
    lines.append(r"\subsection*{Paso 1: Modelo lineal general}")
    lines.append(r"\[ y = \beta_0 + \beta_1 x \]")
    lines.append(r"\textbf{Donde:}")
    lines.append(r"\begin{itemize}")
    lines.append(r"  \item $x$ : Variable independiente.")
    lines.append(r"  \item $y$ : Variable dependiente.")
    lines.append(r"  \item $\beta_0$ : Intercepto (ordenada al origen).")
    lines.append(r"  \item $\beta_1$ : Pendiente (coeficiente de regresión).")
    lines.append(r"\end{itemize}")

    # === Paso 2: Cálculo de los coeficientes ===
    lines.append(r"\subsection*{Paso 2: Cálculo de los coeficientes}")
    lines.append(r"\textbf{Datos estadísticos:}")
    lines.append(
        rf"\[\sum x_i = {_fmt_number(sum_x)}, \quad "
        rf"\sum y_i = {_fmt_number(sum_y)}, \quad "
        rf"\sum x_i y_i = {_fmt_number(sum_xy)}, \quad "
        rf"\sum x_i^2 = {_fmt_number(sum_x2)}\]"
    )
    lines.append(
        rf"\[\bar{{x}} = {_fmt_number(x_mean)}, \quad \bar{{y}} = {_fmt_number(y_mean)}\]"
    )

    # Fórmula pendiente
    lines.append(r"\textbf{Fórmula para la pendiente:}")
    lines.append(
        r"\[\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}"
        rf" = \frac{{{_fmt_number(num)}}}{{{_fmt_number(den)}}} = {_fmt_number(beta1)} \]"
    )

    # Fórmula intercepto
    lines.append(r"\textbf{Fórmula para la ordenada al origen:}")
    lines.append(
        r"\[\beta_0 = \bar{y} - \beta_1 \bar{x}"
        rf" = {_fmt_number(y_mean)} - ({_fmt_number(beta1)})({_fmt_number(x_mean)}) = {_fmt_number(beta0)} \]"
    )

    # Ecuación final
    lines.append(r"\textbf{Ecuación final del modelo:}")
    lines.append(
        rf"\[\hat{{y}} = {_fmt_number(beta0)} + {_fmt_number(beta1)}x \]"
    )

    # === Paso 3: Coeficiente de determinación ===
    lines.append(r"\subsection*{Paso 3: Coeficiente de determinación $(R^2)$}")
    lines.append(
        r"El coeficiente de determinación mide qué tan bien el modelo explica la variabilidad observada de $y$."
    )
    lines.append(r"\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \]")
    lines.append(
        r"donde: \\[0.25em]"
        r"\begin{itemize}"
        r"\item $SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ es la suma de cuadrados del residuo (error)."
        r"\item $SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2$ es la suma total de cuadrados respecto a la media."
        r"\item $\bar{y}$ es la media aritmética de los valores observados de $y$."
        r"\end{itemize}"
    )

    lines.append(r"\textbf{Sustitución paso a paso:}")
    lines.append(r"\[ SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]")
    lines.append(r"\[ SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2 \]")
    lines.append(r"\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} \]")
    lines.append(r"\\[0.5em]")
    lines.append(
        r"En este modelo, al sustituir los valores obtenidos para cada término, se obtiene el siguiente resultado numérico:"
    )
    lines.append(rf"\\[0.25em] R^2 = {_fmt_number(r2)}")

    # ✅ corrección del texto con format()
    lines.append(
        "El modelo logra explicar aproximadamente el \\textbf{{{}}}\\% de la variabilidad total de $Y$, "
        "lo que representa un ajuste excelente.".format(_fmt_number(r2 * 100))
    )

    # === Paso 4: Predicción para instancias ===
    for i, inst in enumerate(instances, start=instance_start):
        val = float(inst[x_col])
        y_pred_inst = beta0 + beta1 * val
        lines.append(r"\subsection*{Paso 4: Sustitución de X en Y (predicción)}")
        lines.append(f"Atributos: {x_col}={_fmt_number(val)}")
        lines.append(
            rf"\[\hat{{y}} = {_fmt_number(beta0)} + {_fmt_number(beta1)}({_fmt_number(val)})"
            rf" = \textbf{{{_fmt_number(y_pred_inst)}}} \]"
        )

    lines.append("\\newpage")
    return "\n".join(lines)

