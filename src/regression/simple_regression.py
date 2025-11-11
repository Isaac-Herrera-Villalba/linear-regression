#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/regression/simple_regression.py
 ------------------------------------------------------------
 Módulo de apoyo para la generación del bloque LaTeX asociado a una
 regresión lineal simple (una variable independiente).

 El procedimiento calcula de manera explícita:
   - Estadísticos base (∑x, ∑y, ∑xy, ∑x², medias).
   - Pendiente (β₁) e intercepto (β₀).
   - Ecuación del modelo ŷ = β₀ + β₁ x.
   - Métrica de ajuste R², junto con una tabla interpretativa.
   - Sustitución numérica de instancias para predicción.

 La salida es una cadena LaTeX lista para insertarse en el reporte.
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
    """
    Construye el bloque LaTeX completo correspondiente a una regresión
    lineal simple.

    Parámetros
    ----------
    df : pd.DataFrame
        Conjunto de datos preprocesado. Debe contener columnas numéricas
        para `x_col` y `y_col`. La conversión a float se robustece con
        `pd.to_numeric(..., errors="coerce")`.
    y_col : str
        Nombre de la variable dependiente (Y).
    x_col : str
        Nombre de la variable independiente (X).
    instances : List[Dict[str, float]]
        Lista de instancias a evaluar. Cada elemento es un diccionario
        con el par {x_col: valor}.
    instance_start : int
        Índice con el que inicia el numerado de secciones de instancia
        en el reporte.

    Retorna
    -------
    str
        Cadena con el contenido LaTeX de la sección completa:
        resumen del dataset, modelo, cómputo de coeficientes, R² e
        interpretación, y sustitución de instancias.

    Detalles del flujo
    ------------------
    1) Conversión a float de las columnas `x_col` y `y_col`.
    2) Cálculo de sumatorias y medias.
    3) Obtención de β₁ y β₀:
         β₁ = Σ[(xᵢ- x̄)(yᵢ- ȳ)] / Σ[(xᵢ- x̄)²]
         β₀ = ȳ - β₁ x̄
    4) Cálculo de R² a partir de SS_res y SS_tot.
    5) Construcción del contenido LaTeX en secciones.
    6) Para cada instancia, sustitución numérica y predicción ŷ.
    """
    # === Conversión robusta a float ===
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    n = len(x)

    # === Sumatorias y medias ===
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # === Coeficientes del modelo (β₁, β₀) ===
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    beta1 = num / den
    beta0 = y_mean - beta1 * x_mean

    # === Ajuste: R² ===
    y_pred = beta0 + beta1 * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None

    # === Construcción del bloque LaTeX ===
    lines: List[str] = []
    lines.append(f"\\section*{{Instancia {instance_start}}}")
    lines.append(
        rf"\textbf{{Regresión lineal simple}} sobre la variable ${x_col}$ con $n = {n}$ observaciones."
    )

    # Resumen del dataset
    rows, cols = df.shape
    lines.append(r"\subsection*{Resumen del dataset}")
    lines.append(rf"\textit{{Dimensiones del dataset:}} {rows} filas $\times$ {cols} columnas\\[0.5em]")
    lines.append(dataset_preview_table(df))

    # Paso 1: Modelo general
    lines.append(r"\subsection*{Paso 1: Modelo lineal general}")
    lines.append(r"\[ y = \beta_0 + \beta_1 x \]")
    lines.append(r"\textbf{Donde:}")
    lines.append(r"\begin{itemize}")
    lines.append(r"  \item $x$ : Variable independiente.")
    lines.append(r"  \item $y$ : Variable dependiente.")
    lines.append(r"  \item $\beta_0$ : Intercepto (ordenada al origen).")
    lines.append(r"  \item $\beta_1$ : Pendiente (coeficiente de regresión).")
    lines.append(r"\end{itemize}")

    # Paso 2: Cálculo de coeficientes
    lines.append(r"\subsection*{Paso 2: Cálculo de los coeficientes}")
    lines.append(r"\textbf{Datos estadísticos:}")
    lines.append(
        rf"\[\sum x_i = {_fmt_number(sum_x)}, \quad "
        rf"\sum y_i = {_fmt_number(sum_y)}, \quad "
        rf"\sum x_i y_i = {_fmt_number(sum_xy)}, \quad "
        rf"\sum x_i^2 = {_fmt_number(sum_x2)}\]"
    )
    lines.append(rf"\[\bar{{x}} = {_fmt_number(x_mean)}, \quad \bar{{y}} = {_fmt_number(y_mean)}\]")

    # Fórmula de la pendiente β₁
    lines.append(r"\textbf{Fórmula para la pendiente:}")
    lines.append(
        r"\[\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}"
        rf" = \frac{{{_fmt_number(num)}}}{{{_fmt_number(den)}}} = {_fmt_number(beta1)} \]"
    )

    # Fórmula del intercepto β₀
    lines.append(r"\textbf{Fórmula para la ordenada al origen:}")
    lines.append(
        r"\[\beta_0 = \bar{y} - \beta_1 \bar{x}"
        rf" = {_fmt_number(y_mean)} - ({_fmt_number(beta1)})({_fmt_number(x_mean)}) = {_fmt_number(beta0)} \]"
    )

    # Ecuación final del modelo
    lines.append(r"\textbf{Ecuación final del modelo:}")
    lines.append(rf"\[\hat{{y}} = {_fmt_number(beta0)} + {_fmt_number(beta1)}x \]")

    # Paso 3: Coeficiente de determinación (R²)
    lines.append(r"\subsection*{Paso 3: Coeficiente de determinación $(R^2)$}")
    lines.append(r"El coeficiente de determinación mide qué tan bien el modelo explica la variabilidad observada de $y$.")

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

    # Sustitución numérica de R²
    lines.append(r"\[ SS_{res} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = " + _fmt_number(ss_res) + r" \]")
    lines.append(r"\[ SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2 = " + _fmt_number(ss_tot) + r" \]")
    lines.append(
        r"\[ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{" + _fmt_number(ss_res) +
        "}{" + _fmt_number(ss_tot) + "} = " + _fmt_number(r2) + r" \]"
    )

    lines.append(
        r"\textbf{Interpretación:} un valor de $R^2$ cercano a 1 indica que el modelo explica "
        r"casi toda la variabilidad de los datos."
    )
    lines.append(
        "\\\\[0.5em]El modelo logra explicar aproximadamente el \\textbf{{{}}}\\% "
        "de la variabilidad total de $Y$.".format(_fmt_number(r2 * 100))
    )

    # Paso 4: Predicciones para instancias
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

