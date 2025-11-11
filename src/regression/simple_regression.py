#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/regression/simple_regression.py
 ------------------------------------------------------------
 Genera el bloque LaTeX correspondiente a una regresión lineal simple
 (una sola variable independiente).

 El procedimiento realiza los siguientes cálculos paso a paso:

   1. Conversión robusta de los datos numéricos.
   2. Cálculo de estadísticos base:
        ∑x, ∑y, ∑xy, ∑x², medias (x̄, ȳ).
   3. Obtención explícita de los coeficientes:
        β₁ = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / Σ[(xᵢ - x̄)²]
        β₀ = ȳ - β₁ x̄
   4. Cálculo del coeficiente de determinación R².
   5. Verificación de las ecuaciones normales sustituyendo β:
        - Muestra los valores resultantes.
        - Calcula los residuos numéricos, que deben ser ≈ 0.
   6. Sustitución numérica de X en Y para obtener ŷ.

 Cada paso se documenta y genera en formato LaTeX, listo para insertarse
 en el reporte final.

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
    Construye el bloque LaTeX completo para una regresión lineal simple.

    Parámetros
    ----------
    df : pd.DataFrame
        Conjunto de datos que contiene las columnas numéricas X y Y.
    y_col : str
        Nombre de la variable dependiente.
    x_col : str
        Nombre de la variable independiente.
    instances : List[Dict[str, float]]
        Lista de instancias con valores de X a sustituir.
    instance_start : int
        Índice base de la instancia dentro del reporte.

    Retorna
    -------
    str
        Cadena en formato LaTeX con el desarrollo teórico, los cálculos
        y la verificación de consistencia numérica.
    """
    # === Conversión robusta a float ===
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    n = len(x)

    # === Estadísticos base ===
    sum_x = float(np.sum(x))
    sum_y = float(np.sum(y))
    sum_xy = float(np.sum(x * y))
    sum_x2 = float(np.sum(x ** 2))
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    # === Cálculo de coeficientes β₀ y β₁ ===
    num = float(np.sum((x - x_mean) * (y - y_mean)))
    den = float(np.sum((x - x_mean) ** 2))
    beta1 = num / den
    beta0 = y_mean - beta1 * x_mean

    # === Ajuste global: R² ===
    y_pred = beta0 + beta1 * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None

    # === Verificación: ecuaciones normales ===
    # Sistema de dos ecuaciones:
    # Σy = nβ₀ + β₁Σx
    # Σxy = β₀Σx + β₁Σx²
    res_eq0 = sum_y - (n * beta0 + beta1 * sum_x)
    res_eq1 = sum_xy - (beta0 * sum_x + beta1 * sum_x2)

    # === Construcción del bloque LaTeX ===
    lines: List[str] = []
    lines.append(f"\\section*{{Instancia {instance_start}}}")
    lines.append(
        rf"\textbf{{Regresión lineal simple}} sobre la variable ${x_col}$ con $n = {n}$ observaciones."
    )

    # --- Paso 1: Modelo lineal general ---
    lines.append(r"\subsection*{Paso 1: Modelo lineal general}")
    lines.append(r"\[ y = \beta_0 + \beta_1 x \]")
    lines.append(r"\textbf{Donde:}")
    lines.append(r"\begin{itemize}")
    lines.append(r"  \item $x$ : Variable independiente.")
    lines.append(r"  \item $y$ : Variable dependiente.")
    lines.append(r"  \item $\beta_0$ : Intercepto (ordenada al origen).")
    lines.append(r"  \item $\beta_1$ : Pendiente (coeficiente de regresión).")
    lines.append(r"\end{itemize}")

    # --- Paso 2: Cálculo de los coeficientes ---
    lines.append(r"\subsection*{Paso 2: Cálculo de los coeficientes}")
    lines.append(r"\textbf{Datos estadísticos:}")
    lines.append(
        rf"\[\sum x_i = {_fmt_number(sum_x)}, \quad "
        rf"\sum y_i = {_fmt_number(sum_y)}, \quad "
        rf"\sum x_i y_i = {_fmt_number(sum_xy)}, \quad "
        rf"\sum x_i^2 = {_fmt_number(sum_x2)}\]"
    )
    lines.append(rf"\[\bar{{x}} = {_fmt_number(x_mean)}, \quad \bar{{y}} = {_fmt_number(y_mean)}\]")

    # --- Fórmulas de β₁ y β₀ ---
    lines.append(r"\textbf{Fórmula para la pendiente:}")
    lines.append(
        r"\[\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}"
        rf" = \frac{{{_fmt_number(num)}}}{{{_fmt_number(den)}}} = {_fmt_number(beta1)} \]"
    )

    lines.append(r"\textbf{Fórmula para la ordenada al origen:}")
    lines.append(
        r"\[\beta_0 = \bar{y} - \beta_1 \bar{x}"
        rf" = {_fmt_number(y_mean)} - ({_fmt_number(beta1)})({_fmt_number(x_mean)}) = {_fmt_number(beta0)} \]"
    )

    lines.append(r"\textbf{Ecuación final del modelo:}")
    lines.append(rf"\[\hat{{y}} = {_fmt_number(beta0)} + {_fmt_number(beta1)}x \]")

    # --- Paso 3: Coeficiente de determinación ---
    lines.append(r"\subsection*{Paso 3: Coeficiente de determinación $(R^2)$}")
    lines.append(r"El coeficiente de determinación mide qué tan bien el modelo explica la variabilidad de $y$.")

    # --- Paso 4: Sustitución de β en las ecuaciones normales ---
    lines.append(r"\subsection*{Paso 4: Sustitución de $\boldsymbol{\beta}$ en las ecuaciones normales (verificación)}")
    lines.append(r"Se sustituyen los valores calculados de $\beta_0$ y $\beta_1$ para comprobar la validez del sistema:")

    # Primera ecuación
    lines.append(r"\textbf{Primera ecuación:}")
    lines.append(
        rf"\[\sum y_i = n\beta_0 + \beta_1 \sum x_i"
        rf" = {n}({_fmt_number(beta0)}) + ({_fmt_number(beta1)})({_fmt_number(sum_x)})"
        rf" = {_fmt_number(n*beta0 + beta1*sum_x)} \]"
    )
    lines.append(
        rf"\[\sum y_i - (n\beta_0 + \beta_1 \sum x_i)"
        rf" = {_fmt_number(sum_y)} - ({_fmt_number(n*beta0 + beta1*sum_x)})"
        rf" = {_fmt_number(res_eq0)} \]"
    )

    # Segunda ecuación
    lines.append(r"\textbf{Segunda ecuación:}")
    lines.append(
        rf"\[\sum x_i y_i = \beta_0 \sum x_i + \beta_1 \sum x_i^2"
        rf" = ({_fmt_number(beta0)})({_fmt_number(sum_x)}) + ({_fmt_number(beta1)})({_fmt_number(sum_x2)})"
        rf" = {_fmt_number(beta0*sum_x + beta1*sum_x2)} \]"
    )
    lines.append(
        rf"\[\sum x_i y_i - (\beta_0 \sum x_i + \beta_1 \sum x_i^2)"
        rf" = {_fmt_number(sum_xy)} - ({_fmt_number(beta0*sum_x + beta1*sum_x2)})"
        rf" = {_fmt_number(res_eq1)} \]"
    )

    lines.append(r"\textit{Valores cercanos a cero indican que las ecuaciones normales se satisfacen correctamente.}")

    # --- Paso 5: Predicción para instancias ---
    for i, inst in enumerate(instances, start=instance_start):
        val = float(inst[x_col])
        y_pred_inst = beta0 + beta1 * val
        lines.append(r"\subsection*{Paso 5: Sustitución de X en Y (predicción)}")
        lines.append(f"Atributos: {x_col}={_fmt_number(val)}")
        lines.append(
            rf"\[\hat{{y}} = {_fmt_number(beta0)} + {_fmt_number(beta1)}({_fmt_number(val)})"
            rf" = \textbf{{{_fmt_number(y_pred_inst)}}} \]"
        )

    lines.append("\\newpage")
    return "\n".join(lines)

