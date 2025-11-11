#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report/derivations_latex.py
 ------------------------------------------------------------
 Módulo responsable de generar el bloque LaTeX con el desarrollo
 teórico completo del modelo de regresión lineal múltiple.

 Integra los resultados simbólicos y numéricos producidos por
 `src/regression/derivations.py` y los presenta en formato
 matemático estructurado para el reporte PDF.

 Incluye:
   - Función objetivo general S(β)
   - Derivadas parciales y condiciones de optimalidad
   - Ecuaciones normales (forma de sumas)
   - Sustitución numérica de sumatorias
   - Transición a forma matricial (XᵀXβ = Xᵀy)
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List
import pandas as pd
from src.regression.derivations import (
    compute_sums,
    normal_equations_symbolic,
    normal_equations_numeric,
)
from src.report.report_latex import _fmt_number


def build_derivation_block(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> str:
    """
    Genera el bloque completo en formato LaTeX correspondiente al
    desarrollo teórico de la regresión lineal múltiple.

    Parámetros
    ----------
    df : pd.DataFrame
        Conjunto de datos numéricos a partir del cual se obtienen
        las sumatorias de las variables dependiente e independientes.
    y_col : str
        Nombre de la variable dependiente.
    x_cols : List[str]
        Lista de variables independientes.

    Retorna
    -------
    str
        Cadena con el contenido LaTeX listo para insertarse en el
        reporte final.
    """
    # Obtención de sumatorias y ecuaciones base
    sums = compute_sums(df, y_col, x_cols)
    sym = normal_equations_symbolic(x_cols)
    num = normal_equations_numeric(sums, x_cols)

    lines: List[str] = []

    # --- Función objetivo general ---
    lines.append(r"\textbf{Función objetivo (caso general):}")
    lines.append(
        r"\[ S(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left(y_i - \Big(\beta_0 + \sum_{j=1}^{m} \beta_j x_{i,j}\Big)\right)^2 \]"
    )
    lines.append(r"\textbf{Donde:}")
    lines.append(r"\begin{itemize}")
    lines.append(r"  \item $y_i$ : Valor observado de la variable dependiente.")
    lines.append(r"  \item $x_{i,j}$ : Valor de la variable independiente $j$ en la observación $i$.")
    lines.append(r"  \item $\beta_0$ : Intercepto del modelo.")
    lines.append(r"  \item $\beta_j$ : Coeficiente asociado a $x_j$.")
    lines.append(r"  \item $n$ : Número total de observaciones.")
    lines.append(r"  \item $m$ : Número de variables independientes.")
    lines.append(r"\end{itemize}")

    # --- Derivadas parciales ---
    lines.append(
        r"\textbf{Derivadas parciales:}\\[0.75em]"
        r"\begin{center}"
        r"$\displaystyle \frac{\partial S}{\partial \beta_j}=0, \quad j=0,1,2,\dots,m.$"
        r"\end{center}"
        r"\\[-0.25em]"
        r"\textit{Estas condiciones representan el punto donde la función $S(\beta)$ alcanza su valor mínimo, "
        r"definiendo las condiciones de optimalidad.}"
    )

    # --- Ecuaciones normales simbólicas ---
    lines.append(r"\paragraph{Ecuaciones normales (forma de sumas).}")
    lines.append(r"\begin{align*}")
    for s in sym:
        lines.append(s.replace("β", r"\beta").replace("Σ", r"\sum") + r" \\")
    lines.append(r"\end{align*}")

    # --- Sustitución numérica de sumatorias ---
    lines.append(r"\paragraph{Sustitución numérica de sumatorias (con el dataset).}")
    lines.append(r"\begin{align*}")
    for s in num:
        lines.append(s.replace("β", r"\beta").replace("Σ", r"\sum") + r" \\")
    lines.append(r"\end{align*}")

    # --- Forma matricial ---
    lines.append(
        r"\textbf{De sumatorias a forma matricial:}\\[0.75em]"
        r"\begin{center}"
        r"$\displaystyle (\mathbf{X}^\top \mathbf{X})\,\boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y}$"
        r"\end{center}"
        r"\textbf{Donde:}"
    )
    lines.append(r"\begin{itemize}")
    lines.append(r"  \item $\mathbf{X}$ : Matriz de diseño con una columna de 1's para el intercepto.")
    lines.append(r"  \item $\mathbf{y}$ : Vector columna con los valores observados de la variable dependiente.")
    lines.append(r"  \item $\boldsymbol{\beta}$ : Vector de parámetros $(\beta_0, \beta_1, \ldots, \beta_m)^\top$.")
    lines.append(r"\end{itemize}")

    return "\n".join(lines)

