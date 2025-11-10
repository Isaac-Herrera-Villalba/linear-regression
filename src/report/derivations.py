#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/regression/derivations.py
 ------------------------------------------------------------
 Genera el desarrollo completo de derivadas parciales y ecuaciones normales
 para el Paso 2 del reporte de Regresión Lineal.
 - Incluye explicaciones "Donde:" y significados de símbolos.
 - Incluye saltos automáticos en ecuaciones numéricas largas.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from src.report.report_latex import _fmt_number


# ------------------------------------------------------------
def _compute_sums(df: pd.DataFrame, y_col: str, x_cols: List[str]):
    """Calcula todas las sumatorias requeridas."""
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    X = df[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n, m = X.shape

    sum_y = float(np.sum(y))
    sum_y2 = float(np.sum(y * y))
    sum_x = [float(np.sum(X[:, j])) for j in range(m)]
    sum_xx = [[float(np.sum(X[:, j] * X[:, k])) for k in range(m)] for j in range(m)]
    sum_xy = [float(np.sum(X[:, j] * y)) for j in range(m)]

    return {
        "n": n, "m": m, "sum_y": sum_y, "sum_y2": sum_y2,
        "sum_x": sum_x, "sum_xx": sum_xx, "sum_xy": sum_xy
    }


# ------------------------------------------------------------
def _normal_equations_symbolic(x_cols: List[str]) -> List[str]:
    """Devuelve ecuaciones normales en forma simbólica (sumatorias)."""
    m = len(x_cols)
    eqs = []

    rhs = "n\\,\\beta_0" + "".join([f" + \\beta_{j} \\sum x_{{i,{j}}}" for j in range(1, m+1)])
    eqs.append(r"\sum y_i \;=\; " + rhs)

    for k in range(1, m+1):
        rhs_k = f"\\beta_0 \\sum x_{{i,{k}}}" + "".join(
            [f" + \\beta_{j} \\sum x_{{i,{k}}}x_{{i,{j}}}" for j in range(1, m+1)]
        )
        eqs.append(rf"\sum x_{{i,{k}}} y_i \;=\; " + rhs_k)
    return eqs


# ------------------------------------------------------------
def _normal_equations_numeric(sums: Dict, x_cols: List[str]) -> List[str]:
    """
    Sustituye los valores numéricos en las ecuaciones normales,
    aplicando saltos de línea automáticos para ecuaciones largas.
    """
    n, m = sums["n"], len(x_cols)
    sum_y = sums["sum_y"]
    sum_x = sums["sum_x"]
    sum_xx = sums["sum_xx"]
    sum_xy = sums["sum_xy"]

    lines = []

    def _wrap_latex(eq: str, width: int = 80) -> str:
        """Divide ecuaciones extensas en varias líneas."""
        chunks, current, length = [], [], 0
        for part in eq.split():
            if length + len(part) > width:
                chunks.append(" ".join(current))
                current, length = [part], len(part)
            else:
                current.append(part)
                length += len(part)
        if current:
            chunks.append(" ".join(current))
        return " \\\\\n".join(chunks)

    # β₀
    rhs_terms = [f"{_fmt_number(n)}\\,\\beta_0"] + [
        f"{_fmt_number(1.0)}\\,\\beta_{j}\\,{_fmt_number(sum_x[j-1])}" for j in range(1, m+1)
    ]
    eq0 = f"{_fmt_number(sum_y)} = " + " + ".join(rhs_terms)
    lines.append(_wrap_latex(eq0))

    # β₁...βₘ
    for k in range(1, m+1):
        rhs_k = [f"{_fmt_number(sum_x[k-1])}\\,\\beta_0"]
        rhs_k += [f"{_fmt_number(sum_xx[k-1][j-1])}\\,\\beta_{j}" for j in range(1, m+1)]
        eq = f"{_fmt_number(sum_xy[k-1])} = " + " + ".join(rhs_k)
        lines.append(_wrap_latex(eq))

    return lines


# ------------------------------------------------------------
def build_derivation_block(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> str:
    """
    Devuelve bloque LaTeX del desarrollo teórico completo:
      - Función objetivo con explicación de variables
      - Derivadas parciales
      - Ecuaciones normales (forma general y numérica)
      - Transición a la forma matricial con explicación
    """
    sums = _compute_sums(df, y_col, x_cols)
    lines: List[str] = []

    # --- Función objetivo general ---
    lines.append(r"\textbf{Función objetivo (caso general):}")
    lines.append(
        r"\[ S(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left(y_i - \Big(\beta_0 + \sum_{j=1}^{m} \beta_j x_{i,j}\Big)\right)^2 \]"
    )
    lines.append(r"\textbf{Donde:}")
    lines.append(r"\begin{itemize}")
    lines.append(r"  \item $y_i$ : Valor observado de la variable dependiente para la observación $i$.")
    lines.append(r"  \item $x_{i,j}$ : Valor de la variable independiente $j$ en la observación $i$.")
    lines.append(r"  \item $\beta_0$ : Término independiente (intercepto).")
    lines.append(r"  \item $\beta_j$ : Coeficiente asociado a la variable $x_j$.")
    lines.append(r"  \item $n$ : Número total de observaciones en el dataset.")
    lines.append(r"  \item $m$ : Número de variables independientes.")
    lines.append(r"\end{itemize}")

    lines.append(
        r"\textbf{Derivadas Parciales:}\\[0.75em]"
        r"\begin{center}"
        r"$\displaystyle \frac{\partial S}{\partial \beta_j}=0, \quad j=0,1,2,\dots,m.$"
        r"\end{center}"
        r"\\[-0.25em]"
        r"\textit{Estas condiciones representan el punto donde la función $S(\beta)$ alcanza su mínimo valor, "
        r"y constituyen las condiciones de optimalidad.}"
    )

    # --- Ecuaciones normales simbólicas ---
    lines.append(r"\paragraph{Ecuaciones normales (forma de sumas).}")
    sym = _normal_equations_symbolic(x_cols)
    lines.append(r"\begin{align*}")
    for s in sym:
        lines.append(s + r" \\")
    lines.append(r"\end{align*}")

    # --- Sustitución numérica ---
    lines.append(r"\paragraph{Sustitución numérica de sumatorias (con el dataset).}")
    num = _normal_equations_numeric(sums, x_cols)
    lines.append(r"\begin{align*}")
    for s in num:
        lines.append(r"\begin{split}" + s + r"\end{split}\\")
    lines.append(r"\end{align*}")

    # --- Transición matricial ---
    lines.append(
        r"\textbf{De sumatorias a forma matricial.} "
        r"El sistema anterior puede escribirse como:\\[0.75em]"
    )
    lines.append(r"\begin{center}")
    lines.append(r"$\displaystyle (\mathbf{X}^\top \mathbf{X})\,\boldsymbol{\beta}=\mathbf{X}^\top\mathbf{y}$")
    lines.append(r"\end{center}")
    lines.append(
        r"que es la condensación algebraica de las mismas ecuaciones normales.\\[0.5em]"
    )
    lines.append(r"\textbf{Donde:}")
    lines.append(r"\begin{itemize}")
    lines.append(r"  \item $\mathbf{X}$ : Matriz de diseño, compuesta por las variables independientes $x_{i,j}$ y una columna de 1's para el intercepto.")
    lines.append(r"  \item $\mathbf{y}$ : Vector columna con los valores observados de la variable dependiente.")
    lines.append(r"  \item $\boldsymbol{\beta}$ : Vector de parámetros del modelo $(\beta_0, \beta_1, \ldots, \beta_m)^\top$.")
    lines.append(r"\end{itemize}")

    return "\n".join(lines)

