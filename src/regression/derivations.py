#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/regression/derivations.py
 ------------------------------------------------------------
 Cálculo matemático y simbólico del desarrollo teórico de la
 regresión lineal múltiple. No genera LaTeX directamente, sino
 estructuras con la información necesaria para formatear el reporte.

 Incluye:
   - Cálculo de sumatorias.
   - Derivadas parciales (simbólicas).
   - Ecuaciones normales (simbólicas y numéricas).
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd


# ------------------------------------------------------------
def compute_sums(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Dict:
    """
    Calcula todas las sumatorias requeridas para construir las
    ecuaciones normales.

    Retorna un diccionario con:
      n, m, sum_y, sum_y2, sum_x, sum_xx, sum_xy
    """
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
def normal_equations_symbolic(x_cols: List[str]) -> List[str]:
    """
    Devuelve las ecuaciones normales en forma simbólica (sumatorias).
    """
    m = len(x_cols)
    eqs = []

    rhs = "n β₀" + "".join([f" + β{j} Σxᵢ{j}" for j in range(1, m + 1)])
    eqs.append(f"Σyᵢ = {rhs}")

    for k in range(1, m + 1):
        rhs_k = f"β₀ Σxᵢ{k}" + "".join(
            [f" + β{j} Σxᵢ{k}xᵢ{j}" for j in range(1, m + 1)]
        )
        eqs.append(f"Σxᵢ{k}yᵢ = {rhs_k}")

    return eqs


# ------------------------------------------------------------
def normal_equations_numeric(sums: Dict, x_cols: List[str]) -> List[str]:
    """
    Sustituye valores numéricos en las ecuaciones normales.

    Retorna una lista de cadenas numéricas representando las ecuaciones.
    """
    n, m = sums["n"], len(x_cols)
    sum_y = sums["sum_y"]
    sum_x = sums["sum_x"]
    sum_xx = sums["sum_xx"]
    sum_xy = sums["sum_xy"]

    lines = []

    # Ecuación para β₀
    rhs_terms = [f"{n}β₀"] + [f"{sum_x[j - 1]}β{j}" for j in range(1, m + 1)]
    eq0 = f"{sum_y} = " + " + ".join(rhs_terms)
    lines.append(eq0)

    # Ecuaciones para β₁..βₘ
    for k in range(1, m + 1):
        rhs_k = [f"{sum_x[k - 1]}β₀"]
        rhs_k += [f"{sum_xx[k - 1][j - 1]}β{j}" for j in range(1, m + 1)]
        eq = f"{sum_xy[k - 1]} = " + " + ".join(rhs_k)
        lines.append(eq)

    return lines

