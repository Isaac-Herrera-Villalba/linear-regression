#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/regression/derivations.py
 ------------------------------------------------------------
 Genera las ecuaciones simbólicas y numéricas del desarrollo
 teórico de la regresión lineal múltiple, en notación LaTeX pura.
 ------------------------------------------------------------
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd


# ------------------------------------------------------------
def compute_sums(df: pd.DataFrame, y_col: str, x_cols: List[str]) -> Dict:
    """Calcula todas las sumatorias requeridas para las ecuaciones normales."""
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
    Devuelve las ecuaciones normales en forma simbólica LaTeX segura.
    Ejemplo:
      \sum y_i = n\beta_0 + \beta_1\sum x_i1 + \beta_2\sum x_i2 + ...
    """
    m = len(x_cols)
    eqs = []

    rhs = "n\\beta_0" + "".join([f" + \\beta_{j}\\sum x_i{j}" for j in range(1, m + 1)])
    eqs.append(f"\\sum y_i = {rhs}")

    for k in range(1, m + 1):
        rhs_k = f"\\beta_0\\sum x_i{k}" + "".join(
            [f" + \\beta_{j}\\sum x_i{k}x_i{j}" for j in range(1, m + 1)]
        )
        eqs.append(f"\\sum x_i{k}y_i = {rhs_k}")

    return eqs


# ------------------------------------------------------------
def normal_equations_numeric(sums: Dict, x_cols: List[str]) -> List[str]:
    """
    Sustituye los valores numéricos dentro de las ecuaciones normales.
    Todas las expresiones se devuelven en formato LaTeX seguro.
    """
    n, m = sums["n"], len(x_cols)
    sum_y = sums["sum_y"]
    sum_x = sums["sum_x"]
    sum_xx = sums["sum_xx"]
    sum_xy = sums["sum_xy"]

    lines = []

    # Ecuación para β₀
    rhs_terms = [f"{n}\\beta_0"] + [f"{sum_x[j - 1]}\\beta_{j}" for j in range(1, m + 1)]
    eq0 = f"\\sum y_i = " + " + ".join(rhs_terms)
    lines.append(eq0)

    # Ecuaciones para β₁..βₘ
    for k in range(1, m + 1):
        rhs_k = [f"{sum_x[k - 1]}\\beta_0"]
        rhs_k += [f"{sum_xx[k - 1][j - 1]}\\beta_{j}" for j in range(1, m + 1)]
        eq = f"\\sum x_i{k}y_i = " + " + ".join(rhs_k)
        lines.append(eq)

    return lines

