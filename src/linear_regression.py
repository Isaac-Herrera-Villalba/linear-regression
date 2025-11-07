#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/linear_regression.py
 ------------------------------------------------------------
 Descripción:

 Módulo principal para calcular la Regresión Lineal (múltiple) por
 mínimos cuadrados ordinarios (OLS). Dado un DataFrame, una
 variable dependiente (y) y un conjunto de variables independientes
 (X₁..Xₘ), construye la matriz de diseño, resuelve β mediante la
 ecuación normal y, opcionalmente, calcula predicciones para
 instancias específicas.

 Flujo matemático (alineado al PDF):
  1) Modelo:          y = β₀ + β₁ x₁ + ... + βₘ xₘ
  2) Costo S(β):      min Σ (yᵢ - (β₀ + Σ βⱼ xᵢⱼ))²  ⇒  ∂S/∂βⱼ = 0
  3) Ecuación normal: β = (Xᵀ X)⁻¹ Xᵀ y  (usa pseudoinversa si XᵀX es singular)
  4) Predicción:      ŷ = β₀ + β₁ x₁ + ... + βₘ xₘ  (sustitución de X en Y)

 ------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class RegressionResult:
    """Estructura de resultados de la regresión lineal."""
    feature_names: List[str]                   # Nombres de X en el orden de la matriz de diseño
    beta: np.ndarray                           # Vector de coeficientes β (incluye intercepto)
    intercept: float                           # β₀
    coefficients: Dict[str, float]             # {nombre_variable: β_j}
    r2: Optional[float]                        # Coeficiente de determinación (si aplica)
    X: np.ndarray                              # Matriz de diseño (con columna de 1s)
    y: np.ndarray                              # Vector objetivo
    XtX: np.ndarray                            # Xᵀ X (para el reporte)
    Xty: np.ndarray                            # Xᵀ y (para el reporte)


def _add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Agrega una columna de 1s a X para el intercepto β₀.
    """
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])


def _safe_pinv_solve(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resuelve β usando las ecuaciones normales:
        β = (Xᵀ X)⁻¹ Xᵀ y
    Si XᵀX es singular o mal condicionada, usa pseudoinversa.

    Retorna:
        beta, XtX, Xty
    """
    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ Xty
    return beta, XtX, Xty


def run_linear_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str]
) -> RegressionResult:
    """
    Ejecuta la Regresión Lineal:
      - Construye X (solo numérica), agrega intercepto.
      - Resuelve β con ecuaciones normales.
      - Calcula R².

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset ya filtrado y convertido a numérico en columnas relevantes.
    y_col : str
        Nombre de la variable dependiente.
    x_cols : List[str]
        Lista en orden de variables independientes.

    Retorna
    -------
    RegressionResult
        Resultados completos, matrices y coeficientes.
    """
    # Extrae y, X y garantiza tipo float
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    X_raw = df[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Construye matriz de diseño con intercepto
    X = _add_intercept(X_raw)

    # Resuelve ecuación normal (con fallback a pseudoinversa)
    beta, XtX, Xty = _safe_pinv_solve(X, y)

    # Desglosa resultados
    intercept = float(beta[0])
    coefs = {name: float(b) for name, b in zip(x_cols, beta[1:])}

    # R²
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return RegressionResult(
        feature_names=x_cols,
        beta=beta,
        intercept=intercept,
        coefficients=coefs,
        r2=r2,
        X=X,
        y=y,
        XtX=XtX,
        Xty=Xty
    )


def predict(beta: np.ndarray, x_vector: List[float]) -> float:
    """
    Realiza la predicción para una sola instancia:
        ŷ = β₀ + Σ βⱼ xⱼ

    Parámetros
    ----------
    beta : np.ndarray
        Vector de coeficientes β (incluye intercepto en la posición 0).
    x_vector : List[float]
        Valores numéricos de las variables independientes en el mismo orden
        usado para entrenar (feature_names).

    Retorna
    -------
    float
        Predicción ŷ.
    """
    x = np.array([1.0] + [float(v) for v in x_vector], dtype=float)
    return float(x @ beta)
# ---------------------------------------------------------------------------

