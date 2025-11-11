#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/regression/linear_regression.py
 ------------------------------------------------------------
 Implementación de la regresión lineal múltiple mediante el método
 de Mínimos Cuadrados Ordinarios (OLS: Ordinary Least Squares).

 Este módulo realiza el cálculo completo del modelo lineal en forma
 matricial, partiendo de un conjunto de datos (X, y). Se basa en la
 ecuación normal para determinar el vector de coeficientes β y
 ofrece las matrices intermedias (XᵀX, Xᵀy) para su inclusión en el
 reporte técnico.

 Flujo matemático general:
   1. Modelo teórico:
        y = β₀ + β₁x₁ + β₂x₂ + ... + βₘxₘ
   2. Minimización del error cuadrático:
        min Σ (yᵢ - (β₀ + Σ βⱼxᵢⱼ))²  ⇒  ∂S/∂βⱼ = 0
   3. Solución matricial (ecuaciones normales):
        β = (XᵀX)⁻¹ Xᵀy
      * Si XᵀX es singular, se aplica pseudoinversa.
   4. Predicción para una instancia:
        ŷ = β₀ + β₁x₁ + ... + βₘxₘ

 Devuelve estructuras con resultados numéricos y matrices relevantes
 para el reporte en formato LaTeX.
 ------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ============================================================
# === ESTRUCTURA DE RESULTADOS ===============================
# ============================================================
@dataclass
class RegressionResult:
    """
    Contenedor estructurado de resultados de una regresión lineal múltiple.

    Atributos
    ---------
    feature_names : List[str]
        Lista de nombres de las variables independientes (X₁..Xₘ).
    beta : np.ndarray
        Vector columna de coeficientes β, incluyendo el intercepto.
    intercept : float
        Término independiente β₀.
    coefficients : Dict[str, float]
        Mapeo entre nombres de variables y sus coeficientes βⱼ.
    r2 : Optional[float]
        Coeficiente de determinación R² del modelo (puede ser None si no aplica).
    X : np.ndarray
        Matriz de diseño, incluyendo la columna de 1s para el intercepto.
    y : np.ndarray
        Vector columna de valores observados (variable dependiente).
    XtX : np.ndarray
        Producto XᵀX, utilizado para las ecuaciones normales.
    Xty : np.ndarray
        Producto Xᵀy, también parte de las ecuaciones normales.
    """
    feature_names: List[str]
    beta: np.ndarray
    intercept: float
    coefficients: Dict[str, float]
    r2: Optional[float]
    X: np.ndarray
    y: np.ndarray
    XtX: np.ndarray
    Xty: np.ndarray


# ============================================================
# === FUNCIONES AUXILIARES ===================================
# ============================================================
def _add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Agrega una columna de unos (1s) a la izquierda de la matriz X
    para representar el intercepto β₀.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de variables independientes (sin intercepto).

    Retorna
    -------
    np.ndarray
        Matriz extendida [1 | X].
    """
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])


def _safe_pinv_solve(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula el vector β mediante las ecuaciones normales:

        β = (XᵀX)⁻¹ Xᵀy

    Si la matriz (XᵀX) no es invertible o está mal condicionada,
    se emplea la pseudoinversa de Moore–Penrose.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de diseño con intercepto.
    y : np.ndarray
        Vector columna de valores dependientes.

    Retorna
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - β (vector de coeficientes),
        - XᵀX,
        - Xᵀy.
    """
    Xt = X.T
    XtX = Xt @ X
    Xty = Xt @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ Xty
    return beta, XtX, Xty


# ============================================================
# === FUNCIÓN PRINCIPAL DE REGRESIÓN ==========================
# ============================================================
def run_linear_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str]
) -> RegressionResult:
    """
    Ejecuta el proceso completo de regresión lineal múltiple.

    Incluye:
      - Conversión de datos a tipo float.
      - Construcción de la matriz de diseño con intercepto.
      - Resolución de β mediante ecuaciones normales.
      - Cálculo del coeficiente de determinación R².

    Parámetros
    ----------
    df : pd.DataFrame
        Conjunto de datos numéricos (ya preprocesado).
    y_col : str
        Nombre de la variable dependiente.
    x_cols : List[str]
        Lista con las variables independientes.

    Retorna
    -------
    RegressionResult
        Estructura con todos los resultados de la regresión.
    """
    # Conversión de columnas relevantes a float
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    X_raw = df[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    # Construcción de la matriz de diseño
    X = _add_intercept(X_raw)

    # Resolución de las ecuaciones normales
    beta, XtX, Xty = _safe_pinv_solve(X, y)

    # Descomposición de resultados
    intercept = float(beta[0])
    coefs = {name: float(b) for name, b in zip(x_cols, beta[1:])}

    # Cálculo del coeficiente de determinación R²
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


# ============================================================
# === PREDICCIÓN PARA INSTANCIAS ==============================
# ============================================================
def predict(beta: np.ndarray, x_vector: List[float]) -> float:
    """
    Calcula la predicción para una instancia específica.

    Aplica el modelo lineal:
        ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₘxₘ

    Parámetros
    ----------
    beta : np.ndarray
        Vector de coeficientes (incluye el intercepto en posición 0).
    x_vector : List[float]
        Valores numéricos de las variables independientes.

    Retorna
    -------
    float
        Valor predicho ŷ.
    """
    x = np.array([1.0] + [float(v) for v in x_vector], dtype=float)
    return float(x @ beta)
# ---------------------------------------------------------------------------

