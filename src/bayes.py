#!/usr/bin/env python3
"""
 src/bayes.py
 ------------------------------------------------------------
 Descripción:

 Módulo principal del sistema de Clasificación Bayesiana. 
 Procesa un conjunto de datos de entrada, calcula las 
 probabilidades a priori y condicionales de las clases y
 atributos, evalúa nuevas instancias mediante el modelo 
 Naive Bayes y genera un conjunto estructurado de resultados
 con las tablas de frecuencias, probabilidades y valores 
 posteriores para cada clase.
 ------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

# Estructura de datos para almacenar los resultados del algoritmo Bayesiano
@dataclass
class BayesResult:
    priors: Dict[str, float] # Probabilidades a priori de cada clase
    cond_tables: Dict[str, pd.DataFrame] # Tablas de probabilidad condicional
    raw_counts: Dict[str, pd.DataFrame] # Tablas con los conteos originales
    scores: Dict[str, float] # Valor sin normalizar de cada clase
    posteriors: Dict[str, float] # Probabilidades a posteriori normalizadas

# Calcula las probabilidades a priori de cada clase P(Y)
def compute_priors(df: pd.DataFrame, target: str) -> Dict[str, float]:
    total = len(df)
    counts = df[target].value_counts(dropna=False)
    return {str(k): v / total for k, v in counts.items()}

# Genera las tablas de probabilidad condicional P(X|Y) y sus conteos
def conditional_tables(df: pd.DataFrame, target: str, attrs: List[str], alpha: float = 0.0):
    cond_probs, raw_counts = {}, {}
    for attr in attrs:
        tmp = df[[attr, target]].astype(str).value_counts().rename("count").reset_index()
        pivot = tmp.pivot(index=target, columns=attr, values="count").fillna(0)
        raw_counts[attr] = pivot.copy()

        if alpha > 0:
            pivot = pivot + alpha
        prob = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
        cond_probs[attr] = prob
    return cond_probs, raw_counts

# Evalúa una instancia aplicando la regla de Bayes
def evaluate_instance(priors, conds, instance):
    scores = {}
    for c, p_y in priors.items():
        prod = p_y
        for attr, val in instance.items():
            tbl = conds[attr]
            p = float(tbl.loc[c].get(str(val), 0.0)) if c in tbl.index else 0.0
            prod *= p
        scores[c] = prod
    total = sum(scores.values()) # Normalización para obtener las probabilidades a posteriori
    post = {c: (s / total if total > 0 else 0.0) for c, s in scores.items()}
    return scores, post

# Función principal: ejecuta el flujo completo del clasificador Naive Bayes
def run_naive_bayes(df, target, attrs, instance, alpha=0.0) -> BayesResult:
    priors = compute_priors(df, target) 
    conds, raw_counts = conditional_tables(df, target, attrs, alpha)
    scores, post = evaluate_instance(priors, conds, instance)
    return BayesResult(priors, conds, raw_counts, scores, post)
# ---------------------------------------------------------------------------------

