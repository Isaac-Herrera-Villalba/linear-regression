# Inteligencia Artificial, 2026-1
## Equipo 6

### Integrantes
· Fuentes Jiménez Yasbeck Dailanny  
· Herrera Villalba Isaac  
· Juárez García Yuliana  
· Ruiz Bastián Óscar  
· Sampayo Aguilar Cinthia Gloricel  
· Velarde Valencia Josue  

---

## Proyecto: Linear Regression

### Descripción general

Este proyecto implementa un sistema de **Regresión Lineal (múltiple)** en **Python**, diseñado para analizar datasets en formato **CSV**, **XLSX** o **ODS**.  
El programa ajusta una **función matemática del tipo**:

\[
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_mx_m
\]

mediante el método de **Mínimos Cuadrados Ordinarios (Ordinary Least Squares, OLS)**.  
A partir del modelo calculado, se generan **predicciones numéricas** y un **reporte en PDF** con los pasos teóricos y cálculos detallados.

---

### Características principales

- Lectura automática de archivos `.csv`, `.xlsx` y `.ods`.  
- Detección automática del bloque de datos, encabezados y tipos válidos.  
- Selección flexible de:
  - Variable dependiente (*Y*).  
  - Variables independientes (*X₁, X₂, ..., Xₙ*).  
- Procesamiento automático de valores numéricos.  
- Cálculo del **vector de coeficientes β** usando la ecuación normal:
  \[
  \boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
  \]
- Obtención de métricas de ajuste:
  - **Intercepto (β₀)**  
  - **Pendientes (β₁..βₙ)**  
  - **Coeficiente de determinación (R²)**  
- Soporte para múltiples instancias (valores de entrada) definidas en `input.txt`.  
- Generación de **un único reporte PDF** con todos los pasos matemáticos por cada instancia:  
  1. Modelo general.  
  2. Función objetivo.  
  3. Ecuaciones normales.  
  4. Sustitución de valores y predicción numérica (ŷ).  

---

### Funcionamiento general

#### 1️⃣ Entrada: archivo `input.txt`

El archivo `input.txt` actúa como **fuente de configuración principal** del sistema.  
En él se definen los parámetros de análisis, el dataset a utilizar y las instancias a evaluar.

##### Estructura general

| Clave | Descripción |
|-------|--------------|
| `DATASET` | Ruta del archivo de datos (`.ods`, `.xlsx`, `.csv`). |
| `HOJA` / `SHEET` | Nombre de la hoja (en caso de archivos Excel o LibreOffice). |
| `DEPENDENT_VARIABLE` | Variable dependiente o de salida (*Y*). |
| `INDEPENDENT_VARIABLES` | Variables independientes o predictoras (*X₁, X₂, …*). |
| `USE_ALL_ATTRIBUTES` | Si es `true`, usa todas las columnas excepto *Y*. |
| `REPORT` | Ruta y nombre del archivo PDF a generar. |
| `INSTANCE` | Una o más instancias con valores para predecir ŷ. |

##### Ejemplo de configuración activa

```txt
DATASET=data/regresion_lineal_2_y_3_dimensiones.ods
HOJA=Hoja5
DEPENDENT_VARIABLE=y
INDEPENDENT_VARIABLES=x1, x2
USE_ALL_ATTRIBUTES=false
REPORT=output/reporte_2D.pdf

# --- Instancia 1 ---
INSTANCE:
  x1=8
  x2=10

# --- Instancia 2 ---
INSTANCE:
  x1=12
  x2=15

