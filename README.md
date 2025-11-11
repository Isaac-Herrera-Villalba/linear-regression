# Linear Regression
## Inteligencia Artificial · 2026-1  
### Equipo 6

**Integrantes**
- Fuentes Jiménez Yasbeck Dailanny  
- Herrera Villalba Isaac  
- Juárez García Yuliana  
- Ruiz Bastián Óscar  
- Sampayo Aguilar Cinthia Gloricel  
- Velarde Valencia Josue  

---

## Descripción general

Sistema de **Regresión Lineal (simple y múltiple)** implementado en **Python**, diseñado para el análisis de datasets numéricos y la generación automática de un **reporte técnico en formato PDF** con los resultados teóricos y computacionales del modelo.

El programa emplea el método de **Mínimos Cuadrados Ordinarios (Ordinary Least Squares, OLS)** para estimar el vector de parámetros β del modelo lineal general:

\[
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
\]

El reporte resultante integra tanto la formulación matemática como las etapas de resolución y predicción, proporcionando una descripción completa del proceso de ajuste.

---

## Características principales

- Soporte para archivos de entrada en formato **CSV**, **XLSX** y **ODS**.  
- Detección automática del bloque de datos y encabezados válidos.  
- Selección flexible de variables dependientes (Y) e independientes (X₁, X₂, …, Xₘ).  
- Procesamiento automático de valores numéricos y eliminación de filas no convertibles.  
- Ejecución automática de:
  - **Regresión lineal simple** cuando se especifica una sola variable X.  
  - **Regresión lineal múltiple** cuando se definen dos o más variables X.  
- Resolución mediante la ecuación normal:  
  \[
  \boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
  \]
- Generación de un único reporte PDF con todas las instancias y datasets procesados.  
- Representación matricial detallada (X, Xᵀ, XᵀX, Xᵀy, (XᵀX)⁻¹, β).  
- Cálculo del **coeficiente de determinación (R²)** e interpretación del grado de ajuste.  
- Predicción numérica automática de ŷ mediante sustitución directa de valores.

---

## Funcionamiento general

El sistema se ejecuta a partir de un archivo de configuración `input.txt`, el cual define los parámetros de análisis y las instancias de evaluación.

Cada bloque de configuración contiene los siguientes campos:

| Clave | Descripción |
|-------|--------------|
| `DATASET` | Ruta del archivo de datos (`.ods`, `.xlsx`, `.csv`). |
| `SHEET` / `HOJA` | Nombre de la hoja dentro del archivo (opcional). |
| `DEPENDENT_VARIABLE` | Variable dependiente o de salida (Y). |
| `INDEPENDENT_VARIABLES` | Lista separada por comas de variables predictoras (X₁, X₂, …). |
| `USE_ALL_ATTRIBUTES` | Si es `true`, utiliza todas las columnas excepto Y. |
| `REPORT` | Ruta y nombre del archivo PDF a generar. |
| `INSTANCE` | Conjunto de valores numéricos para generar predicciones ŷ. |

El sistema permite incluir varios bloques `DATASET=` dentro del mismo archivo, generando un **reporte consolidado** con una sección independiente por dataset.

---

## Contenido del reporte generado

Cada sección del PDF incluye los siguientes apartados:

1. **Modelo lineal general**  
   Presentación de la ecuación base y descripción de las variables.  

2. **Desarrollo teórico**  
   Formulación de la función objetivo, derivadas parciales y ecuaciones normales.  

3. **Forma matricial**  
   Representación de las matrices involucradas y solución del vector β.  

4. **Coeficiente de determinación (R²)**  
   Evaluación del ajuste y tabla interpretativa del nivel de correlación.  

5. **Sustitución numérica y predicción (ŷ)**  
   Sustitución de los valores de X en la ecuación final del modelo y cálculo del resultado.  

---

## Ejemplo de configuración

```txt
DATASET = data/regresion_6d.ods
SHEET = Sheet1
DEPENDENT_VARIABLE = Y
INDEPENDENT_VARIABLES = X1, X2, X3, X4, X5
USE_ALL_ATTRIBUTES = false
REPORT = output/reporte.pdf

INSTANCE:
  X1 = 27
  X2 = 7
  X3 = 3
  X4 = 2
  X5 = 4

INSTANCE:
  X1 = 30
  X2 = 8
  X3 = 4
  X4 = 3
  X5 = 6

