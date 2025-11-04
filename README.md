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

Este proyecto implementa un modelo de **Regresión Lineal** en **Python**, diseñado para analizar datasets en formatos **CSV**, **XLSX** o **ODS** y **ajustar una función matemática** que describa la relación entre una o más variables independientes (*predictoras*) y una variable dependiente (*objetivo*).

El sistema permite al usuario cargar datasets de diversa naturaleza (por ejemplo: economía, educación, salud, rendimiento académico, etc.), procesarlos automáticamente y generar un **modelo predictivo** basado en la **minimización del error cuadrático medio (MSE)**.

También ofrece la posibilidad de **visualizar los resultados** mediante gráficos, obtener el **coeficiente de determinación (R²)** y **generar reportes automáticos en formato PDF** con los cálculos, la ecuación resultante y las gráficas de ajuste.

---

### Características principales

- Lectura automática de archivos `.csv`, `.xlsx` y `.ods`.  
- Detección dinámica del número de columnas y tipos de datos.  
- Selección flexible de variables dependientes e independientes.  
- Cálculo de:
  - Pendiente y ordenada al origen.  
  - Error cuadrático medio (MSE).  
  - Coeficiente de determinación (R²).  
- Generación automática de reportes en **LaTeX → PDF**.  
- Soporte para múltiples datasets en un mismo archivo de configuración (`input.txt`).  
- Posibilidad de graficar el modelo de regresión lineal simple o múltiple.  

---

### Funcionamiento general

#### 1. Entrada

El archivo `input.txt` actúa como **fuente de configuración principal**.  
En él se definen las rutas de los datasets, las variables a analizar y los parámetros de salida del reporte.

##### Estructura general

| Clave | Descripción |
|-------|--------------|
| `DATASET` | Ruta del archivo de datos (`.ods`, `.xlsx`, `.csv`). |
| `SHEET` | Nombre de la hoja (en caso de archivos de hoja de cálculo). |
| `DEPENDENT_VAR` | Variable dependiente o de salida (Y). |
| `INDEPENDENT_VARS` | Variables predictoras (X₁, X₂, ...). |
| `REPORT` | Ruta y nombre del archivo PDF generado. |
| `PLOT` | Habilita la generación de gráficas (`true` / `false`). |

##### Ejemplo de configuración activa

```txt
DATASET=data/ventas.ods
SHEET=Sheet1
DEPENDENT_VAR=Ingresos
INDEPENDENT_VARS=Publicidad,Promociones
REPORT=output/reporte_ventas.pdf
PLOT=true

