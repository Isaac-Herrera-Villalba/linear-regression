# Makefile
# -----------------------------------------------------------------------
# Descripción:
# Automatiza el flujo completo del proyecto **Linear Regression**.
# Este proyecto implementa un sistema de **Regresión Lineal** (simple y múltiple)
# desarrollado en Python, que genera automáticamente un **reporte en PDF**
# con los resultados teóricos, matrices y ecuaciones calculadas.
#
# Estructura general del flujo:
#   1. Crea un entorno virtual e instala dependencias necesarias (make env).
#   2. Ejecuta el análisis definido en input.txt (make run).
#   3. Compila el archivo LaTeX a PDF (make latex o make pdf).
#   4. Visualiza el reporte (make view).
#
# Incluye compatibilidad con datasets .csv, .xlsx y .ods, y genera un
# reporte matemático completo con:
#   - Modelo lineal general.
#   - Derivadas parciales y ecuaciones normales.
#   - Matrices (X, Xᵀ, XᵀX, Xᵀy, β).
#   - Sustitución de valores y predicciones numéricas.
#
# Dependencias gestionadas en requirements.txt
# -----------------------------------------------------------------------

PYTHON         = python3
SRC_DIR        = src
SRC_DIR_2      = core
SRC_DIR_3      = data_extractor
SRC_DIR_4      = regression
SRC_DIR_5      = report
OUT_DIR        = output
INPUT_FILE     = input.txt
MAIN_FILE      = $(SRC_DIR)/main.py
PDF_READER     = okular
PYTHON_MODULES = __pycache__

LATEX_REPORT_NAME = reporte

EXT           = ods

DATASET       = ejemplo
DATASET_PATH  = data/$(DATASET).$(EXT)

VENV_DIR      = .venv
PYTHON_VENV   = $(VENV_DIR)/bin/python
PIP_VENV      = $(VENV_DIR)/bin/pip

PDF_FILE      = $(OUT_DIR)/$(LATEX_REPORT_NAME).pdf
TEX_FILE      = $(OUT_DIR)/$(LATEX_REPORT_NAME).tex

.PHONY: all help env run latex pdf view clean full show_dataset

# -----------------------------------------------------------------------
all: help

help:
	@echo "Comandos disponibles:"
	@echo "  make env    -> Crea entorno virtual e instala dependencias"
	@echo "  make run    -> Ejecuta la regresión lineal (genera .tex y PDF automático)"
	@echo "  make latex  -> Compila manualmente el archivo .tex con LaTeX"
	@echo "  make pdf    -> Alias de make run (genera reporte PDF desde input.txt)"
	@echo "  make view   -> Abre el PDF resultante"
	@echo "  make clean  -> Elimina archivos temporales y auxiliares de LaTeX"
	@echo "  make full   -> Ejecuta todo el flujo (env + run + latex + view)"
	@echo "---------------------------------------------------------------"

# -----------------------------------------------------------------------
env:
	@echo "=== Creando entorno virtual (.venv) ==="
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "[OK] Entorno virtual creado en $(VENV_DIR)"; \
	else \
		echo "[INFO] Entorno virtual ya existe."; \
	fi
	@echo "=== Instalando dependencias ==="
	$(PIP_VENV) install --upgrade pip >/dev/null
	$(PIP_VENV) install -r requirements.txt
	@echo "[OK] Dependencias instaladas correctamente."

# -----------------------------------------------------------------------
run:
	@echo "=== Ejecutando regresión lineal ==="
	mkdir -pv $(OUT_DIR)/
	$(PYTHON_VENV) -m $(SRC_DIR).main $(INPUT_FILE)
	@echo "[OK] Ejecución completada. Si el .tex fue generado, puedes compilarlo con 'make latex'"

pdf: run

# -----------------------------------------------------------------------
latex:
	@echo "=== Compilando LaTeX manualmente con pdflatex ==="
	@if [ -f "$(TEX_FILE)" ]; then \
		cd $(OUT_DIR) && \
		pdflatex -interaction=nonstopmode $(notdir $(TEX_FILE)) >/dev/null 2>&1; \
		echo "[OK] Compilación LaTeX completada: $(PDF_FILE)"; \
	else \
		echo "[ERROR] No se encontró $(TEX_FILE). Ejecuta primero 'make run'."; \
	fi

# -----------------------------------------------------------------------
view:
	@echo "Abriendo PDF con Okular..."
	$(PDF_READER) $(PDF_FILE) &

# -----------------------------------------------------------------------
full: run latex view show_dataset

# -----------------------------------------------------------------------
clean:
	@echo "Eliminando archivos generados..."
	rm -rf $(OUT_DIR)/
	rm -rf $(SRC_DIR)/$(PYTHON_MODULES)
	rm -rf $(SRC_DIR)/$(SRC_DIR_2)/$(PYTHON_MODULES)
	rm -rf $(SRC_DIR)/$(SRC_DIR_2)/$(SRC_DIR_3)/$(PYTHON_MODULES)
	rm -rf $(SRC_DIR)/$(SRC_DIR_4)/$(PYTHON_MODULES)
	rm -rf $(SRC_DIR)/$(SRC_DIR_5)/$(PYTHON_MODULES)
	
	rm -f data/.~lock.*
	@echo "Limpieza completada."

