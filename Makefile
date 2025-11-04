# Makefile
# -----------------------------------------------------------------------
# Descripción:
# Automatiza la ejecución del proyecto bayesian-classification.
# Incluye la creación del entorno virtual, instalación de dependencias,
# ejecución del clasificador bayesiano y compilación del reporte
# en PDF mediante LaTeX.
# -----------------------------------------------------------------------

PYTHON        = python3
SRC_DIR       = src
OUT_DIR       = output
INPUT_FILE    = input.txt
MAIN_FILE     = $(SRC_DIR)/main.py
PDF_READER    = okular

SPREADSHEET = libreoffice
EXT = ods

DATASET = peliculas
DATASET_PATH = data/$(DATASET).$(EXT)

VENV_DIR      = .venv
PYTHON_VENV   = $(VENV_DIR)/bin/python
PIP_VENV      = $(VENV_DIR)/bin/pip

PDF_FILE      = $(OUT_DIR)/reporte_1.pdf
TEX_FILE      = $(OUT_DIR)/reporte_1.tex

.PHONY: all help env run latex pdf view clean full show_dataset

# -----------------------------------------------------------------------
all: help

help:
	@echo "Comandos disponibles:"
	@echo "  make env    -> Crea entorno virtual e instala dependencias"
	@echo "  make run    -> Ejecuta el clasificador bayesiano (genera .tex y PDF automático)"
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
	@echo "=== Ejecutando clasificador bayesiano ==="
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
show_dataset:
	@echo "Muestra el dataset utilizado"
	@if [ -n "$$DISPLAY" ]; then \
		echo "[OK] Abriendo $(DATASET_PATH) con LibreOffice..."; \
		$(SPREADSHEET) "$(DATASET_PATH)" & \
	else \
		echo "[ERROR] No hay entorno gráfico (DISPLAY). Ejecuta este comando dentro de una sesión con GUI."; \
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
	find . -type f \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" -o -name "*.tex" -o -name "*.synctex.gz" \) -delete
	rm -rf $(SRC_DIR)/__pycache__
	rm -f data/.~lock.*
	@echo "Limpieza completada."

