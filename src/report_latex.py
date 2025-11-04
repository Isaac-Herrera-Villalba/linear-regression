#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 src/report_latex.py
 ------------------------------------------------------------
 Descripción:
 Genera un reporte en formato LaTeX para el proceso de
 clasificación Bayesiana, incluyendo:
  - Vista previa del dataset (con límite de filas/columnas)
  - Cálculo paso a paso con fracciones y decimales en una sola línea
  - Texto explicativo y conclusión automática
  - Puntos decimales uniformes
  - Saltos automáticos en ecuaciones largas
"""

from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Plantilla base en LaTeX con secciones para datos, tablas y resultados
latex_template = r"""
\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage[spanish]{babel}
\decimalpoint
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{breqn}
\usepackage{microtype}
\usepackage{ragged2e}
\usepackage{siunitx}

\sisetup{output-decimal-marker = {.}}

\begin{document}
\RaggedRight

\section*{Clasificación Bayesiana}

\subsection*{Resumen del dataset}
Filas: %(rows)d, Columnas: %(cols)d\\
Atributos usados: %(attrs)s\\
Columna de clase: %(target)s

\subsection*{Vista previa del dataset}
%(dataset_table)s

\subsection*{Probabilidades a priori}
\begin{tabular}{l r}
\toprule
Clase & $P(y)$ \\
\midrule
%(priors_rows)s
\bottomrule
\end{tabular}

\subsection*{Verosimilitudes $P(A{=}v|y)$}
%(likelihoods_tables)s

\subsection*{Cálculo para la instancia}
Instancia: %(instance)s

%(trace)s

\subsection*{Posteriores normalizados}
\begin{tabular}{l r}
\toprule
Clase & $P(y|\mathbf{x})$ \\
\midrule
%(post_rows)s
\bottomrule
\end{tabular}

\subsection*{Predicción}
La clase predicha es: \textbf{%(pred)s}.

\end{document}
"""


# Genera una tabla de vista previa del dataset en formato LaTeX
def dataset_preview_table(df: pd.DataFrame, max_rows: int = 15, max_cols: int = 8) -> str:
    rows, cols = df.shape
    truncated = rows > max_rows or cols > max_cols
    df_disp = df.iloc[:max_rows, :max_cols].copy()
    df_disp.columns = [str(c).replace("_", "\\_") for c in df_disp.columns]
    # Formato de tabla
    header_fmt = " ".join(["l"] * len(df_disp.columns))
    lines = [
        "\\begin{tabular}{" + header_fmt + "}",
        "\\toprule",
        " & ".join(df_disp.columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in df_disp.iterrows():
        vals = [str(v).replace("_", "\\_") for v in row.values]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    # Mensaje si la tabla fue truncada
    if truncated:
        lines.append(
            "\\\\[0.3em]\\textit{Nota: el dataset contiene más filas o columnas. "
            "Consulta el archivo original para ver la tabla completa.}"
        )
    return "\n".join(lines)


# Convierte un DataFrame a una tabla LaTeX con título
def _tabular_from_df(df: pd.DataFrame, title: str) -> str:
    df = df.copy()
    df.index = [str(i) for i in df.index]
    df.columns = [str(c) for c in df.columns]
    header = " ".join(["l"] + ["r"] * len(df.columns))
    # Construcción de tabla
    lines = [f"\\subsubsection*{{{title}}}", f"\\begin{{tabular}}{{{header}}}", "\\toprule"]
    lines.append("Clase & " + " & ".join(df.columns) + " \\\\")
    lines.append("\\midrule")
    for idx, row in df.iterrows():
        vals = " & ".join(f"{float(v):.6f}".replace(",", ".") for v in row.values)
        lines.append(f"{idx} & {vals} \\\\")
    lines.append("\\bottomrule\n\\end{tabular}")
    return "\n".join(lines)


# Construye las ecuaciones del cálculo Bayesiano paso a paso
def build_trace(
    priors: Dict[str, float],
    conds: Dict[str, pd.DataFrame],
    instance: Dict[str, str],
    posteriors: Dict[str, float] | None = None,
    raw_counts: Dict[str, pd.DataFrame] | None = None
) -> str:
    """
    Construye las expresiones del cálculo paso a paso:
    P(c) × P(A=v|c) = fracciones × fracciones = decimales × decimales = resultado
    Usa breqn (dmath*) para ajuste automático de ecuaciones largas.
    """
    if not instance:
        return "\\textit{No se proporcionó una instancia.}"

    desc_parts = [f"{a} = {v}" for a, v in instance.items()]
    instance_desc = ", ".join(desc_parts)
    lines = [
        f"La clasificación de una muestra con {instance_desc} se puede predecir de la siguiente forma:",
        ""
    ]

    # Cálculo del total general para fracciones
    total_general = None
    if raw_counts:
        any_table = next(iter(raw_counts.values()))
        total_general = int(any_table.sum().sum())

    # Generación de las fórmulas por clase
    for c, p_y in priors.items():
        frac_parts = []

        # P(y) como fracción
        if total_general and raw_counts:
            class_count = 0
            for table in raw_counts.values():
                if c in table.index:
                    class_count += int(table.loc[c].sum())
            class_count //= len(raw_counts)
            frac_parts.append(f"\\frac{{{class_count}}}{{{total_general}}}")

        # P(A=v|y) como fracciones
        if raw_counts:
            for a, v in instance.items():
                tbl = raw_counts[a]
                if c in tbl.index and str(v) in tbl.columns:
                    num = int(tbl.loc[c, str(v)])
                    den = int(tbl.loc[c].sum())
                    frac_parts.append(f"\\frac{{{num}}}{{{den}}}")

        # Versión decimal del producto
        parts = [f"{p_y:.4f}".replace(",", ".")]
        mult_vals = [p_y]
        for a, v in instance.items():
            p = conds[a].loc[c, str(v)] if (c in conds[a].index and str(v) in conds[a].columns) else 0.0
            parts.append(f"{p:.4f}".replace(",", "."))
            mult_vals.append(p)

        # Resultado final del producto
        prod = 1
        for val in mult_vals:
            prod *= val
        prod_str = str(round(prod, 6)).replace(",", ".")

        # Construcción de la línea completa
        frac_expr = " \\times ".join(frac_parts) if frac_parts else ""
        dec_expr = " \\times ".join(parts)

        base_expr = (
            f"P({c}) \\times " +
            " \\times ".join([f"P({a}={v}|{c})" for a, v in instance.items()]) + " = " +
            (frac_expr + " = " if frac_expr else "") +
            f"{dec_expr} = {prod_str}"
        )

        # Si la ecuación es muy larga, usa breqn y reduce ligeramente la fuente
        if len(base_expr) > 120:
            line_expr = (
                "\\begingroup\n"
                "\\scriptsize\n"
                "\\begin{dmath*}\n"
                f"{base_expr}\n"
                "\\end{dmath*}\n"
                "\\endgroup"
            )
        else:
            line_expr = f"\\[{base_expr}\\]"

        # Agregamos la ecuación
        lines.append(line_expr)

    # Conclusión final
    if posteriors:
        best = max(posteriors, key=posteriors.get)
        lines.append(f"Por lo tanto, la clase predicha para esta instancia es \\textbf{{{best}}}.")

    return "\n".join(lines)

# Genera el archivo TEX y compila a PDF con pdflatex/xelatex
def render_pdf(
    out_pdf: str,
    df: pd.DataFrame,
    target: str,
    attrs: List[str],
    priors: Dict[str, float],
    conds: Dict[str, pd.DataFrame],
    instance: Dict[str, str],
    posteriors: Dict[str, float],
    raw_counts: Dict[str, pd.DataFrame] | None = None,
):
    # Preparación de datos y tablas
    rows, cols = df.shape
    priors_rows = "\n".join(f"{c} & {p:.6f}".replace(",", ".") + " \\\\" for c, p in priors.items())
    like_tables = "\n\n".join(_tabular_from_df(conds[a], f"Atributo: {a}") for a in attrs)
    post_rows = "\n".join(f"{c} & {p:.6f}".replace(",", ".") + " \\\\" for c, p in posteriors.items())
    pred = max(posteriors, key=posteriors.get) if posteriors else "—"
    dataset_table = dataset_preview_table(df)

    # Inserta los valores en la plantilla LaTeX
    tex_content = latex_template % {
        "rows": rows,
        "cols": cols,
        "attrs": ", ".join(attrs),
        "target": target,
        "dataset_table": dataset_table,
        "priors_rows": priors_rows,
        "likelihoods_tables": like_tables,
        "instance": ", ".join(f"{k}={v}" for k, v in instance.items()),
        "trace": build_trace(priors, conds, instance, posteriors, raw_counts),
        "post_rows": post_rows,
        "pred": pred,
    }

    # Escribe el archivo .tex y crea carpeta si no existe
    out_pdf_path = Path(out_pdf)
    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path = out_pdf_path.with_suffix(".tex")
    tex_path.write_text(tex_content, encoding="utf-8")

    # Intenta compilar a PDF con xelatex o pdflatex
    for binname in ("xelatex", "pdflatex"):
        try:
            subprocess.run(
                [binname, "-interaction=nonstopmode", tex_path.name],
                cwd=tex_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return
        except Exception:
            continue

# Aviso si no se logra compilar el PDF
    print(f"[WARN] No se pudo compilar PDF, se dejó el archivo TEX en: {tex_path}")

