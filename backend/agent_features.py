"""
Capacidades del agente: todo lo que decide QUÉ responder y CÓMO mostrarlo.
"""
from __future__ import annotations

import json
import logging
import os
import re
from decimal import Decimal
from typing import Any, Optional, AsyncGenerator

import httpx

from vanna.capabilities.sql_runner import RunSqlToolArgs

from . import agent_state as state
from .schema_analyzer import build_dynamic_prompt_sections
from .sql_pipeline import (
    _run_llm_generated_sql,
    _get_corrected_sql,
    _extract_sql_error_detail,
    _validate_and_fix_sql,
    PERSON_NAME_RE,
)

logger = logging.getLogger(__name__)


# ── FORMATEO DE TEXTO/NÚMEROS ────────────────────────────────────────────────

# Detecta números en formato europeo: 4.096.554,84 o 1.234.567
_EU_NUMBER_RE = re.compile(r'\b(\d{1,3}(?:\.\d{3})+)(?:,(\d+))?\b')


def _fix_eu_numbers(text: str) -> str:
    """Convierte números europeos (1.234.567,89) a formato US (1,234,567.89)."""
    def _to_us(m: re.Match) -> str:
        integer = m.group(1).replace('.', ',')
        decimals = m.group(2)
        return f"{integer}.{decimals}" if decimals else integer
    text = _EU_NUMBER_RE.sub(_to_us, text)
    # Decimal con coma en porcentajes: 46,69% → 46.69%
    text = re.sub(r'(-?\d+),(\d{1,4})%', r'\1.\2%', text)
    # Decimal con coma en números negativos o sueltos: -99,91 → -99.91
    text = re.sub(r'(-\d+),(\d{2})\b', r'\1.\2', text)
    return text


def _clean_markdown(text: str) -> str:
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    return text


def _safe_str(v: Any) -> str:
    try:
        return "" if v is None else str(v)
    except Exception:
        return ""


_SKIP_FORMAT_KEYWORDS = {
    "year", "month", "day", "quarter", "key", "id",
    "number", "code", "type", "flag", "index", "rank",
    "level", "version", "semester",
}

_PERCENT_KEYWORDS = {
    "pct", "percent", "percentage", "porcentaje", "porc",
    "ratio", "tasa", "share", "participacion", "participación",
    "margen", "margin", "rate", "variacion", "variación",
}


def _col_words(col: str) -> set[str]:
    """Separa un nombre de columna en palabras (PascalCase/camelCase → palabras
    sueltas), para comparar por palabra completa y no por substring — así
    'AverageMonthlySales' no coincide con la keyword 'month' solo por contenerla
    como substring dentro de 'Monthly'."""
    return set(re.sub(r'([A-Z])', r' \1', col).lower().split())


def _is_percent_col(col: str) -> bool:
    return bool(_col_words(col) & _PERCENT_KEYWORDS)


def _format_cell(v: Any, col: str = "") -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, str):
        try:
            v = Decimal(v)
        except Exception:
            return v
    skip = bool(_col_words(col) & _SKIP_FORMAT_KEYWORDS)
    is_pct = _is_percent_col(col)
    if isinstance(v, (float, Decimal)):
        fv = float(v)
        if is_pct:
            return f"{fv:,.2f}%"
        return f"{fv:.2f}" if skip else f"{fv:,.2f}"
    if isinstance(v, int):
        if is_pct:
            return f"{v:,}%"
        return str(v) if skip else f"{v:,}"
    return str(v)


# ── GRÁFICOS (ECharts) ──────────────────────────────────────────────────────


_CHART_KEYWORDS = {
    "gráfico", "grafico", "gráfica", "grafica", "chart", "plot",
    "visualiza", "visualización", "visualizacion", "diagrama",
    "barras", "barra", "pastel", "torta", "dona", "donut",
    "linea", "línea", "dispersión", "dispersion", "scatter",
}

_PIE_KEYWORDS     = {"pastel", "torta", "dona", "donut", "pie"}
_LINE_KEYWORDS    = {"linea", "línea", "tendencia", "evolución", "evolucion", "histórico", "historico"}
_BAR_KEYWORDS     = {"barra", "barras", "columnas", "bar chart"}
_SCATTER_KEYWORDS = {"dispersión", "dispersion", "scatter"}
_TEMPORAL_COLS = {"year", "month", "quarter", "date", "año", "mes", "trimestre"}

_EC_COLORS = [
    "#4A90D9", "#E74C3C", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
    "#27AE60", "#8E44AD",
]
_EC_GRAD_END = [
    "#1a5276", "#922b21", "#1e8449", "#9a7d0a",
    "#6c3483", "#0e6655", "#935116", "#1f618d",
    "#196f3d", "#5b2c6f",
]
_EC_AREA_RGBA = [
    "rgba(74,144,217,0.22)",  "rgba(231,76,60,0.22)",
    "rgba(46,204,113,0.22)",  "rgba(243,156,18,0.22)",
    "rgba(155,89,182,0.22)",  "rgba(26,188,156,0.22)",
    "rgba(230,126,34,0.22)",  "rgba(52,152,219,0.22)",
    "rgba(39,174,96,0.22)",   "rgba(142,68,173,0.22)",
]


def _is_chart_question(question: str) -> bool:
    return any(kw in question.lower() for kw in _CHART_KEYWORDS)


def _detect_chart_type(question: str, cols: list) -> str:
    """Detecta el tipo de gráfico. Las palabras EXPLÍCITAS del usuario (pastel,
    línea, barras, dispersión) siempre tienen prioridad sobre la heurística de
    columnas temporales — si el usuario pidió barras, se respeta aunque haya
    una columna de año/mes en el resultado."""
    q = question.lower()
    if any(kw in q for kw in _PIE_KEYWORDS):
        return "pie"
    if any(kw in q for kw in _SCATTER_KEYWORDS):
        return "scatter"
    if any(kw in q for kw in _LINE_KEYWORDS):
        return "line"
    if any(kw in q for kw in _BAR_KEYWORDS):
        return "bar"
    if any(kw in " ".join(cols).lower() for kw in _TEMPORAL_COLS):
        return "line"
    return "bar"


def _is_numeric_col(col: str, rows: list) -> bool:
    has_value = False
    for r in rows[:10]:
        v = r.get(col)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float, Decimal)):
            return False
        has_value = True
    return has_value


def _raw_numeric(v: Any) -> Optional[float]:
    """Convierte a float o retorna None si el valor no es numérico."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            return None
    return None


def _build_chart_payload(question: str, cols: list, rows: list) -> dict:
    if not cols or len(rows) < 1:
        return {}

    chart_type = _detect_chart_type(question, cols)

    option: dict = {
        "backgroundColor": "transparent",
        "color": _EC_COLORS,
        "animation": True,
        "animationDuration": 900,
        "animationEasing": "cubicOut",
        "grid": {"left": "3%", "right": "4%", "bottom": "18%", "top": "8%", "containLabel": True},
    }

    if chart_type == "scatter":
        numeric_cols = [c for c in cols if _is_numeric_col(c, rows)]
        if len(numeric_cols) < 2:
            return {}
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        points = [
            [_raw_numeric(r.get(x_col)), _raw_numeric(r.get(y_col))]
            for r in rows
        ]
        points = [p for p in points if p[0] is not None and p[1] is not None]
        if not points:
            return {}
        option.update({
            "tooltip": {"trigger": "item"},
            "xAxis": {
                "type": "value", "name": x_col, "nameLocation": "middle", "nameGap": 28,
                "axisLabel": {"fontSize": 11, "color": "#333"},
            },
            "yAxis": {
                "type": "value", "name": y_col,
                "axisLabel": {"fontSize": 11, "color": "#333"},
            },
            "series": [{
                "type": "scatter",
                "data": points,
                "symbolSize": 9,
                "itemStyle": {"color": _EC_COLORS[0], "opacity": 0.75},
            }],
        })
        return option

    # Detección robusta de la columna-etiqueta: la primera columna NO numérica,
    # en vez de asumir siempre cols[0] — el SQL generado por LLM no garantiza el
    # orden de columnas, y asumirlo producía ejes horizontales con datos incorrectos
    # cuando la columna numérica venía antes que la de período/categoría.
    numeric_flags = {c: _is_numeric_col(c, rows) for c in cols}
    non_numeric_cols = [c for c in cols if not numeric_flags[c]]
    label_col  = non_numeric_cols[0] if non_numeric_cols else cols[0]
    value_cols = [c for c in cols if c != label_col and numeric_flags[c]]
    if not value_cols:
        return {}

    # Deduplicar por etiqueta — evita filas duplicadas por JOINs innecesarios a tablas de dimensión
    seen: set = set()
    deduped: list = []
    for r in rows:
        lbl = _safe_str(r.get(label_col, ""))
        if lbl not in seen:
            seen.add(lbl)
            deduped.append(r)
    rows = deduped

    # Defensivo: en líneas el orden cronológico es crítico para el eje horizontal.
    # Si la etiqueta es puramente numérica (año, número de mes/trimestre) y el SQL
    # no la ordenó, se reordena aquí — sin esto, un ORDER BY faltante en el SQL
    # generado deja el eje horizontal con el orden incorrecto.
    if chart_type == "line" and all(_raw_numeric(r.get(label_col)) is not None for r in rows):
        rows = sorted(rows, key=lambda r: _raw_numeric(r.get(label_col)))

    labels  = [_safe_str(r.get(label_col, "")) for r in rows]
    rotate  = 40 if len(labels) > 6 else 0

    if chart_type == "pie":
        values   = [_raw_numeric(r.get(value_cols[0])) for r in rows]
        pie_data = [{"name": l, "value": v} for l, v in zip(labels, values) if v is not None]
        if not pie_data:
            return {}
        option.update({
            "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c} ({d}%)"},
            "legend": {
                "type": "scroll", "orient": "vertical",
                "right": "2%", "top": "middle",
                "textStyle": {"fontSize": 12, "color": "#333"},
            },
            "series": [{
                "type": "pie",
                "radius": ["42%", "70%"],
                "center": ["42%", "52%"],
                "data": pie_data,
                "itemStyle": {"borderRadius": 7, "borderColor": "#fff", "borderWidth": 2},
                "label": {"show": True, "formatter": "{b}\n{d}%", "fontSize": 11, "color": "#333"},
                "labelLine": {"length": 10, "length2": 14},
                "emphasis": {
                    "itemStyle": {"shadowBlur": 14, "shadowColor": "rgba(0,0,0,0.25)"},
                    "scaleSize": 8,
                },
            }],
        })

    elif chart_type == "line":
        option.update({
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "xAxis": {
                "type": "category", "data": labels, "boundaryGap": False,
                "axisLabel": {"rotate": rotate, "fontSize": 11, "color": "#333"},
                "axisLine": {"lineStyle": {"color": "#ccc"}},
            },
            "yAxis": {"type": "value", "axisLabel": {"fontSize": 11, "color": "#333"}},
            "series": [],
        })
        if len(value_cols) > 1:
            option["legend"] = {"data": value_cols, "bottom": 0, "textStyle": {"color": "#333"}}
        for i, vcol in enumerate(value_cols):
            c     = _EC_COLORS[i % len(_EC_COLORS)]
            area  = _EC_AREA_RGBA[i % len(_EC_AREA_RGBA)]
            clear = re.sub(r',\s*[\d.]+\)', ", 0)", area)
            option["series"].append({
                "name": vcol, "type": "line",
                "data": [_raw_numeric(r.get(vcol)) for r in rows],
                "smooth": True,
                "symbol": "circle", "symbolSize": 7,
                "lineStyle": {"width": 3, "color": c},
                "itemStyle": {"color": c},
                "areaStyle": {
                    "color": {
                        "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": area},
                            {"offset": 1, "color": clear},
                        ],
                    }
                },
            })

    else:  # bar
        option.update({
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "xAxis": {
                "type": "category", "data": labels,
                "axisLabel": {"rotate": rotate, "fontSize": 11, "color": "#333"},
                "axisLine": {"lineStyle": {"color": "#ccc"}},
            },
            "yAxis": {"type": "value", "axisLabel": {"fontSize": 11, "color": "#333"}},
            "series": [],
        })
        if len(value_cols) > 1:
            option["legend"] = {"data": value_cols, "bottom": 0, "textStyle": {"color": "#333"}}
        for i, vcol in enumerate(value_cols):
            cs = _EC_COLORS[i % len(_EC_COLORS)]
            ce = _EC_GRAD_END[i % len(_EC_GRAD_END)]
            option["series"].append({
                "name": vcol, "type": "bar",
                "data": [_raw_numeric(r.get(vcol)) for r in rows],
                "barMaxWidth": 52,
                "itemStyle": {
                    "borderRadius": [5, 5, 0, 0],
                    "color": {
                        "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                        "colorStops": [
                            {"offset": 0, "color": cs},
                            {"offset": 1, "color": ce},
                        ],
                    },
                },
                "emphasis": {"itemStyle": {"opacity": 0.82}},
            })

    return option


# ── FIN GRÁFICOS ─────────────────────────────────────────────────────────────


# ── FILTROS DEL STREAMING DEL AGENTE ─────────────────────────────────────────

_VAGUE_PLACEHOLDERS = [" de x", " a y", " del z%", " de x,", "fueron de x", "aumentaron a y",
                       "ventas de x", "total de x", " x millones", " y millones"]

# Detecta alias de columnas SQL usados como placeholders en el texto (ej: "TotalSalesAmount")
_SQL_ALIAS_AS_PLACEHOLDER = re.compile(
    r'\b[A-Z][a-zA-Z]*(Amount|Sales|Total|Count|Revenue|Cost|Price|Quantity|Value|Sum|Avg|Average)\b'
)


def _is_vague_analysis(text: str) -> bool:
    lower = text.lower()
    if any(p in lower for p in _VAGUE_PLACEHOLDERS):
        return True
    return bool(_SQL_ALIAS_AS_PLACEHOLDER.search(text))


def _should_hide_text(text: str) -> bool:
    lower = text.lower()
    indicators = [
        "tool completed successfully", "running tool", "executing tool",
        "run_sql", "similarity:", "**arguments:", "**timestamp:",
        "guardados en un archivo csv", "**id:",
        "query executed successfully. no rows returned",
        "**retrieved memories", "tool failed:", "error executing query:",
        "pyodbc", "sqlexecdirectw", "background on this error",
    ]
    return any(i in lower for i in indicators)


def _is_row_enumeration(text: str) -> bool:
    return bool(re.match(r"^\d+\.\s", text.strip()))


def _is_markdown_table_text(text: str) -> bool:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False
    pipe_lines = sum(1 for l in lines if l.startswith("|") and l.endswith("|"))
    return pipe_lines >= 2


def extract_text_from_component(component: Any) -> str:
    if component is None:
        return ""

    if isinstance(component, str):
        text = component.strip()
        if _should_hide_text(text) or _is_row_enumeration(text) or _is_markdown_table_text(text):
            return ""
        return text

    rc = getattr(component, "rich_component", None)
    if rc is not None:
        if hasattr(rc, "tool_name"):
            return ""

        if hasattr(rc, "rows") and hasattr(rc, "columns"):
            rows = getattr(rc, "rows", [])
            cols = getattr(rc, "columns", [])
            if rows and cols:
                max_rows = min(len(rows), state.MAX_ROWS_LIMIT)
                html = ['<table class="data-table"><thead><tr>']
                for col in cols:
                    html.append(f'<th>{_safe_str(col)}</th>')
                html.append('</tr></thead><tbody>')
                for r in rows[:max_rows]:
                    html.append('<tr>')
                    for c in cols:
                        v = r.get(c, "") if isinstance(r, dict) else ""
                        html.append(f'<td>{_format_cell(v, c)}</td>')
                    html.append('</tr>')
                html.append('</tbody></table>')
                return "\n".join(html)

        content = getattr(rc, "content", None)
        if isinstance(content, str) and content.strip():
            text = content.strip()
            if _should_hide_text(text) or _is_row_enumeration(text) or _is_markdown_table_text(text):
                return ""
            return _fix_eu_numbers(_clean_markdown(text))

        return ""

    sc = getattr(component, "simple_component", None)
    if sc is not None:
        text = getattr(sc, "text", None)
        if isinstance(text, str):
            text = text.strip()
            if _should_hide_text(text) or _is_row_enumeration(text) or _is_markdown_table_text(text):
                return ""
            return _fix_eu_numbers(_clean_markdown(text))

    return ""


def _merge_multiple_tables(html_chunks: list[str]) -> str:
    all_html = "\n".join(html_chunks)
    theads = re.findall(r'<thead>.*?</thead>', all_html, re.DOTALL | re.IGNORECASE)
    tbodies = re.findall(r'<tbody>(.*?)</tbody>', all_html, re.DOTALL | re.IGNORECASE)

    if not theads or not tbodies:
        return all_html

    all_rows = []
    for tbody in tbodies:
        all_rows.extend(re.findall(r'<tr>.*?</tr>', tbody, re.DOTALL | re.IGNORECASE))

    if not all_rows:
        return all_html

    merged = f'<table class="data-table">{theads[0]}<tbody>'
    merged += "\n".join(all_rows)
    merged += '</tbody></table>'
    logger.info("Tablas fusionadas: %d tbody → %d filas", len(tbodies), len(all_rows))
    return merged


# ── NARRATIVA DE ANÁLISIS ─────────────────────────────────────────────────────

async def _generate_analysis(question: str, data_rows: list, columns: list) -> str:
    """Llama al LLM para generar un párrafo de análisis basado en los datos frescos."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return ""

    # Si TODOS los valores son NULL/vacíos (ej. un filtro que no matcheó ninguna
    # fila real, como comparar una ciudad contra una columna de país), no hay
    # nada que narrar — no llamar al LLM en absoluto. Sin este guard, el LLM
    # recibe un prompt sin datos reales y puede "rellenar" con el número de
    # ejemplo usado en las reglas de formato más abajo, inventando una cifra
    # que nunca estuvo en los datos.
    if not data_rows or all(
        r.get(c) is None or r.get(c) == "" for r in data_rows for c in columns
    ):
        return "No encontré datos para esa consulta. Intenta con un período o filtro diferente."

    # Construir resumen de datos para el LLM con números formateados
    rows_preview = data_rows[:10]
    if len(columns) == 1 and len(rows_preview) == 1:
        col = columns[0]
        val = _format_cell(rows_preview[0].get(col), col)
        data_str = f"{col}: {val}"
    else:
        data_str = " | ".join(columns) + "\n"
        for r in rows_preview:
            data_str += " | ".join(_format_cell(r.get(c), c) for c in columns) + "\n"

    # Si hay varias filas, calcular en Python cuál tiene el valor más alto/bajo
    # de la columna numérica relevante — el LLM (sobre todo gpt-4o-mini) compara
    # mal varios números de varias filas y puede señalar como "el mayor" uno que
    # no lo es. Esto pasó en vivo: dijo "máximo en 2012" cuando 2013 era mayor.
    # Pre-calcular el hecho evita que el LLM tenga que "razonar" la comparación.
    highlight = ""
    if len(rows_preview) > 1:
        numeric_cols = [
            c for c in columns
            if all(
                isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)
                for r in rows_preview
            )
        ]
        if numeric_cols:
            metric_col = numeric_cols[-1]
            label_cols = [c for c in columns if c != metric_col] or columns
            best_row = max(rows_preview, key=lambda r: r.get(metric_col))
            worst_row = min(rows_preview, key=lambda r: r.get(metric_col))
            best_label = ", ".join(f"{c}={_format_cell(best_row.get(c), c)}" for c in label_cols)
            worst_label = ", ".join(f"{c}={_format_cell(worst_row.get(c), c)}" for c in label_cols)
            highlight = (
                f"\nDATO DESTACADO (ya calculado en Python, NO lo recalcules ni lo cuestiones): "
                f"el valor MÁS ALTO de '{metric_col}' es {_format_cell(best_row.get(metric_col), metric_col)} "
                f"({best_label}). El MÁS BAJO es {_format_cell(worst_row.get(metric_col), metric_col)} "
                f"({worst_label}).\n"
            )

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"Datos obtenidos:\n{data_str}"
        f"{highlight}\n"
        "Escribe UN párrafo en español (2-3 oraciones) respondiendo la pregunta con los datos de arriba.\n"
        "REGLAS ESTRICTAS:\n"
        "- Nombra EXACTAMENTE los valores que aparecen en los datos (nombres de territorios, productos, "
        "clientes, años, porcentajes, montos, etc.). NUNCA un valor que no esté en 'Datos obtenidos' arriba.\n"
        "- NUNCA uses frases vagas como 'el especificado en la consulta', 'los datos muestran', "
        "'según los resultados', 'el territorio analizado'. Si tienes el nombre, úsalo.\n"
        "- Si hay un número ganador (mejor, mayor, top), usa el DATO DESTACADO ya calculado arriba — "
        "NUNCA determines tú mismo comparando filas, tiendes a equivocarte con varias filas.\n"
        "- NÚMEROS: usa SIEMPRE coma como separador de miles y punto como decimal, "
        "ej. 1,234,567.89 (patrón de formato, no un valor real — usa los números reales de los datos).\n"
        "- Sin markdown, bullets ni encabezados. Solo texto narrativo directo."
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 200,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            data = response.json()
            return _fix_eu_numbers(data["choices"][0]["message"]["content"].strip())
    except Exception:
        logger.exception("Error generando análisis fresco")
        return ""


async def _stream_no_results(question: str) -> AsyncGenerator[str, None]:
    """Genera un mensaje contextual cuando la consulta no devuelve filas."""
    api_key = os.getenv("OPENAI_API_KEY")
    model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        yield "No encontré datos para esa consulta. Intenta con un período o filtro diferente."
        return

    prompt = (
        f"El usuario preguntó: \"{question}\"\n\n"
        "La consulta SQL ejecutada no devolvió ningún resultado. "
        "Escribe 2 oraciones en español explicando que no se encontraron datos y sugiriendo "
        "qué podría cambiar el usuario para obtener resultados (diferente año, región, producto, etc.). "
        "Sé específico con los filtros que mencionó el usuario. Sin markdown."
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 120,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        pass
    except Exception:
        yield "No encontré datos para esa consulta. Intenta con un período o filtro diferente."


# ── FORECASTING ───────────────────────────────────────────────────────────────

# Años futuros respecto al dataset (AdventureWorks termina en 2014)
_FUTURE_YEAR_RE = re.compile(r'\b(201[5-9]|20[2-9]\d)\b')
_ANY_YEAR_RE = re.compile(r'\b(20\d{2}|19\d{2})\b')


def _is_historical_sales_question(question: str) -> bool:
    """True si la pregunta menciona un año que ya existe en el dataset (no futuro)."""
    return bool(_ANY_YEAR_RE.search(question) and not _FUTURE_YEAR_RE.search(question))


# ── ANÁLISIS ESTADÍSTICO (significancia, correlación, outliers) ─────────────

# SIGNIFICANCE/CORRELATION exigen datos A NIVEL DE FILA (una observación por fila) —
# si el corrector, al arreglar un error de ejecución, "resuelve" el problema agregando
# (GROUP BY/SUM, o incluso DISTINCT/window functions que colapsan a 1 fila por grupo),
# la consulta ejecuta sin error pero el resultado no sirve para el test estadístico
# (ej. un t-test necesita >=2 observaciones por grupo, no 1 total ya agregado). Se
# detecta por la FORMA de los datos ya devueltos — no por el texto del SQL — porque
# no importa qué técnica de agregación haya usado el LLM para llegar ahí.
def _stats_granularity_problem(df, stats_type: str) -> Optional[str]:
    cols = df.columns.tolist()
    if len(cols) < 2:
        return None
    if stats_type == "SIGNIFICANCE":
        counts = df[cols[0]].value_counts()
        if (counts >= 2).sum() < 2:
            return (
                "La consulta devolvió los datos ya agregados (como máximo 1 fila por grupo) "
                "en vez de una fila por observación individual. Necesito AL MENOS 2 filas por "
                "grupo (ej. una fila por orden/venta individual), sin agregar con SUM/COUNT/AVG "
                "ni con DISTINCT/window functions que colapsen los datos a un total por grupo."
            )
    elif stats_type == "CORRELATION" and len(df) < 3:
        return (
            "La consulta devolvió muy pocas filas para calcular correlación. Necesito muchas "
            "filas, una por observación individual (ej. una por orden/línea de venta), sin "
            "agregar ni agrupar."
        )
    return None


_STATS_SQL_INSTRUCTIONS = {
    "CORRELATION": (
        "Escribe una consulta T-SQL que devuelva DOS columnas numéricas a nivel de fila "
        "(SIN agregar/agrupar) para las dos variables mencionadas en la pregunta. "
        "Si necesitas limitar las filas (tabla muy grande), usa "
        "'ORDER BY NEWID() OFFSET 0 ROWS FETCH NEXT 500 ROWS ONLY' para una muestra aleatoria. "
        "NUNCA ordenes por las columnas que vas a correlacionar (ni columnas relacionadas, ej. "
        "ListPrice/UnitPrice) antes de limitar — produce una muestra sin varianza y arruina el "
        "cálculo de correlación. Cada fila debe representar una observación individual "
        "(ej. una orden, una línea de venta), no un total agregado."
    ),
    "SIGNIFICANCE": (
        "Escribe una consulta T-SQL que devuelva DOS columnas: una columna de texto que "
        "identifique a qué grupo pertenece cada fila (los grupos mencionados en la pregunta) "
        "y una columna numérica con la métrica a comparar, A NIVEL DE FILA (SIN agregar). "
        "Si necesitas limitar las filas, usa ROW_NUMBER() OVER (PARTITION BY <columna de grupo> "
        "ORDER BY NEWID()) AS rn en un CTE, y filtra 'WHERE rn <= 500' desde una consulta "
        "EXTERNA que seleccione de ese CTE — NUNCA filtres por rn en el mismo SELECT donde lo "
        "calculas (SQL Server no lo permite). "
        "NUNCA uses ORDER BY (SELECT NULL) ni omitas el ORDER BY del ROW_NUMBER — eso NO es "
        "aleatorio, respeta el orden físico/del índice de la tabla (normalmente orden de "
        "inserción o fecha) y produce una muestra sesgada que puede mostrar una diferencia "
        "invertida a la real. ORDER BY NEWID() es obligatorio para que la muestra sea aleatoria. "
        "NUNCA uses un TOP o OFFSET/FETCH global — sesgaría la muestra hacia el grupo con más "
        "filas en el orden físico de la tabla y dejaría sin datos a los demás grupos."
    ),
    "OUTLIERS": (
        "Escribe una consulta T-SQL que devuelva UNA fila por entidad, usando ÚNICAMENTE el "
        "tipo de entidad EXPLÍCITAMENTE mencionado en la pregunta (ej. si dice 'productos', usa "
        "solo productos — NUNCA combines varios tipos de entidad distintos con UNION en la misma "
        "consulta). La fila debe traer la métrica numérica agregada de esa entidad. "
        "Incluye TODAS las entidades de ese tipo, sin TOP ni WHERE/HAVING que filtre por "
        "la métrica en sí (ej. NUNCA 'HAVING SUM(...) > promedio') — la detección de outliers la "
        "hace Python después con el conjunto completo de datos, no SQL. "
        "Usa como VALOR el nombre descriptivo real de la entidad que elijas (nunca su columna "
        "*Key numérica) para que el resultado sea legible. Independientemente de qué entidad "
        "sea (producto, cliente, territorio...), nombra (alias) SIEMPRE la primera columna "
        "exactamente 'Entidad' — no copies nombres de columna de ejemplo de estas instrucciones, "
        "son solo ilustrativos. La segunda columna (la métrica) sí debe tener un nombre "
        "descriptivo real (ej. TotalSalesAmount)."
    ),
}


async def _classify_stats_type(question: str) -> str:
    """Retorna 'CORRELATION', 'SIGNIFICANCE' u 'OUTLIERS' según el tipo de análisis pedido."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "SIGNIFICANCE"

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 10,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Clasifica la pregunta estadística en UNA categoría:\n"
                                "CORRELATION — pregunta si dos variables numéricas están relacionadas "
                                "o correlacionadas (ej. precio y cantidad, descuento y ventas).\n"
                                "SIGNIFICANCE — pregunta si hay una diferencia significativa entre "
                                "dos o más grupos/categorías (ej. territorios, productos, períodos).\n"
                                "OUTLIERS — pregunta por valores atípicos, anómalos o fuera de lo "
                                "normal en una métrica.\n"
                                "Responde ÚNICAMENTE con CORRELATION, SIGNIFICANCE u OUTLIERS."
                            ),
                        },
                        {"role": "user", "content": question},
                    ],
                },
            )
            raw = resp.json()["choices"][0]["message"]["content"].strip().upper()
            for opt in ("CORRELATION", "SIGNIFICANCE", "OUTLIERS"):
                if opt in raw:
                    return opt
            return "SIGNIFICANCE"
    except Exception:
        logger.exception("Error clasificando tipo de análisis estadístico")
        return "SIGNIFICANCE"


async def _get_stats_sql(question: str, stats_type: str) -> Optional[str]:
    """Genera, vía LLM + contexto RAG del esquema, el SQL necesario para el análisis estadístico."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None

    schema_context = ""
    if state.SCHEMA_STORE:
        try:
            hits = state.SCHEMA_STORE.query(question, k=8)
            schema_context = "\n".join(h.get("doc", "") for h in hits if h.get("doc"))
        except Exception:
            pass

    schema_meta = state.SCHEMA_META
    secs = build_dynamic_prompt_sections(schema_meta) if schema_meta else {}
    sales_source = secs.get("sales_source", "")

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"Esquema disponible:\n{schema_context}\n\n"
        f"{sales_source}\n\n"
        f"{_STATS_SQL_INSTRUCTIONS[stats_type]}\n\n"
        "REGLAS:\n"
        "- Esta es una base SQL Server de SOLO LECTURA — únicamente SELECT.\n"
        "- Los valores de texto en la base de datos (países, categorías, nombres) están en "
        "INGLÉS aunque la pregunta esté en español — usa el equivalente en inglés en los "
        "filtros WHERE (ej. 'France' no 'Francia', 'Spain' no 'España', 'Germany' no 'Alemania').\n"
        "- NUNCA uses AS, ON, IN, BY como alias de tabla.\n"
        "- ORDER BY solo en el SELECT final, nunca dentro de un CTE sin TOP/FETCH.\n"
        "- No agregues explicaciones ni markdown — responde ÚNICAMENTE con el SQL.\n"
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 500,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            raw = response.json()["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            return raw.strip() or None
    except Exception:
        logger.exception("Error generando SQL para análisis estadístico")
        return None


def _interpret_correlation_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "fuerte"
    if abs_r >= 0.4:
        return "moderada"
    if abs_r >= 0.2:
        return "débil"
    return "muy débil o nula"


async def _narrate_stats(question: str, stats_summary: str) -> str:
    """Redacta la narrativa final a partir de números YA calculados — el LLM solo redacta,
    nunca recalcula, para evitar que invente cifras."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return stats_summary

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"Resultado estadístico ya calculado:\n{stats_summary}\n\n"
        "Escribe UN párrafo en español (2-4 oraciones) explicando este resultado en lenguaje "
        "claro para alguien sin formación estadística, usando los valores concretos ya "
        "calculados arriba. Si el resultado incluye un p-value, explica qué significa en este "
        "contexto; si NO incluye un p-value (ej. detección de outliers por rango intercuartílico), "
        "NO lo menciones ni te disculpes por su ausencia — simplemente explica el resultado tal "
        "como está. NO inventes ni recalcules números — usa EXACTAMENTE los que se te dieron. "
        "NÚMEROS: coma para miles, punto para decimal. Sin markdown."
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 220,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            return _fix_eu_numbers(resp.json()["choices"][0]["message"]["content"].strip())
    except Exception:
        logger.exception("Error narrando resultado estadístico")
        return stats_summary


async def _run_stats(question: str) -> tuple[str, str]:
    """Ejecuta análisis estadístico (correlación, significancia u outliers).
    Retorna (narrativa, tabla_html)."""
    if not state.SQL_RUNNER:
        return "No hay conexión a la base de datos disponible.", ""

    stats_type = await _classify_stats_type(question)
    logger.info("Tipo de análisis estadístico: %s", stats_type)

    sql = await _get_stats_sql(question, stats_type)
    if not sql:
        return "No pude generar la consulta necesaria para este análisis. Intenta reformular la pregunta.", ""

    # Hasta 3 intentos (1 original + 2 correcciones) — un solo intento de
    # corrección no siempre alcanza: el LLM a veces introduce un error nuevo
    # (ej. alias inconsistente, o agregar cuando se necesita nivel de fila)
    # al arreglar el anterior.
    df = None
    max_attempts = 3
    for attempt in range(max_attempts):
        is_last = attempt == max_attempts - 1
        try:
            sql = await _validate_and_fix_sql(sql, question, extra_instructions=_STATS_SQL_INSTRUCTIONS[stats_type])
            df = await _run_llm_generated_sql(sql)
        except ValueError as e:
            return f"No puedo ejecutar este análisis: {e}", ""
        except RuntimeError as e:
            return f"No puedo ejecutar este análisis: {e}", ""
        except Exception as e:
            if is_last:
                logger.exception("SQL de análisis estadístico siguió fallando tras agotar los reintentos")
                return "No pude obtener los datos para este análisis. Intenta reformular la pregunta.", ""
            logger.warning(
                "SQL de análisis estadístico falló (intento %d/%d) — solicitando corrección al LLM: %s",
                attempt + 1, max_attempts, e,
            )
            # Se reenvía la regla de granularidad del tipo de análisis (ej. "sin agregar,
            # a nivel de fila") — si no, el corrector tiende a "arreglar" el error agregando
            # con SUM/GROUP BY, lo que rompe el cálculo estadístico posterior en Python.
            problem = (
                f"Error SQL de SQL Server: {_extract_sql_error_detail(str(e))}. "
                f"Al corregir, respeta esta regla: {_STATS_SQL_INSTRUCTIONS[stats_type]}"
            )
            corrected = await _get_corrected_sql(question, sql, problem)
            if not corrected or corrected == sql:
                return "No pude obtener los datos para este análisis. Intenta reformular la pregunta.", ""
            sql = corrected
            continue

        # La consulta ejecutó sin error, pero puede haber vuelto agregada
        # (1 fila por grupo) por algún camino de corrección previo — eso
        # ejecuta bien pero no sirve para el test estadístico.
        if df is not None and not df.empty:
            gran_problem = _stats_granularity_problem(df, stats_type)
            if gran_problem and not is_last:
                logger.warning(
                    "Datos insuficientes a nivel de fila (intento %d/%d): %s",
                    attempt + 1, max_attempts, gran_problem,
                )
                problem = f"{gran_problem} Regla del análisis: {_STATS_SQL_INSTRUCTIONS[stats_type]}"
                corrected = await _get_corrected_sql(question, sql, problem)
                if not corrected or corrected == sql:
                    break
                sql = corrected
                continue
        break

    if df is None or df.empty or len(df.columns) < 2:
        return "No hay suficientes datos para realizar este análisis.", ""

    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats

    cols = df.columns.tolist()

    try:
        if stats_type == "CORRELATION":
            x = pd.to_numeric(df[cols[0]], errors="coerce")
            y = pd.to_numeric(df[cols[1]], errors="coerce")
            paired = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(paired) < 3:
                return "No hay suficientes datos numéricos pareados para calcular la correlación.", ""

            r, p = scipy_stats.pearsonr(paired["x"], paired["y"])
            strength = _interpret_correlation_strength(r)
            summary = (
                f"Variables: {cols[0]} vs {cols[1]} (n={len(paired)})\n"
                f"Coeficiente de correlación de Pearson r = {r:.3f}\n"
                f"p-value = {p:.4f}\n"
                f"Fuerza de la relación: {strength}\n"
                f"Significativa al 95%: {'sí' if p < 0.05 else 'no'}"
            )
            narrative = await _narrate_stats(question, summary)
            return narrative, ""

        if stats_type == "SIGNIFICANCE":
            group_col, value_col = cols[0], cols[1]
            work = df[[group_col, value_col]].copy()
            work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
            work = work.dropna()
            groups = {
                name: g[value_col].values
                for name, g in work.groupby(group_col)
                if len(g) >= 2
            }
            if len(groups) < 2:
                return "No hay al menos dos grupos con suficientes datos para comparar.", ""

            if len(groups) == 2:
                (name_a, vals_a), (name_b, vals_b) = list(groups.items())
                statistic, p = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)
                test_name = "t-test (Welch)"
            else:
                statistic, p = scipy_stats.f_oneway(*groups.values())
                test_name = "ANOVA de un factor"

            rows_html = [
                '<table class="data-table"><thead><tr>'
                '<th>Grupo</th><th>n</th><th>Media</th><th>Desv. estándar</th>'
                '</tr></thead><tbody>'
            ]
            summary_lines = [f"Test aplicado: {test_name}", f"Estadístico = {statistic:.3f}", f"p-value = {p:.4f}"]
            for name, vals in groups.items():
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                rows_html.append(
                    f"<tr><td>{_safe_str(name)}</td><td>{len(vals)}</td>"
                    f"<td>{_format_cell(mean_v)}</td><td>{_format_cell(std_v)}</td></tr>"
                )
                summary_lines.append(f"Grupo '{name}': n={len(vals)}, media={mean_v:,.2f}")
            rows_html.append("</tbody></table>")
            summary_lines.append(f"Significativo al 95%: {'sí' if p < 0.05 else 'no'}")

            narrative = await _narrate_stats(question, "\n".join(summary_lines))
            return narrative, "\n".join(rows_html)

        # OUTLIERS
        entity_col, value_col = cols[0], cols[1]
        work = df[[entity_col, value_col]].copy()
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna()
        if len(work) < 4:
            return "No hay suficientes datos para detectar valores atípicos.", ""

        q1 = work[value_col].quantile(0.25)
        q3 = work[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = work[(work[value_col] < lower) | (work[value_col] > upper)].sort_values(
            value_col, ascending=False
        )

        summary = (
            f"Métrica analizada: {value_col} (n={len(work)})\n"
            f"Q1={q1:,.2f}, Q3={q3:,.2f}, IQR={iqr:,.2f}\n"
            f"Rango normal esperado: [{lower:,.2f}, {upper:,.2f}]\n"
            f"Valores atípicos encontrados: {len(outliers)}"
        )

        table_html = ""
        if not outliers.empty:
            rows_html = [
                f'<table class="data-table"><thead><tr>'
                f'<th>{_safe_str(entity_col)}</th><th>{_safe_str(value_col)}</th>'
                f'</tr></thead><tbody>'
            ]
            for _, r in outliers.head(state.MAX_ROWS_LIMIT).iterrows():
                rows_html.append(
                    f"<tr><td>{_safe_str(r[entity_col])}</td>"
                    f"<td>{_format_cell(r[value_col], value_col)}</td></tr>"
                )
            rows_html.append("</tbody></table>")
            table_html = "\n".join(rows_html)

        narrative = await _narrate_stats(question, summary)
        return narrative, table_html

    except Exception:
        logger.exception("Error calculando análisis estadístico")
        return "Ocurrió un error calculando el análisis estadístico. Intenta de nuevo.", ""


# ── FIN ANÁLISIS ESTADÍSTICO ─────────────────────────────────────────────────


# ── DISCOVERY ─────────────────────────────────────────────────────────────────

DISCOVERY_KEYWORDS = [
    "qué tablas", "que tablas", "tablas disponibles", "vistas disponibles",
    "tablas y vistas", "lista de tablas", "muestra las tablas", "muéstrame las tablas",
    "cuáles tablas", "cuales tablas", "todas las tablas", "show tables",
    "esquema disponible", "qué esquema", "que esquema",
]


def _is_discovery_question(message: str) -> bool:
    return any(kw in message.lower() for kw in DISCOVERY_KEYWORDS)


_DISCOVERY_SQL = """
SELECT
    t.name                          AS Tabla,
    SUM(ps.row_count)               AS Registros
FROM sys.tables t
JOIN sys.dm_db_partition_stats ps
     ON t.object_id = ps.object_id AND ps.index_id IN (0, 1)
WHERE t.is_ms_shipped = 0
GROUP BY t.name
ORDER BY t.name
"""


async def _run_discovery(question: str = "") -> tuple[str, str]:
    """Consulta nombres de tablas y cantidad de registros reales de la BD.
    Aplica ordenamiento y límite según lo que el usuario pidió.
    Retorna (html_tabla, texto_resumen)."""
    if not state.SQL_RUNNER:
        return "", "No hay conexión a la base de datos disponible."
    try:
        df = await state.SQL_RUNNER.run_sql(RunSqlToolArgs(sql=_DISCOVERY_SQL), None)
        if df.empty:
            return "", "No se encontraron tablas en la base de datos."

        total_tables = len(df)
        total_records = int(df["Registros"].sum())

        # Determinar orden según la pregunta
        q = question.lower()
        sort_desc = any(w in q for w in [
            "más registros", "mayor", "más grande", "top", "mayor cantidad",
            "más datos", "más filas", "más grande"
        ])
        sort_asc = any(w in q for w in [
            "menos registros", "menor", "más pequeña", "menos datos", "menos filas"
        ])

        if sort_desc:
            df = df.sort_values("Registros", ascending=False).reset_index(drop=True)
        elif sort_asc:
            df = df.sort_values("Registros", ascending=True).reset_index(drop=True)

        # Extraer límite numérico de la pregunta (ej. "10", "5")
        import re as _re
        num_match = _re.search(r'\b(\d+)\b', q)
        limit = int(num_match.group(1)) if num_match else None

        # Si pide "la tabla" (singular) en lugar de "las tablas" (plural) y no
        # dio un número explícito, se refiere a una sola tabla (la extrema).
        if limit is None and (sort_asc or sort_desc) and "tabla" in q and "tablas" not in q:
            limit = 1

        if limit and 1 <= limit < total_tables:
            df = df.head(limit)

        cols = df.columns.tolist()
        rows = df.to_dict("records")

        html = ['<table class="data-table"><thead><tr>']
        for col in cols:
            html.append(f'<th>{_safe_str(col)}</th>')
        html.append('</tr></thead><tbody>')
        for r in rows:
            html.append('<tr>')
            for c in cols:
                html.append(f'<td>{_format_cell(r.get(c, ""), c)}</td>')
            html.append('</tr>')
        html.append('</tbody></table>')

        shown = len(rows)
        if shown < total_tables:
            summary = (
                f"Mostrando {shown} de {total_tables} tablas "
                f"(total de registros en la BD: {total_records:,})."
            )
        else:
            summary = (
                f"La base de datos contiene {total_tables} tablas "
                f"con un total de {total_records:,} registros."
            )
        logger.info("Discovery: mostrando %d/%d tablas", shown, total_tables)
        return "\n".join(html), summary
    except Exception:
        logger.exception("Error ejecutando discovery de tablas")
        return "", "No se pudo consultar la estructura de la base de datos."


# ── CLASIFICADOR DE INTENCIÓN ─────────────────────────────────────────────────
# Determina si el mensaje requiere SQL o es conversación general.
# Usa el LLM con max_tokens bajo para ser robusto ante cualquier formulación.

async def _classify_intent(message: str) -> str:
    """Retorna 'SQL', 'STATS', 'PREDICTION', 'DISCOVERY' o 'CHAT' según el tipo de mensaje."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "SQL"

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 20,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Clasifica el mensaje del usuario en UNA de estas cinco categorías:\n"
                                "DISCOVERY — el usuario quiere ver qué tablas existen en la base de datos, "
                                "cuántos registros tiene cada tabla, qué contiene la BD, qué datos hay disponibles, "
                                "la estructura o el contenido general de la base de datos.\n"
                                "STATS — el usuario pregunta por significancia estadística entre grupos "
                                "('¿es significativa la diferencia entre X y Y?', '¿son realmente distintos "
                                "A y B o es casualidad?'), correlación entre dos variables ('¿hay relación/"
                                "correlación entre X e Y?', '¿el precio influye en la cantidad vendida?'), "
                                "o valores atípicos/outliers/anómalos en una métrica ('¿qué productos tienen "
                                "ventas atípicas?'). Estas preguntas requieren un test estadístico con p-value, "
                                "no un simple cálculo SQL.\n"
                                "SQL — el usuario quiere datos históricos reales o análisis estadístico "
                                "sobre los datos: ventas pasadas, métricas, rankings, comparaciones, "
                                "productos, clientes, empleados, territorios, tendencias pasadas, "
                                "gráficos de datos existentes, promedios, totales, conteos, "
                                "análisis estadísticos (promedio móvil, mediana, varianza, "
                                "crecimiento, participación, ratio), o cualquier pregunta analítica "
                                "aunque no especifique el período o dimensión exacta.\n"
                                "PREDICTION — el usuario pregunta por ventas o métricas FUTURAS de UNA "
                                "entidad concreta nombrada (o de los totales): un año fuera del dataset "
                                "(2015 en adelante), 'el próximo año', 'el año que viene', 'a futuro', etc. "
                                "Aplica tanto a TOTALES GLOBALES ('¿cuánto se venderá en total en 2015?') "
                                "como a una dimensión específica NOMBRADA ('¿cuánto se venderá en Francia "
                                "en 2016?', '¿cómo estarán las ventas de Bikes el próximo año?'). "
                                "Si la pregunta pide RANKEAR o comparar varias entidades en el futuro sin "
                                "nombrar una en particular ('¿qué producto/región/empleado liderará en el "
                                "futuro?') → SQL, NO PREDICTION (no hay una sola entidad que proyectar). "
                                "Si la pregunta usa tiempo PASADO ('cuánto se vendió', 'cuánto fue', "
                                "'cuántas ventas hubo', 'cuánto generó') o menciona un año ya pasado "
                                "(2013, 2012, 2011...) → SQL, NUNCA PREDICTION.\n"
                                "CHAT — el usuario saluda, agradece, se despide, pregunta qué eres, "
                                "qué puedes hacer, cómo funcionas, o hace preguntas fuera del dominio de datos.\n"
                                "Responde ÚNICAMENTE con la palabra SQL, STATS, PREDICTION, DISCOVERY o CHAT."
                            ),
                        },
                        {"role": "user", "content": message},
                    ],
                },
            )
            raw = resp.json()["choices"][0]["message"]["content"].strip().upper()
            # Extraer primera palabra — el LLM a veces añade contexto extra
            first_word = raw.split()[0] if raw.split() else ""
            if first_word in ("DISCOVERY", "SQL", "STATS", "PREDICTION", "CHAT"):
                intent = first_word
            elif "DISCOVERY" in raw:
                intent = "DISCOVERY"
            elif "STATS" in raw:
                intent = "STATS"
            elif "PREDICTION" in raw:
                intent = "PREDICTION"
            elif "CHAT" in raw:
                intent = "CHAT"
            else:
                intent = "SQL"
            logger.info("Clasificación de intención: '%s' → %s", message[:60], intent)
            return intent
    except Exception:
        logger.exception("Error clasificando intención — fallback a SQL")
        return "SQL"


async def _get_chat_response(message: str) -> str:
    """Responde directamente con el LLM sin invocar el agente SQL."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "¡Hola! Soy TIARA, tu asistente de análisis de datos de ventas. ¿En qué puedo ayudarte?"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 200,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Eres TIARA, un asistente de análisis de datos. "
                                "Solo puedes hacer dos cosas:\n"
                                "1. Responder saludos, despedidas y agradecimientos de forma breve y amigable.\n"
                                "2. Explicar qué puedes hacer cuando el usuario pregunta por tus capacidades, "
                                "funciones o cómo funcionar. Ejemplos que activan esto: "
                                "'qué puedes hacer', 'dime qué puedes hacer', 'para qué sirves', "
                                "'qué tipos de preguntas respondes', 'eso es todo lo que puedes hacer', "
                                "'qué más puedes hacer', 'cómo funcionas'. "
                                "En ese caso explica brevemente que puedes responder preguntas sobre "
                                "ventas, productos, clientes, territorios, empleados y tendencias "
                                "basadas en la base de datos de la empresa.\n"
                                "Para CUALQUIER otra cosa (chistes, preguntas generales, temas externos, "
                                "opiniones, etc.), responde EXACTAMENTE: "
                                "'Solo puedo ayudarte a responder preguntas de la base de datos proporcionada.'\n"
                                "Sin markdown ni bullets. Solo texto natural en español."
                            ),
                        },
                        {"role": "user", "content": message},
                    ],
                },
            )
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Error generando respuesta conversacional")
        return "Solo puedo ayudarte a responder preguntas de la base de datos proporcionada."


# ── RAG / CONSTRUCCIÓN DEL PROMPT DE ESQUEMA ─────────────────────────────────

def _filter_and_deduplicate(hits: list) -> list:
    sorted_hits = sorted(hits, key=lambda h: h.get("distance", 999))

    # Limitar join_path/join_chain a máximo 3 para que los docs de esquema también aparezcan
    result: list = []
    join_doc_count = 0
    MAX_JOIN_DOCS = 3
    for h in sorted_hits:
        if len(result) >= state.RAG_K_FINAL:
            break
        meta_type = h.get("meta", {}).get("type", "")
        if meta_type in ("join_path", "join_chain"):
            if join_doc_count < MAX_JOIN_DOCS:
                result.append(h)
                join_doc_count += 1
        else:
            result.append(h)

    return result


def _build_schema_prompt(message: str, hits: list) -> str:
    schema_lines = "\n".join([f"- {h['doc']}" for h in hits if h.get("doc")])

    schema_meta = state.SCHEMA_META

    # Secciones dinámicas derivadas del esquema real de la BD
    secs = build_dynamic_prompt_sections(schema_meta) if schema_meta else {}
    aliases_section  = secs.get("aliases", "Usa alias cortos y seguros para las tablas.")
    columns_section  = secs.get("columns", "Usa nombres descriptivos; nunca expongas *Key en SELECT.")
    date_rules       = secs.get("date_rules", "NUNCA uses funciones de fecha del sistema. Usa columnas de fecha reales.")
    sales_source     = secs.get("sales_source", "")
    persons_section  = secs.get("persons", "")
    syntax_section   = secs.get("syntax", "")

    # Hint de fecha para la regla 2 (filtrar por año)
    if schema_meta and schema_meta.date_table:
        dta = schema_meta.date_alias
        dt  = schema_meta.date_table
        ds  = schema_meta.date_schema
        dk  = schema_meta.date_key_col
        yc  = schema_meta.date_year_col
        date_hint = (
            f"Para filtrar por año usa: JOIN {ds}.{dt} {dta} "
            f"ON <Fact>.OrderDateKey = {dta}.{dk} — luego WHERE {dta}.{yc} IN (...)"
        )
        chart_year_hint = (
            f"- Gráfico de ventas de UN año específico → "
            f"GROUP BY {dta}.MonthNumberOfYear ORDER BY {dta}.MonthNumberOfYear\n"
            f"- Gráfico de ventas sin año específico → "
            f"GROUP BY {dta}.{yc} ORDER BY {dta}.{yc}\n"
        )
    else:
        date_hint = "Para filtrar por año usa la tabla de tiempo disponible en el esquema."
        chart_year_hint = (
            "- Gráfico de ventas de UN año → desglose por mes.\n"
            "- Gráfico de ventas sin año → desglose por año.\n"
        )

    # Hints de análisis de negocio derivados de las tablas detectadas
    business_hints: list[str] = [
        "- Si la pregunta es analítica, infiere la métrica más relevante y responde con datos reales."
    ]
    if schema_meta:
        if len(schema_meta.union_fact_tables) >= 2:
            business_hints += [
                "- Para preguntas de regiones/países → combina todas las tablas de venta (AllSales) + tabla de territorio.",
                "- Para preguntas de productos → combina todas las tablas de venta (AllSales) + tabla de productos.",
            ]
        if schema_meta.customer_dim and schema_meta.customer_fact:
            ct  = schema_meta.customer_dim
            cft = schema_meta.customer_fact
            ca  = schema_meta.aliases.get(ct, "DC")
            cfa = schema_meta.aliases.get(cft, "FIS")
            business_hints.append(
                f"- Para preguntas de clientes → usa {cft} {cfa} + {ct} {ca} "
                f"(solo esa tabla tiene clientes directos)."
            )
        if schema_meta.date_table:
            dta = schema_meta.date_alias
            yc  = schema_meta.date_year_col
            business_hints.append(
                f"- Para tendencias temporales → usa AllSales + {schema_meta.date_table} {dta}, "
                f"filtra por {dta}.{yc}."
            )

    return (
        "Eres un experto en SQL Server y análisis de negocio. Reglas:\n"
        "0. ACCESO DE SOLO LECTURA (REGLA ABSOLUTA E IRROMPIBLE):\n"
        "   Esta base de datos es de SOLO LECTURA. Está COMPLETAMENTE PROHIBIDO generar, sugerir\n"
        "   o ejecutar cualquier sentencia DELETE, UPDATE, INSERT, DROP, TRUNCATE, ALTER, MERGE,\n"
        "   CREATE, EXEC o cualquier operación que modifique o elimine datos.\n"
        "   Si el usuario pide eliminar, borrar, modificar o insertar datos, responde EXACTAMENTE:\n"
        "   'Solo tengo acceso de lectura a la base de datos. No puedo eliminar, modificar ni "
        "insertar registros. Si necesitas hacer cambios, contacta al administrador del sistema.'\n"
        "   NUNCA digas que realizaste una operación de escritura. Solo puedes consultar (SELECT).\n"
        "   IMPORTANTE: este rechazo es SOLO para pedidos de escritura (eliminar/modificar/insertar "
        "datos reales). Una pregunta sobre el FUTURO sin nombrar una entidad específica (ej. '¿qué "
        "producto liderará?', '¿qué región será mejor?') NO es una operación de escritura y NUNCA "
        "debe responderse con este mensaje — no puedes predecir el futuro con certeza, pero SÍ "
        "puedes consultar y mostrar la tendencia HISTÓRICA real (quién ha liderado hasta ahora) "
        "y aclarar en tu respuesta que es una referencia histórica, no una predicción exacta.\n"
        "1. Usa SOLO tablas y columnas del esquema dado.\n"
        "2. Llama a run_sql EXACTAMENTE UNA VEZ. ABSOLUTAMENTE PROHIBIDO ejecutar run_sql más de una vez.\n"
        "   Si necesitas múltiples datos, combínalos en UNA sola query con CTEs o subconsultas.\n"
        "   Si la query falla NO reintentes — reporta el error exacto al usuario.\n"
        f"   {date_hint}\n"
        "3. Para limitar filas usa TOP N al inicio del SELECT principal: SELECT TOP N ...\n"
        "   ORDER BY va SIEMPRE en el SELECT final, NUNCA dentro de una CTE (SQL Server devuelve error).\n"
        "4. No uses **, ##, ni markdown en tus respuestas.\n"
        "5. Los valores de texto en la base de datos (países, categorías, nombres) están en INGLÉS "
        "aunque la pregunta esté en español — usa el equivalente en inglés en los filtros WHERE "
        "(ej. 'France' no 'Francia', 'Spain' no 'España', 'Germany' no 'Alemania', "
        "'United States' no 'Estados Unidos').\n\n"
        + (f"{syntax_section}\n" if syntax_section else "")
        + "ALIAS DE TABLAS (CRÍTICO — error frecuente):\n"
        "NUNCA uses palabras reservadas de SQL como alias de tabla o CTE. Lista negra PROHIBIDA:\n"
        "  IS, AS, IN, ON, BY, OR, AND, NOT, TO, AT, GO, IF, DO,\n"
        "  CURRENT, PREVIOUS, NEXT, KEY, SET, VALUE, USER, TABLE, VIEW,\n"
        "  INDEX, ORDER, GROUP, SELECT, WHERE, FROM, JOIN, CASE, WHEN\n"
        "En CTEs usa nombres descriptivos: YearlySales, SalesGrowth, CurYear, PrevYear, BaseData.\n"
        f"{aliases_section}\n"
        "Ejemplo CTE correcto:\n"
        "  WITH YearlySales AS (...) ,\n"
        "  SalesGrowth AS (SELECT CurYear.Col FROM YearlySales CurYear JOIN YearlySales PrevYear ...)\n\n"
        f"{columns_section}\n\n"
        "PORCENTAJES (CRÍTICO):\n"
        "SIEMPRE calcula porcentajes multiplicados por 100 para que el valor sea legible (57.0, no 0.57):\n"
        "  SalesAmount * 100.0 / NULLIF(TotalAmount, 0) AS Percentage\n"
        "Para análisis de concentración / Pareto ('territorios que concentran el X% de ventas'),\n"
        "usa suma acumulada con SUM() OVER (ORDER BY col DESC) y filtra por CumulativePct:\n"
        "  WITH Ranked AS (\n"
        "    SELECT col, SalesAmount, Pct,\n"
        "           SUM(Pct) OVER (ORDER BY SalesAmount DESC) AS CumulativePct\n"
        "    FROM ...\n"
        "  )\n"
        "  SELECT col, SalesAmount, Pct FROM Ranked\n"
        "  WHERE CumulativePct - Pct < 90.0\n"
        "  ORDER BY SalesAmount DESC\n\n"
        "FUNCIONES DE VENTANA (CRÍTICO):\n"
        "- LAG(), LEAD(), FIRST_VALUE(), LAST_VALUE() SIEMPRE requieren ORDER BY dentro del OVER().\n"
        "- Correcto: LAG(col) OVER (PARTITION BY grp ORDER BY YearCol)\n"
        "- NUNCA omitas el ORDER BY en el OVER de estas funciones — SQL Server lanza error 4112.\n"
        "- Para acceder a tablas encadenadas por FK usa los join paths del esquema proporcionado.\n\n"
        f"{date_rules}\n"
        f"{sales_source}\n"
        "ANÁLISIS DE NEGOCIO (importante):\n"
        + "\n".join(business_hints) + "\n\n"
        + (f"{persons_section}\n\n" if persons_section else "")
        + "USO DE RELACIONES:\n"
        "- Úsalas para hacer JOIN entre tablas correctamente.\n\n"
        "GRÁFICOS — SQL (CRÍTICO):\n"
        "Si el usuario pide un gráfico, la query DEBE devolver MÚLTIPLES FILAS para que sea significativa.\n"
        f"{chart_year_hint}"
        "- Gráfico por categoría, producto o territorio → GROUP BY la dimensión correspondiente.\n"
        "NUNCA devuelvas una sola fila cuando se pide un gráfico. Un gráfico con 1 punto no tiene sentido.\n\n"
        "FORMATO (obligatorio):\n"
        "- Pregunta analítica simple (total, promedio, conteo, mejor/peor) → SOLO texto narrativo, sin tabla.\n"
        "- Múltiples filas con múltiples columnas (ranking, comparación, top N) → tabla + párrafo breve.\n"
        "- Resultado de 1 sola fila O 1 sola columna → SOLO texto narrativo, SIN tabla.\n"
        "- Si se pidió un gráfico → tabla con los datos + párrafo de análisis.\n"
        "- Respeta exactamente el N de filas pedido (TOP N en el SQL).\n"
        "- NUNCA uses formato | col | col | (markdown pipe).\n"
        "- NÚMEROS: usa SIEMPRE coma como separador de miles y punto como decimal, "
        "ej. 1,234,567.89 (patrón de formato, no un valor real).\n\n"
        "ESQUEMA:\n"
        f"{schema_lines}\n\n"
        f"PREGUNTA: {message}\n\n"
        "SI NO HAY DATOS (CRÍTICO): Si run_sql devuelve 0 filas, o devuelve filas pero todos los "
        "valores son NULL/vacíos (ej. un filtro que no coincidió con ningún registro real), NUNCA "
        "inventes un número ni completes con un valor plausible — di explícitamente que no se "
        "encontraron resultados para esos filtros y sugiere qué cambiar (otro año, región, "
        "producto, etc.). Cualquier cifra que menciones debe venir literalmente del resultado de "
        "run_sql, nunca de un ejemplo de las reglas de formato ni de tu propio conocimiento.\n\n"
        "INSTRUCCIÓN FINAL (obligatoria): Después de mostrar los datos, escribe SIEMPRE un párrafo "
        "en español (2-4 oraciones) analizando los resultados. Menciona valores específicos, "
        "tendencias o el dato más destacado. Este párrafo va DESPUÉS de la tabla, nunca antes."
    )


def _get_pinned_tables() -> set[str]:
    """Retorna las tablas Fact principales que siempre deben estar en el contexto RAG."""
    if state.SCHEMA_META:
        return state.SCHEMA_META.pinned_tables()
    return {"dbo.FactInternetSales", "dbo.FactResellerSales"}


def _get_person_tables_for_question(question: str) -> set[str]:
    """
    Si la pregunta contiene un nombre propio (Nombre Apellido), retorna las tablas
    de empleados y/o clientes del esquema para incluirlas en el contexto RAG.
    Solo se activa si hay al menos un nombre propio en la pregunta.
    """
    schema_meta = state.SCHEMA_META
    if not schema_meta or not PERSON_NAME_RE.search(question):
        return set()
    tables: set[str] = set()
    for tname in (
        schema_meta.employee_dim,
        schema_meta.employee_fact,
        schema_meta.customer_dim,
        schema_meta.customer_fact,
    ):
        if tname:
            t = schema_meta.tables.get(tname)
            schema = t.schema if t else "dbo"
            tables.add(f"{schema}.{tname}")
    return tables


def _inject_schema_rag(message: str) -> str:
    if not state.SCHEMA_STORE:
        return message
    try:
        if _is_discovery_question(message):
            total = state.SCHEMA_STORE.count()
            all_hits = state.SCHEMA_STORE.query(message, k=total) if total > 0 else []
            hits = sorted(all_hits, key=lambda h: h.get("distance", 999))
        else:
            raw_hits = state.SCHEMA_STORE.query(message, k=state.RAG_K_FETCH)
            hits = _filter_and_deduplicate(raw_hits)

            # Calcular qué tablas schema ya están en los hits
            hit_tables = {
                f"{h.get('meta', {}).get('schema', 'dbo')}.{h.get('meta', {}).get('table', '')}"
                for h in hits
                if h.get('meta', {}).get('type') not in ('join_path', 'join_chain')
            }

            # Pinear tablas Fact principales (ventas totales siempre en contexto)
            for pinned in _get_pinned_tables():
                if pinned not in hit_tables:
                    pinned_hits = state.SCHEMA_STORE.query(pinned.split(".")[-1], k=1)
                    if pinned_hits:
                        hits.append(pinned_hits[0])
                        hit_tables.add(pinned)
                        logger.info("RAG: fact pineada → %s", pinned)

            # Pinear tablas de personas si la pregunta tiene "Nombre Apellido"
            for person_t in _get_person_tables_for_question(message):
                if person_t not in hit_tables:
                    person_hits = state.SCHEMA_STORE.query(person_t.split(".")[-1], k=1)
                    if person_hits:
                        hits.append(person_hits[0])
                        hit_tables.add(person_t)
                        logger.info("RAG: tabla de persona pineada → %s", person_t)

        if not hits:
            return message

        logger.info("RAG hits para '%s':", message)
        for h in hits:
            meta = h.get("meta") or {}
            tabla = meta.get("table") or meta.get("from_table", "?")
            tipo  = meta.get("type", "schema")
            logger.info("  score=%.4f tipo=%-10s tabla=%s", h.get("score", 0), tipo, tabla)

        return _build_schema_prompt(message, hits)
    except Exception:
        logger.exception("RAG falló")
        return message
