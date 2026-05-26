from __future__ import annotations

import json
import logging
import os
import uuid
import re
from decimal import Decimal
from typing import Any, Optional, AsyncGenerator

import httpx
from dotenv import load_dotenv

from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.core.user import UserResolver, User, RequestContext
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.tools import RunSqlTool
from vanna.capabilities.sql_runner import RunSqlToolArgs
from vanna.capabilities.file_system import FileSystem

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from .schema_store import SchemaVectorStore
from .database import SqlServerRunner

logger = logging.getLogger(__name__)

SCHEMA_STORE: SchemaVectorStore | None = None
SQL_RUNNER: SqlServerRunner | None = None
SQL_CACHE = None
TRACKING_TOOL: "TrackingSqlTool | None" = None

RAG_K_FETCH = 10
RAG_K_FINAL = 6
SQL_CACHE_THRESHOLD = 0.97
MAX_ROWS_LIMIT = 500
MAX_RESPONSE_CACHE_LEN = 100_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# NullFileSystem para no generar archivos csv

class NullFileSystem(FileSystem):
    async def write_file(self, filename, content, context=None, overwrite=False): pass
    async def read_file(self, filename, context=None): return ""
    async def list_files(self, context=None): return []
    async def delete_file(self, filename, context=None): pass
    async def exists(self, filename, context=None): return False
    async def is_directory(self, path, context=None): return False
    async def run_bash(self, command, context=None): return ""
    async def search_files(self, pattern, context=None): return []


# TrackingSqlTool

# Palabras reservadas de SQL Server que el LLM usa frecuentemente como alias
_RESERVED_ALIAS_REPLACEMENTS = {
    r'\bCurrent\b':  'CurRow',
    r'\bPrevious\b': 'PrevRow',
    r'\bNext\b':     'NextRow',
    r'\bPrev\b':     'PrevRow',
}


_SQL_KEYWORD_GUARDS = [
    # FETCH NEXT n ROWS — paginación estándar de SQL Server
    (r'\bFETCH\s+NEXT\b', 'FETCH __NEXT__'),
    # CURRENT_TIMESTAMP, CURRENT_DATE, etc.
    (r'\bCURRENT_', '__CURRENT_'),
]


def _remove_cte_order_by(sql: str) -> str:
    """Elimina ORDER BY dentro de CTEs sin TOP/FETCH (inválido en SQL Server)."""
    def _strip_if_no_top(m: re.Match) -> str:
        before = sql[max(0, m.start() - 600): m.start()]

        # No eliminar ORDER BY que esté dentro de un OVER() aún abierto
        last_over = max(before.upper().rfind('OVER ('), before.upper().rfind('OVER('))
        if last_over >= 0:
            depth = sum(1 if c == '(' else -1 if c == ')' else 0
                        for c in before[last_over:])
            if depth > 0:
                return m.group()  # ORDER BY pertenece a una función de ventana

        last_select = before.upper().rfind('SELECT')
        context = before[last_select:] if last_select >= 0 else before
        if re.search(r'\bTOP\b|\bFETCH\b', context, re.IGNORECASE):
            return m.group()   # ORDER BY válido (tiene TOP/FETCH)
        return ''              # ORDER BY inválido dentro de CTE

    return re.sub(
        r'\s+ORDER\s+BY\s+[\w\s,\.\[\]]+(?=\s*\))',
        _strip_if_no_top,
        sql,
        flags=re.IGNORECASE,
    )


_WINDOW_REQUIRES_ORDER_BY = re.compile(
    r'\b(LAG|LEAD|FIRST_VALUE|LAST_VALUE)\b', re.IGNORECASE
)
_DATE_LIKE_COL = re.compile(
    r'\b(CalendarYear|OrderDate|ShipDate|DueDate|\w+Year|\w+Date|\w+Month|\w+Quarter)\b',
    re.IGNORECASE,
)


def _fix_window_order_by(sql: str) -> str:
    """Añade ORDER BY faltante en OVER de LAG/LEAD/FIRST_VALUE/LAST_VALUE."""
    if not _WINDOW_REQUIRES_ORDER_BY.search(sql):
        return sql

    def _patch(m: re.Match) -> str:
        content = m.group(1)
        if re.search(r'\bORDER\s+BY\b', content, re.IGNORECASE):
            return m.group(0)  # ya tiene ORDER BY

        # Solo actuar si la función que precede a este OVER lo requiere
        preceding = sql[max(0, m.start() - 300): m.start()]
        if not _WINDOW_REQUIRES_ORDER_BY.search(preceding):
            return m.group(0)

        # Inferir columna ORDER BY — preferir CalendarYear u otra columna fecha
        date_cols = _DATE_LIKE_COL.findall(sql)
        order_col = date_cols[0] if date_cols else None
        if not order_col:
            return m.group(0)  # no se puede inferir con seguridad

        fixed = content.rstrip() + f' ORDER BY {order_col}'
        logger.info(
            "SQL corregido — ORDER BY %s añadido a OVER de función de ventana", order_col
        )
        return f'OVER ({fixed})'

    return re.sub(r'\bOVER\s*\(([^()]*)\)', _patch, sql, flags=re.IGNORECASE)


def _fix_missing_dimdate_join(sql: str) -> str:
    """Añade JOIN a DimDate en CTEs que usan DD. pero olvidaron el JOIN."""
    dimdate_join = "JOIN dbo.DimDate DD ON FIS.OrderDateKey = DD.DateKey"

    def _patch_cte(m: re.Match) -> str:
        body = m.group(0)
        # Solo actuar si el cuerpo usa DD. pero no tiene DimDate
        if not re.search(r'\bDD\.', body, re.IGNORECASE):
            return body
        if re.search(r'\bDimDate\b', body, re.IGNORECASE):
            return body
        # Insertar el JOIN después del primer JOIN/FROM que referencie FIS
        patched = re.sub(
            r'(JOIN\s+dbo\.\w+\s+(?:FIS|DST|DP|DC|DR)\s+ON\s+FIS\.\w+\s*=\s*\w+\.\w+)',
            r'\1\n    ' + dimdate_join,
            body,
            count=1,
            flags=re.IGNORECASE,
        )
        if patched != body:
            logger.info("SQL corregido — JOIN dbo.DimDate añadido a CTE sin él")
        return patched

    # Aplicar sobre cada bloque CTE: AS ( ... )
    return re.sub(
        r'AS\s*\(SELECT[\s\S]*?\)',
        _patch_cte,
        sql,
        flags=re.IGNORECASE,
    )


_LANG_PREFIX_FIXES = [
    # DimProductCategory: ProductCategoryName → EnglishProductCategoryName
    (re.compile(r'\b(?<!\bEnglish)(?<!\bSpanish)(?<!\bFrench)(ProductCategoryName)\b', re.IGNORECASE),
     'EnglishProductCategoryName'),
    # DimProductSubcategory: ProductSubcategoryName → EnglishProductSubcategoryName
    (re.compile(r'\b(?<!\bEnglish)(?<!\bSpanish)(?<!\bFrench)(ProductSubcategoryName)\b', re.IGNORECASE),
     'EnglishProductSubcategoryName'),
    # DimProduct: ProductName → EnglishProductName (solo cuando precede alias de tabla DPC./DPSC./DP.)
    (re.compile(r'\b(?<!\bEnglish)(?<!\bSpanish)(?<!\bFrench)(ProductDescription)\b', re.IGNORECASE),
     'EnglishDescription'),
]


def _fix_lang_prefix_columns(sql: str) -> str:
    """Corrige columnas de DimProductCategory/Subcategory que requieren prefijo English."""
    fixed = sql
    for pattern, replacement in _LANG_PREFIX_FIXES:
        new = pattern.sub(replacement, fixed)
        if new != fixed:
            logger.info("SQL corregido — columna renombrada a %s", replacement)
            fixed = new
    return fixed


def _sanitize_sql_aliases(sql: str) -> str:
    # 1. Corregir nombres de columna con prefijo de idioma faltante
    fixed = _fix_lang_prefix_columns(sql)

    # 2. Añadir ORDER BY faltante en funciones de ventana (LAG/LEAD/etc.)
    fixed = _fix_window_order_by(fixed)

    # 2. Eliminar ORDER BY inválido dentro de CTEs
    fixed = _remove_cte_order_by(fixed)
    if fixed != sql:
        logger.info("SQL corregido — ORDER BY eliminado de CTE sin TOP")

    # 3. Añadir JOIN a DimDate cuando falta en un CTE que lo referencia
    fixed = _fix_missing_dimdate_join(fixed)

    # 2. Proteger contextos donde las palabras son keywords válidos
    guarded = fixed
    for guard_pattern, placeholder in _SQL_KEYWORD_GUARDS:
        guarded = re.sub(guard_pattern, placeholder, guarded, flags=re.IGNORECASE)

    # 3. Reemplazar alias problemáticos
    sanitized = guarded
    for pattern, replacement in _RESERVED_ALIAS_REPLACEMENTS.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    # 4. Restaurar keywords protegidos
    sanitized = sanitized.replace('FETCH __NEXT__', 'FETCH NEXT')
    sanitized = sanitized.replace('__CURRENT_', 'CURRENT_')
    return sanitized


class TrackingSqlTool(RunSqlTool):
    def __init__(self, sql_runner, on_sql_executed=None):
        super().__init__(sql_runner=sql_runner, file_system=NullFileSystem())
        self._on_sql_executed = on_sql_executed or (lambda sql: None)
        self._in_flight: set[str] = set()

    def set_callback(self, callback):
        self._on_sql_executed = callback
        self._in_flight = set()  # reset por nueva solicitud

    async def execute(self, context, args):
        sql = getattr(args, "sql", None)
        if sql:
            sanitized = _sanitize_sql_aliases(sql)
            if sanitized != sql:
                logger.info("SQL sanitizado — alias reservados reemplazados")
                args = RunSqlToolArgs(sql=sanitized)
                sql = sanitized

        sql_key = (sql or "").strip()
        if sql_key and sql_key in self._in_flight:
            logger.warning("TrackingSqlTool — ejecución duplicada bloqueada")
            return None
        if sql_key:
            self._in_flight.add(sql_key)

        logger.info("TrackingSqlTool.execute() — SQL: %s", sql)
        result = await super().execute(context, args)
        if sql:
            try:
                self._on_sql_executed(sql)
            except Exception:
                logger.exception("Error en TrackingSqlTool callback")
        return result


# UserResolver

class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie("vanna_email") or "guest@example.com"
        group = "admin" if user_email == "admin@example.com" else "user"
        return User(id=user_email, email=user_email, group_memberships=[group])

# Normalización temporal para cache

def _normalize_question(question: str) -> str:
    """Reemplaza años, meses y trimestres por placeholders numerados."""
    years = re.findall(r'\b(20\d{2}|19\d{2})\b', question)
    normalized = question
    for i, year in enumerate(years):
        normalized = normalized.replace(year, f'__YEAR{i+1}__', 1)
    normalized = re.sub(
        r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b',
        '__MONTH__', normalized, flags=re.IGNORECASE
    )
    normalized = re.sub(
        r'\b(q1|q2|q3|q4|primer trimestre|segundo trimestre|tercer trimestre|cuarto trimestre)\b',
        '__QUARTER__', normalized, flags=re.IGNORECASE
    )
    return normalized.strip().lower()


def _extract_temporals(question: str) -> dict:
    """Extrae los valores temporales reales de la pregunta original."""
    temporals = {}
    years = re.findall(r'\b(20\d{2}|19\d{2})\b', question)
    for i, year in enumerate(years):
        temporals[f"__YEAR{i+1}__"] = year

    month_match = re.search(
        r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b',
        question, re.IGNORECASE
    )
    if month_match:
        temporals["__MONTH__"] = month_match.group(1).lower()

    quarter_match = re.search(
        r'\b(q1|q2|q3|q4|primer trimestre|segundo trimestre|tercer trimestre|cuarto trimestre)\b',
        question, re.IGNORECASE
    )
    if quarter_match:
        temporals["__QUARTER__"] = quarter_match.group(1).lower()

    return temporals


def _inject_temporals(text: str, temporals: dict) -> str:
    """Reemplaza placeholders por los valores reales."""
    for placeholder, value in temporals.items():
        text = text.replace(placeholder, value)
    return text


def _normalize_with_temporals(text: str, temporals: dict) -> str:
    """Reemplaza valores reales por placeholders."""
    normalized = text
    for placeholder, value in temporals.items():
        normalized = normalized.replace(value, placeholder)
    return normalized


# Generar análisis fresco con LLM

async def _generate_analysis(question: str, data_rows: list, columns: list) -> str:
    """Llama al LLM para generar un párrafo de análisis basado en los datos frescos."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return ""

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

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"Datos obtenidos:\n{data_str}\n\n"
        "Escribe UN párrafo en español (2-3 oraciones) respondiendo la pregunta con los datos de arriba.\n"
        "REGLAS ESTRICTAS:\n"
        "- Nombra EXACTAMENTE los valores que aparecen en los datos (nombres de territorios, productos, "
        "clientes, años, porcentajes, montos, etc.).\n"
        "- NUNCA uses frases vagas como 'el especificado en la consulta', 'los datos muestran', "
        "'según los resultados', 'el territorio analizado'. Si tienes el nombre, úsalo.\n"
        "- Si hay un número ganador (mejor, mayor, top), menciónalo primero.\n"
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
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Error generando análisis fresco")
        return ""


# SQL Cache

def _init_sql_cache(base_persist_dir: str):
    global SQL_CACHE

    cache_path = os.path.join(base_persist_dir, "sql_cache")
    os.makedirs(cache_path, exist_ok=True)

    client = PersistentClient(path=cache_path)
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    try:
        existing = client.get_collection(name="tiara_sql_cache")
        existing_meta = existing.metadata or {}
        existing_space = existing_meta.get("hnsw:space", "l2")
        existing_ef = existing_meta.get("embedding_function", "")

        needs_recreate = existing_space != "cosine" or existing_ef != "openai"
        if needs_recreate:
            logger.warning("Cache incompatible (space=%s, ef=%s). Recreando.", existing_space, existing_ef)
            client.delete_collection("tiara_sql_cache")
    except Exception:
        pass

    SQL_CACHE = client.get_or_create_collection(
        name="tiara_sql_cache",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine", "embedding_function": "openai"},
    )
    logger.info("SQL Cache inicializado (%d entradas)", SQL_CACHE.count())


def _search_sql_cache(question: str) -> Optional[dict]:
    if not SQL_CACHE:
        return None
    try:
        normalized = _normalize_question(question)

        results = SQL_CACHE.query(query_texts=[normalized], n_results=1)
        if (
            not results
            or not results.get("documents", [[]])[0]
            or not results.get("distances", [[]])[0]
            or not results.get("metadatas", [[]])[0]
        ):
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - (distance / 2.0)
        logger.info("Cache similarity: %.4f (threshold: %.2f)", similarity, SQL_CACHE_THRESHOLD)

        if similarity < SQL_CACHE_THRESHOLD:
            return None

        metadata = results["metadatas"][0][0]
        sql_template = metadata.get("sql", "")
        temporals = _extract_temporals(question)
        has_temporals = bool(temporals)

        if has_temporals:
            sql = _inject_temporals(sql_template, temporals)
            logger.info("Cache HIT con temporales sustituidos: %s", temporals)
        else:
            sql = sql_template

        return {
            "sql": sql,
            "full_response": "" if has_temporals else (metadata.get("full_response") or ""),
            "has_temporals": has_temporals,
        }
    except Exception:
        logger.exception("Error consultando SQL cache")
        return None


def _store_sql_cache(
    question: str,
    sql: str,
    full_response: str = "",
):
    if not SQL_CACHE:
        return

    if any(f in sql.lower() for f in ["current_date", "now()", "getdate()"]):
        logger.info("SQL no cacheado: contiene función de fecha dinámica")
        return

    if full_response and len(full_response) > MAX_RESPONSE_CACHE_LEN:
        full_response = full_response[:MAX_RESPONSE_CACHE_LEN]

    normalized_question = _normalize_question(question)
    temporals = _extract_temporals(question)
    normalized_sql = _normalize_with_temporals(sql, temporals)

    try:
        SQL_CACHE.add(
            ids=[str(uuid.uuid4())],
            documents=[normalized_question],
            metadatas=[{
                "sql": normalized_sql,
                # Para preguntas temporales no cacheamos full_response
                # porque el análisis contiene valores numéricos específicos
                "full_response": "" if temporals else (full_response or ""),
            }],
        )
        logger.info("SQL normalizado guardado en cache: '%s'", normalized_question)
    except Exception:
        logger.exception("Error guardando en cache")


# Helpers de texto

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
    "pct", "percent", "porcentaje", "porc",
    "ratio", "tasa", "share", "participacion", "participación",
    "margen", "margin", "rate", "variacion", "variación",
}


def _is_percent_col(col: str) -> bool:
    c = col.lower()
    return any(kw in c for kw in _PERCENT_KEYWORDS)


def _format_cell(v: Any, col: str = "") -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    skip = any(kw in col.lower() for kw in _SKIP_FORMAT_KEYWORDS)
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

CHART_SENTINEL = "\x00CHART\x00"

_CHART_KEYWORDS = {
    "gráfico", "grafico", "gráfica", "grafica", "chart", "plot",
    "visualiza", "visualización", "visualizacion", "diagrama",
    "barras", "barra", "pastel", "torta", "dona", "donut",
    "linea", "línea", "dispersión", "dispersion",
}

_PIE_KEYWORDS  = {"pastel", "torta", "dona", "donut", "pie"}
_LINE_KEYWORDS = {"linea", "línea", "tendencia", "evolución", "evolucion", "histórico", "historico"}
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
    q = question.lower()
    if any(kw in q for kw in _PIE_KEYWORDS):
        return "pie"
    if any(kw in q for kw in _LINE_KEYWORDS):
        return "line"
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


def _raw_numeric(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def _build_chart_payload(question: str, cols: list, rows: list) -> dict:
    if not cols or len(rows) < 1:
        return {}

    chart_type = _detect_chart_type(question, cols)
    label_col  = cols[0]
    value_cols = [c for c in cols[1:] if _is_numeric_col(c, rows)]
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

    labels  = [_safe_str(r.get(label_col, "")) for r in rows]
    rotate  = 40 if len(labels) > 6 else 0

    option: dict = {
        "backgroundColor": "transparent",
        "color": _EC_COLORS,
        "animation": True,
        "animationDuration": 900,
        "animationEasing": "cubicOut",
        "grid": {"left": "3%", "right": "4%", "bottom": "18%", "top": "8%", "containLabel": True},
    }

    if chart_type == "pie":
        values   = [_raw_numeric(r.get(value_cols[0])) for r in rows]
        pie_data = [{"name": l, "value": v} for l, v in zip(labels, values)]
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
            clear = area.replace("0.22", "0")
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


_VAGUE_PLACEHOLDERS = [" de x", " a y", " del z%", " de x,", "fueron de x", "aumentaron a y",
                       "ventas de x", "total de x", " x millones", " y millones"]


def _is_vague_analysis(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _VAGUE_PLACEHOLDERS)


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


# extract_text_from_component

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
                max_rows = min(len(rows), MAX_ROWS_LIMIT)
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
            return _clean_markdown(text)

        return ""

    sc = getattr(component, "simple_component", None)
    if sc is not None:
        text = getattr(sc, "text", None)
        if isinstance(text, str):
            text = text.strip()
            if _should_hide_text(text) or _is_row_enumeration(text) or _is_markdown_table_text(text):
                return ""
            return _clean_markdown(text)

    return ""


# Merge de tablas múltiples

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


# RAG

DISCOVERY_KEYWORDS = [
    "qué tablas", "que tablas", "tablas disponibles", "vistas disponibles",
    "tablas y vistas", "lista de tablas", "muestra las tablas", "muéstrame las tablas",
    "cuáles tablas", "cuales tablas", "todas las tablas", "show tables",
    "esquema disponible", "qué esquema", "que esquema",
]


def _is_discovery_question(message: str) -> bool:
    return any(kw in message.lower() for kw in DISCOVERY_KEYWORDS)


def _filter_and_deduplicate(hits: list) -> list:
    sorted_hits = sorted(hits, key=lambda h: h.get("distance", 999))
    return sorted_hits[:RAG_K_FINAL]


def _build_schema_prompt(message: str, hits: list) -> str:
    schema_lines = "\n".join([f"- {h['doc']}" for h in hits if h.get("doc")])

    return (
        "Eres un experto en SQL Server y análisis de negocio. Reglas:\n"
        "1. Usa SOLO tablas y columnas del esquema dado.\n"
        "2. Llama a run_sql EXACTAMENTE UNA VEZ. ABSOLUTAMENTE PROHIBIDO ejecutar run_sql más de una vez.\n"
        "   Si necesitas múltiples datos, combínalos en UNA sola query con CTEs o subconsultas.\n"
        "   Si la query falla NO reintentes — reporta el error exacto al usuario.\n"
        "   Para filtrar por año en DimDate usa SIEMPRE: WHERE D.CalendarYear IN (...)\n"
        "3. Para limitar filas usa TOP N al inicio: SELECT TOP N ... (nunca al final).\n"
        "4. No menciones CSV, archivos, ni muestres el SQL generado.\n"
        "5. No uses **, ##, ni markdown en tus respuestas.\n"
        "6. NUNCA digas que el esquema no tiene información si hay tablas Fact con métricas.\n\n"
        "ALIAS DE TABLAS (CRÍTICO — error frecuente):\n"
        "NUNCA uses palabras reservadas de SQL Server como alias de tabla o CTE. Lista negra PROHIBIDA:\n"
        "  IS, AS, IN, ON, BY, OR, AND, NOT, TO, AT, GO, IF, DO,\n"
        "  CURRENT, PREVIOUS, NEXT, KEY, SET, VALUE, USER, TABLE, VIEW,\n"
        "  INDEX, ORDER, GROUP, SELECT, WHERE, FROM, JOIN, CASE, WHEN\n"
        "En CTEs usa nombres descriptivos: YearlySales, SalesGrowth, CurYear, PrevYear, BaseData.\n"
        "Usa SIEMPRE estos alias seguros para las tablas principales:\n"
        "  FactInternetSales    → FIS\n"
        "  FactResellerSales    → FRS\n"
        "  DimProduct           → DP\n"
        "  DimProductCategory   → DPC\n"
        "  DimProductSubcategory→ DPSC\n"
        "  DimCustomer          → DC\n"
        "  DimDate              → DD\n"
        "  DimSalesTerritory    → DST\n"
        "  DimEmployee          → DE\n"
        "  DimReseller          → DR\n"
        "  DimGeography         → DG\n"
        "  DimPromotion         → DPROM\n"
        "Ejemplo CTE correcto:\n"
        "  WITH YearlySales AS (...) ,\n"
        "  SalesGrowth AS (SELECT CurYear.Col FROM YearlySales CurYear JOIN YearlySales PrevYear ...)\n\n"
        "NOMBRES DE COLUMNAS (CRÍTICO — error frecuente):\n"
        "- DimProductCategory: usa EnglishProductCategoryName (NO ProductCategoryName ni ProductCategoryKey).\n"
        "- DimProductSubcategory: usa EnglishProductSubcategoryName (NO ProductSubcategoryName ni ProductSubcategoryKey).\n"
        "- DimProduct: usa EnglishProductName (NO ProductName ni ProductKey).\n"
        "- DimSalesTerritory: usa SalesTerritoryRegion o SalesTerritoryCountry (NO SalesTerritoryKey).\n"
        "- DimCustomer: usa FirstName + ' ' + LastName AS Cliente (NO CustomerKey).\n"
        "- DimEmployee: usa FirstName + ' ' + LastName AS Vendedor (NO EmployeeKey).\n"
        "- DimReseller: usa ResellerName (NO ResellerKey).\n"
        "- DimGeography: usa City o EnglishCountryRegionName (NO GeographyKey).\n"
        "REGLA GENERAL: en el SELECT final NUNCA expongas columnas *Key como resultado visible. "
        "Los *Key solo sirven para JOIN — siempre muestra el nombre descriptivo de la dimensión.\n\n"
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
        "  WHERE CumulativePct - Pct < 90.0   -- incluye hasta cruzar el umbral\n"
        "  ORDER BY SalesAmount DESC\n\n"
        "FUNCIONES DE VENTANA (CRÍTICO):\n"
        "- LAG(), LEAD(), FIRST_VALUE(), LAST_VALUE() SIEMPRE requieren ORDER BY dentro del OVER().\n"
        "- Correcto: LAG(col) OVER (PARTITION BY grp ORDER BY CalendarYear)\n"
        "- NUNCA omitas el ORDER BY en el OVER de estas funciones — SQL Server lanza error 4112.\n"
        "- Para llegar a DimProductCategory desde DimProduct usa la cadena completa:\n"
        "  JOIN dbo.DimProductSubcategory DPSC ON DP.ProductSubcategoryKey = DPSC.ProductSubcategoryKey\n"
        "  JOIN dbo.DimProductCategory DPC ON DPSC.ProductCategoryKey = DPC.ProductCategoryKey\n\n"
        "AÑOS Y FECHAS (CRÍTICO):\n"
        "NUNCA uses GETDATE(), YEAR(GETDATE()), ni el año actual del sistema como referencia.\n"
        "Los datos disponibles en la base de datos tienen un rango histórico fijo.\n"
        "- Si la pregunta NO especifica un año → usa los años reales del dataset con:\n"
        "    SELECT MIN(CalendarYear), MAX(CalendarYear) FROM dbo.DimDate\n"
        "  o filtra con: WHERE DD.CalendarYear IN (SELECT DISTINCT CalendarYear FROM dbo.DimDate)\n"
        "- Para comparaciones 'año a año' → usa los años presentes en DimDate, "
        "    haciendo JOIN de la tabla consigo misma por CalendarYear y CalendarYear+1.\n"
        "- Para 'último año disponible' → usa: (SELECT MAX(CalendarYear) FROM dbo.DimDate)\n\n"
        "FUENTE DE VENTAS (CRÍTICO — regla absoluta):\n"
        "Las ventas totales = FactInternetSales + FactResellerSales combinadas.\n"
        "SIEMPRE incluye la columna Canal al hacer UNION ALL para poder distinguir la fuente:\n"
        "  WITH AllSales AS (\n"
        "      SELECT 'Internet' AS Canal, OrderDateKey, SalesAmount, TotalProductCost, UnitPrice\n"
        "      FROM dbo.FactInternetSales\n"
        "      UNION ALL\n"
        "      SELECT 'Reseller' AS Canal, OrderDateKey, SalesAmount, TotalProductCost, UnitPrice\n"
        "      FROM dbo.FactResellerSales\n"
        "  )\n"
        "Si la pregunta pide totales globales sin desglose por canal, omite la columna Canal y agrupa directamente.\n"
        "Si la pregunta pide desglose 'por canal', incluye Canal en SELECT y GROUP BY.\n"
        "NUNCA intentes inferir el canal desde valores de columnas después del UNION ALL.\n"
        "Excepción: usa SOLO FactInternetSales si el usuario dice 'online', 'internet' o 'canal directo'.\n"
        "Excepción: usa SOLO FactResellerSales si el usuario dice 'reseller', 'distribuidor' o 'canal indirecto'.\n"
        "NUNCA uses FactResellerSalesXL_PageCompressed ni FactResellerSalesXL_CCI.\n\n"
        "ANÁLISIS DE NEGOCIO (importante):\n"
        "- Si la pregunta es analítica o estratégica, infiere la métrica más relevante "
        "(SalesAmount, OrderQuantity) y responde con datos reales.\n"
        "- Para preguntas de regiones → combina ambas tablas (AllSales) + DimSalesTerritory DST.\n"
        "- Para preguntas de productos → combina ambas tablas (AllSales) + DimProduct DP.\n"
        "- Para preguntas de clientes → usa FactInternetSales FIS + DimCustomer DC (solo internet tiene clientes directos).\n"
        "- Para tendencias temporales → combina ambas tablas (AllSales) + DimDate DD, filtra por DD.CalendarYear.\n\n"
        "USO DE RELACIONES:\n"
        "- Úsalas para hacer JOIN entre tablas correctamente.\n\n"
        "GRÁFICOS — SQL (CRÍTICO):\n"
        "Si el usuario pide un gráfico, la query DEBE devolver MÚLTIPLES FILAS para que sea significativa.\n"
        "- Gráfico de ventas de UN año específico → desglose por mes: GROUP BY DD.MonthNumberOfYear ORDER BY DD.MonthNumberOfYear\n"
        "- Gráfico de ventas sin año específico → desglose por año: GROUP BY DD.CalendarYear ORDER BY DD.CalendarYear\n"
        "- Gráfico por categoría, producto o territorio → GROUP BY la dimensión correspondiente\n"
        "NUNCA devuelvas una sola fila cuando se pide un gráfico. Un gráfico con 1 punto no tiene sentido.\n\n"
        "FORMATO (obligatorio):\n"
        "- Pregunta analítica simple (total, promedio, conteo, mejor/peor) → SOLO texto narrativo, sin tabla.\n"
        "- Múltiples filas con múltiples columnas (ranking, comparación, top N) → tabla seguida de un párrafo breve de análisis.\n"
        "- Resultado de 1 sola fila O 1 sola columna → SOLO texto narrativo, SIN tabla.\n"
        "- Si se pidió un gráfico → tabla con los datos + párrafo de análisis.\n"
        "- Respeta exactamente el N de filas pedido (TOP N en el SQL).\n"
        "- NUNCA uses formato | col | col | (markdown pipe).\n\n"
        "ESQUEMA:\n"
        f"{schema_lines}\n\n"
        f"PREGUNTA: {message}\n\n"
        "INSTRUCCIÓN FINAL (obligatoria): Después de mostrar los datos, escribe SIEMPRE un párrafo "
        "en español (2-4 oraciones) analizando los resultados. Menciona valores específicos, "
        "tendencias o el dato más destacado. Este párrafo va DESPUÉS de la tabla, nunca antes."
    )


_PINNED_TABLES = {"dbo.FactInternetSales", "dbo.FactResellerSales"}

_GEO_COUNTRY_KEYWORDS = {
    "país", "pais", "region", "región", "territorio",
    "canada", "united states", "estados unidos", "australia", "germany",
    "alemania", "france", "francia", "united kingdom", "reino unido",
    "north america", "europe", "pacific", "norteamerica", "europa",
}
_GEO_CITY_KEYWORDS = {
    "ciudad", "estado", "provincia", "código postal", "codigo postal",
    "city", "state", "postal", "zip",
}
_GEO_KEYWORDS = _GEO_COUNTRY_KEYWORDS | _GEO_CITY_KEYWORDS
_GEO_PINNED = {"dbo.DimSalesTerritory"}
_GEO_CITY_PINNED = {"dbo.DimGeography"}

_PRODUCT_CAT_KEYWORDS = {
    "categoría", "categoria", "categorias", "categorías",
    "subcategoría", "subcategoria", "subcategorias", "subcategorías",
    "tipo de producto", "clase de producto", "grupo de producto",
    "bicicletas", "ropa", "accesorios", "componentes",
    "crecimiento por categoria", "ventas por categoria",
}
_PRODUCT_CAT_PINNED = {"dbo.DimProductSubcategory", "dbo.DimProductCategory"}


def _inject_schema_rag(message: str) -> str:
    if not SCHEMA_STORE:
        return message
    try:
        if _is_discovery_question(message):
            total = SCHEMA_STORE.count()
            all_hits = SCHEMA_STORE.query(message, k=total) if total > 0 else []
            hits = sorted(all_hits, key=lambda h: h.get("distance", 999))
        else:
            raw_hits = SCHEMA_STORE.query(message, k=RAG_K_FETCH)
            hits = _filter_and_deduplicate(raw_hits)

            # Tablas siempre incluidas + geo cuando la pregunta lo requiere
            q_lower = message.lower()
            extra_pins = set(_PINNED_TABLES)
            if any(kw in q_lower for kw in _GEO_KEYWORDS):
                extra_pins |= _GEO_PINNED
            if any(kw in q_lower for kw in _GEO_CITY_KEYWORDS):
                extra_pins |= _GEO_CITY_PINNED
            if any(kw in q_lower for kw in _PRODUCT_CAT_KEYWORDS):
                extra_pins |= _PRODUCT_CAT_PINNED

            hit_tables = {
                f"{h.get('meta', {}).get('schema', 'dbo')}.{h.get('meta', {}).get('table', '')}"
                for h in hits
            }
            for pinned in extra_pins:
                if pinned not in hit_tables:
                    pinned_table = pinned.split(".")[-1]
                    pinned_hits = SCHEMA_STORE.query(pinned_table, k=1)
                    if pinned_hits:
                        hits.append(pinned_hits[0])
                        logger.info("RAG: tabla pineada añadida → %s", pinned)

        if not hits:
            return message

        logger.info("RAG hits para '%s':", message)
        for h in hits:
            logger.info("  score=%.4f tabla=%s", h.get("score", 0), h.get("meta", {}).get("table"))

        return _build_schema_prompt(message, hits)
    except Exception:
        logger.exception("RAG falló")
        return message


# Stream principal

async def run_agent_stream_text(
    agent: Agent,
    request_context: RequestContext,
    message: str,
    conversation_id: Optional[str],
) -> AsyncGenerator[str, None]:

    original_question = message

    # 1. Cache
    cache_hit = _search_sql_cache(original_question)
    if cache_hit:
        cached_sql    = cache_hit.get("sql")
        full_response = cache_hit.get("full_response") or ""
        has_temporals = cache_hit.get("has_temporals", False)

        is_chart = _is_chart_question(original_question)

        # Match exacto: partir full_response en tabla + texto para que el
        # frontend pueda animar cada sección por separado.
        # Si es pregunta de gráfico y la respuesta cacheada no tiene tabla
        # (fue respuesta de 1 fila), ignorar caché y regenerar con RAG.
        if full_response and not has_temporals:
            if is_chart and '<table' not in full_response:
                logger.info("Cache HIT ignorado: pregunta de gráfico con respuesta sin tabla — regenerando")
            else:
                logger.info("Cache HIT con full_response")
                if '<table' in full_response and '</table>' in full_response:
                    table_end = full_response.lower().rfind('</table>') + len('</table>')
                    yield full_response[:table_end].strip()      # → tipo 'table' → fade-in
                    text_part = full_response[table_end:].strip()
                    if text_part:
                        yield text_part                          # → tipo 'text' → typewriter
                else:
                    yield full_response                          # respuesta solo-texto
                return

        # Con temporales: re-ejecutar SQL para obtener valores correctos.
        # Si es pregunta de gráfico y el SQL solo devuelve 1 fila, ignorar
        # caché y regenerar para que el LLM genere desglose por mes/categoría.
        if cached_sql and SQL_RUNNER:
            try:
                tool_args = RunSqlToolArgs(sql=cached_sql)
                df = await SQL_RUNNER.run_sql(tool_args, None)
                logger.info("Cache HIT (SQL re-ejecutado) — %d filas", len(df))

                if not df.empty:
                    if is_chart and len(df) < 2:
                        logger.info("Cache HIT ignorado: pregunta de gráfico con 1 fila — regenerando con RAG")
                    else:
                        cols = df.columns.tolist()
                        rows = df.to_dict("records")

                        if len(cols) > 1 and len(rows) > 1:
                            html = ['<table class="data-table"><thead><tr>']
                            for col in cols:
                                html.append(f'<th>{_safe_str(col)}</th>')
                            html.append('</tr></thead><tbody>')
                            for r in rows[:MAX_ROWS_LIMIT]:
                                html.append('<tr>')
                                for c in cols:
                                    html.append(f'<td>{_format_cell(r.get(c, ""), c)}</td>')
                                html.append('</tr>')
                            html.append('</tbody></table>')
                            yield "\n".join(html)

                        analysis = await _generate_analysis(original_question, rows, cols)
                        if analysis:
                            yield analysis

                        if is_chart:
                            chart_payload = _build_chart_payload(original_question, cols, rows)
                            if chart_payload:
                                yield CHART_SENTINEL + json.dumps(chart_payload)
                                logger.info("Gráfico ECharts generado desde cache")
                        return
                else:
                    yield "La consulta no devolvió resultados."
                    return
            except Exception:
                logger.exception("Error re-ejecutando SQL desde cache, continuando con flujo normal")

    # 2. RAG
    message = _inject_schema_rag(message)

    # 3. Callback SQL
    captured_sql: list[str] = []

    def on_sql_executed(sql: str):
        if sql not in captured_sql:
            captured_sql.append(sql)
            logger.info("SQL capturado: %s", sql)

    if TRACKING_TOOL:
        TRACKING_TOOL.set_callback(on_sql_executed)
        logger.info("TrackingSqlTool callback actualizado (id=%s)", id(TRACKING_TOOL))
    else:
        logger.warning("TRACKING_TOOL no inicializado")

    # 4. Ejecutar agente
    pre_table_buffer: list[str] = []
    tabla_chunks: list[str] = []
    post_table_chunks: list[str] = []
    table_seen = False

    try:
        async for component in agent.send_message(
            request_context=request_context,
            message=message,
            conversation_id=conversation_id,
        ):
            text_output = extract_text_from_component(component)
            if not text_output:
                continue

            is_table = '<table' in text_output
            if is_table:
                table_seen = True
                tabla_chunks.append(text_output)
            elif table_seen:
                post_table_chunks.append(text_output)
            else:
                pre_table_buffer.append(text_output)

    except Exception:
        logger.exception("Error en agent.send_message")
        yield "Ocurrió un error procesando tu pregunta. Por favor intenta de nuevo."
        return

    # 5. Emitir respuesta
    logger.info(
        "Buffers — pre_table:%d tabla:%d post_table:%d",
        len(pre_table_buffer), len(tabla_chunks), len(post_table_chunks),
    )
    if post_table_chunks:
        logger.info("post_table_chunks[0][:120]: %s", post_table_chunks[0][:120])
    elif table_seen:
        logger.warning("Tabla encontrada pero post_table_chunks está vacío — el LLM no generó análisis posterior")

    response_chunks: list[str] = []

    if table_seen and tabla_chunks:
        merged_table = _merge_multiple_tables(tabla_chunks) if len(tabla_chunks) > 1 else tabla_chunks[0]
        response_chunks.append(merged_table)
        yield merged_table

        combined_post = " ".join(post_table_chunks)
        if post_table_chunks and not _is_vague_analysis(combined_post):
            for chunk in post_table_chunks:
                response_chunks.append(chunk)
                yield chunk
        elif captured_sql and SQL_RUNNER:
            if post_table_chunks:
                logger.warning("Análisis del LLM detectado como vago — reemplazando con fallback")
            best_sql = captured_sql[-1]
            try:
                tool_args = RunSqlToolArgs(sql=best_sql)
                df = await SQL_RUNNER.run_sql(tool_args, None)
                if not df.empty:
                    cols = df.columns.tolist()
                    rows = df.to_dict("records")
                    analysis = await _generate_analysis(original_question, rows, cols)
                    if analysis:
                        response_chunks.append(analysis)
                        yield analysis
                        logger.info("Análisis de respaldo generado correctamente")
            except Exception:
                logger.exception("Error generando análisis de respaldo")
    elif captured_sql and SQL_RUNNER:
        # El stream produjo output vacío pero el SQL fue ejecutado — fallback completo
        logger.warning("Stream vacío con SQL capturado — activando fallback de renderizado")
        df = None
        for sql_attempt in reversed(captured_sql):
            try:
                tool_args = RunSqlToolArgs(sql=sql_attempt)
                df = await SQL_RUNNER.run_sql(tool_args, None)
                logger.info("Fallback SQL exitoso (intento con SQL más reciente)")
                break
            except Exception as e:
                logger.warning("Fallback SQL falló, probando anterior: %s", e)
        if df is None:
            yield "Ocurrió un error al procesar los resultados."
            return
        try:
            if not df.empty:
                cols = df.columns.tolist()
                rows = df.to_dict("records")

                if len(cols) > 1 and len(rows) > 1:
                    html = ['<table class="data-table"><thead><tr>']
                    for col in cols:
                        html.append(f'<th>{_safe_str(col)}</th>')
                    html.append('</tr></thead><tbody>')
                    for r in rows[:MAX_ROWS_LIMIT]:
                        html.append('<tr>')
                        for c in cols:
                            html.append(f'<td>{_format_cell(r.get(c, ""), c)}</td>')
                        html.append('</tr>')
                    html.append('</tbody></table>')
                    table_html = "\n".join(html)
                    response_chunks.append(table_html)
                    yield table_html

                analysis = await _generate_analysis(original_question, rows, cols)
                if analysis:
                    response_chunks.append(analysis)
                    yield analysis
                    logger.info("Fallback completo emitido correctamente")
            else:
                msg = "La consulta no devolvió resultados."
                response_chunks.append(msg)
                yield msg
        except Exception:
            logger.exception("Error en fallback de renderizado")
            yield "Ocurrió un error al procesar los resultados."
    else:
        # Sin SQL capturado — emitir texto conversacional del LLM (si lo hay)
        for chunk in pre_table_buffer:
            response_chunks.append(chunk)
            yield chunk

    # 6. Gráfico (si el usuario lo pidió y hay SQL disponible)
    if _is_chart_question(original_question) and captured_sql and SQL_RUNNER:
        try:
            tool_args = RunSqlToolArgs(sql=captured_sql[-1])
            df = await SQL_RUNNER.run_sql(tool_args, None)
            if not df.empty and len(df.columns) >= 2 and len(df) >= 1:
                chart_payload = _build_chart_payload(
                    original_question,
                    df.columns.tolist(),
                    df.to_dict("records"),
                )
                if chart_payload:
                    yield CHART_SENTINEL + json.dumps(chart_payload)
                    logger.info("Gráfico ECharts generado y enviado")
        except Exception:
            logger.exception("Error generando gráfico")

    # 7. Guardar en cache — usar el último SQL capturado (más reciente y probablemente correcto)
    if captured_sql:
        _store_sql_cache(
            original_question,
            captured_sql[-1],
            full_response="\n".join(response_chunks),
        )
        logger.info("SQL guardado en cache para: '%s'", original_question)
    else:
        logger.info("No se capturó SQL, no se guarda en cache")


# Build agent

def build_agent() -> Agent:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    project_root = os.path.dirname(os.path.dirname(__file__))
    base_persist_dir = os.path.join(project_root, "backend", "vanna_chromadb")

    path_memory = os.path.join(base_persist_dir, "agent_memory")
    path_schema = os.path.join(base_persist_dir, "schema_store")

    os.makedirs(path_memory, exist_ok=True)
    os.makedirs(path_schema, exist_ok=True)

    _init_sql_cache(base_persist_dir)

    api_key = os.getenv("OPENAI_API_KEY")
    odbc_str = os.getenv("SQLSERVER_ODBC")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    global SQL_RUNNER, TRACKING_TOOL
    SQL_RUNNER = SqlServerRunner(odbc_str)

    llm = OpenAILlmService(model=model, api_key=api_key)

    tools = ToolRegistry()
    TRACKING_TOOL = TrackingSqlTool(sql_runner=SQL_RUNNER)
    tools.register_local_tool(TRACKING_TOOL, access_groups=["admin", "user"])
    logger.info("TrackingSqlTool registrado (id=%s)", id(TRACKING_TOOL))

    agent_memory = ChromaAgentMemory(
        persist_directory=path_memory,
        collection_name=os.getenv("CHROMA_COLLECTION") or "vanna_tool_memory",
    )

    global SCHEMA_STORE
    SCHEMA_STORE = SchemaVectorStore(
        persist_dir=path_schema,
        collection_name=os.getenv("SCHEMA_COLLECTION") or "tiara_schema",
        embedding_mode="default",
    )

    if SCHEMA_STORE.count() == 0:
        logger.info("Schema store vacío — ejecutando ingest automático del esquema SQL")
        try:
            from .ingest_schema import ingest as _ingest
            _ingest(target_store=SCHEMA_STORE)
            logger.info("Auto-ingest completado: %d entradas", SCHEMA_STORE.count())
        except Exception:
            logger.exception("Error en auto-ingest del esquema — el agente continuará sin RAG de esquema")

    return Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=SimpleUserResolver(),
        agent_memory=agent_memory,
    )