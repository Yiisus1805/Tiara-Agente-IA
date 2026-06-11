from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
import re
from contextvars import ContextVar
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

# FK col set — cargado al arrancar desde la BD real, sin hardcoding
# Contiene (table_name_lower, col_name_lower) para todos los lados FK y PK
_FK_COL_SET: set[tuple[str, str]] = set()

# Contexto por-request: aislamiento total entre requests concurrentes
_ctx_sql_callback: ContextVar = ContextVar("_tiara_sql_cb", default=None)
_ctx_sql_question: ContextVar = ContextVar("_tiara_sql_q", default="")
_ctx_sql_inflight: ContextVar = ContextVar("_tiara_sql_inflight", default=None)
_ctx_intent: ContextVar = ContextVar("_tiara_intent", default="")

RAG_K_FETCH = 15
RAG_K_FINAL = 8
SQL_CACHE_THRESHOLD = 0.98
MAX_ROWS_LIMIT = 500
MAX_RESPONSE_CACHE_LEN = 100_000

CHART_SENTINEL = "\x00CHART\x00"
ERROR_RETRY_SENTINEL = "\x00ERROR_RETRY\x00"
TABLE_FLUSH_SENTINEL = "\x00TABLE_FLUSH\x00"  # indica que no viene chart: emitir tabla ya
AGENT_STEP_TIMEOUT = float(os.getenv("AGENT_STEP_TIMEOUT", "90"))

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


def _remove_cte_order_by(sql: str) -> tuple[str, str]:
    """Elimina ORDER BY dentro de CTEs sin TOP/FETCH (inválido en SQL Server).

    Devuelve (sql_modificado, order_by_eliminado) para que el llamador pueda
    moverlo al SELECT final si corresponde.
    """
    last_removed: list[str] = []

    def _strip_if_no_top(m: re.Match) -> str:
        before = sql[max(0, m.start() - 600): m.start()]

        last_over = max(before.upper().rfind('OVER ('), before.upper().rfind('OVER('))
        if last_over >= 0:
            depth = sum(1 if c == '(' else -1 if c == ')' else 0
                        for c in before[last_over:])
            if depth > 0:
                return m.group()

        last_select = before.upper().rfind('SELECT')
        context = before[last_select:] if last_select >= 0 else before
        if re.search(r'\bTOP\b|\bFETCH\b', context, re.IGNORECASE):
            return m.group()

        last_removed.append(m.group().strip())
        return ''

    result = re.sub(
        r'\s+ORDER\s+BY\s+[\w\s,\.\[\]]+(?=\s*\))',
        _strip_if_no_top,
        sql,
        flags=re.IGNORECASE,
    )
    return result, last_removed[-1] if last_removed else ""


_FETCH_FIRST_RE = re.compile(
    r'\bFETCH\s+FIRST\s+(\d+)\s+ROWS?\s+ONLY\b', re.IGNORECASE
)


def _fix_fetch_first(sql: str) -> str:
    """Convierte FETCH FIRST N ROWS ONLY → SELECT TOP N (SQL Server no lo soporta)."""
    m = _FETCH_FIRST_RE.search(sql)
    if not m:
        return sql

    n = m.group(1)
    pre_fetch = sql[:m.start()]

    # El SELECT principal está después del último ')' que cierra los CTEs
    last_close = pre_fetch.rfind(')')
    search_from = last_close + 1 if last_close >= 0 else 0
    outer_pos = pre_fetch[search_from:].upper().find('SELECT')

    if outer_pos >= 0:
        abs_pos = search_from + outer_pos
        fixed = (
            sql[:abs_pos]
            + f'SELECT TOP {n}'
            + sql[abs_pos + 6: m.start()]
            + sql[m.end():]
        )
    else:
        fixed = pre_fetch.rstrip() + sql[m.end():]

    logger.info("SQL corregido — FETCH FIRST %s ROWS ONLY → SELECT TOP %s", n, n)
    return fixed


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


def _extract_cte_spans(sql: str) -> list[tuple[int, int]]:
    """Devuelve (inicio, fin) del contenido de cada CTE usando conteo de paréntesis."""
    spans = []
    for m in re.finditer(r'\bAS\s*\(', sql, re.IGNORECASE):
        start = m.end()
        depth, pos = 1, start
        while pos < len(sql) and depth > 0:
            if sql[pos] == '(':
                depth += 1
            elif sql[pos] == ')':
                depth -= 1
            pos += 1
        spans.append((start, pos - 1))
    return spans


def _fix_missing_dimdate_join(sql: str) -> str:
    """Añade JOIN a DimDate en CTEs que usan DD. pero olvidaron el JOIN."""
    dimdate_join = "JOIN dbo.DimDate DD ON FIS.OrderDateKey = DD.DateKey"
    result = sql
    offset = 0

    for start, end in _extract_cte_spans(sql):
        body = result[start + offset: end + offset]

        if not re.search(r'\bDD\.', body, re.IGNORECASE):
            continue
        if re.search(r'\bDimDate\b', body, re.IGNORECASE):
            continue

        # Insertar después del último JOIN que referencie una tabla conocida
        last_join = None
        for m in re.finditer(
            r'JOIN\s+dbo\.\w+\s+(?:FIS|DST|DP|DC|DR|FRS)\s+ON\s+\w+\.\w+\s*=\s*\w+\.\w+',
            body, re.IGNORECASE,
        ):
            last_join = m

        if not last_join:
            continue

        insert_at = start + offset + last_join.end()
        addition = f"\n    {dimdate_join}"
        result = result[:insert_at] + addition + result[insert_at:]
        offset += len(addition)
        logger.info("SQL corregido — JOIN dbo.DimDate DD añadido a CTE que usaba DD. sin él")

    return result


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

    # 3. Eliminar ORDER BY inválido dentro de CTEs y capturarlo para moverlo
    fixed, removed_order_by = _remove_cte_order_by(fixed)
    if removed_order_by:
        logger.info("SQL corregido — ORDER BY eliminado de CTE sin TOP")
        # Si el SELECT final tiene TOP pero no ORDER BY, mover allí el ORDER BY capturado
        has_top_outer = bool(re.search(r'\bSELECT\s+TOP\b', fixed, re.IGNORECASE))
        has_order_outer = bool(re.search(r'\bORDER\s+BY\b', fixed, re.IGNORECASE))
        if has_top_outer and not has_order_outer:
            fixed = fixed.rstrip().rstrip(';').rstrip() + '\n' + removed_order_by + ';'
            logger.info("SQL corregido — ORDER BY movido al SELECT final")

    # 4. Convertir FETCH FIRST N ROWS ONLY → SELECT TOP N
    fixed = _fix_fetch_first(fixed)

    # 5. Añadir JOIN a DimDate cuando falta en un CTE que lo referencia
    fixed = _fix_missing_dimdate_join(fixed)

    # 6. Proteger contextos donde las palabras son keywords válidos
    guarded = fixed
    for guard_pattern, placeholder in _SQL_KEYWORD_GUARDS:
        guarded = re.sub(guard_pattern, placeholder, guarded, flags=re.IGNORECASE)

    # 7. Reemplazar alias problemáticos
    sanitized = guarded
    for pattern, replacement in _RESERVED_ALIAS_REPLACEMENTS.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    # 8. Restaurar keywords protegidos
    sanitized = sanitized.replace('FETCH __NEXT__', 'FETCH NEXT')
    sanitized = sanitized.replace('__CURRENT_', 'CURRENT_')
    return sanitized


# ── HARDENING: BLOQUEO DE SQL DESTRUCTIVO ────────────────────────────────────

# Operaciones que modifican o destruyen datos — nunca deben ejecutarse
_DESTRUCTIVE_SQL_RE = re.compile(
    r'\b('
    r'DELETE\b|'
    r'UPDATE\b|'
    r'INSERT\b|'
    r'DROP\b|'
    r'TRUNCATE\b|'
    r'ALTER\b|'
    r'MERGE\b|'
    r'EXEC\b|'
    r'EXECUTE\b|'
    r'CREATE\b|'
    r'GRANT\b|'
    r'REVOKE\b|'
    r'DENY\b'
    r')',
    re.IGNORECASE,
)

# Patrones de inyección SQL — solo casos realmente peligrosos
# ';' y '--' solos son SQL válido; solo son peligrosos cuando van seguidos de comandos destructivos
_INJECTION_RE = re.compile(
    r'('
    r';\s*\b(?:DELETE|DROP|UPDATE|INSERT|ALTER|TRUNCATE|EXEC(?:UTE)?|GRANT|REVOKE)\b|'  # multi-statement
    r'\bOR\b\s+[\'"]?\s*[0-9\'"][^\w]|'          # OR 1=1 / OR '1'='1'
    r'\bAND\b\s+[\'"]?\s*[0-9\'"][^\w]|'         # AND 1=1 / AND '1'='1'
    r'\bxp_\w+|'                                   # xp_cmdshell y similares
    r'\bsp_(?:executesql|oa\w+|cmdexec)\b|'        # stored procs de sistema peligrosos
    r'\bWAITFOR\s+DELAY\b|'                        # time-based blind injection
    r'\bOPENROWSET\b|\bOPENDATASOURCE\b|\bOPENQUERY\b'  # acceso externo
    r')',
    re.IGNORECASE,
)

# Intención destructiva sobre la BD — requiere verbo + objeto explícito de BD
# NO captura: "datos actualizados", "cambió el valor", "modificaciones históricas"
_DB_OBJECT = r'(?:registro|dato|fila|entrada|tabla|record)'
_DESTRUCTIVE_INTENT_RE = re.compile(
    r'\b('
    # Eliminar/borrar + objeto de BD explícito
    r'elimin[ae]r?\s+(?:el|la|los|las|este|ese|un|una|todos?|todas?)\s*' + _DB_OBJECT + r'|'
    r'borr[ae]r?\s+(?:el|la|los|las|este|ese|un|una|todos?|todas?)\s*' + _DB_OBJECT + r'|'
    r'suprim[ei]r?\s+(?:el|la|los|las|este|ese)\s*' + _DB_OBJECT + r'|'
    # SQL keywords escritos en lenguaje natural con objeto de BD
    r'drop\s+(?:la\s+)?tabla|'
    r'delete\s+(?:de|from|el|la)\b|'
    r'trunca[rt](?:e|ar)?\s+(?:la\s+)?tabla|'
    # Insertar un registro nuevo (acepta "un nuevo registro", "un registro", "nueva fila")
    r'insert[ae]r?\s+(?:un|una|nuevo|nueva)(?:\s+(?:nuevo|nueva))?\s*' + _DB_OBJECT + r'|'
    r'agrega[r]?\s+(?:un|una|nuevo|nueva)(?:\s+(?:nuevo|nueva))?\s*' + _DB_OBJECT + r'|'
    # Modificar/actualizar/cambiar + objeto de BD explícito (no pasado, no adjetivo)
    r'(?:modific[ae]r?|actualiz[ae]r?|cambi[ae]r?|edit[ae]r?)\s+'
    r'(?:el|la|los|las|un|una)\s*(?:' + _DB_OBJECT + r'|campo|valor\s+de\s+\w+)'
    r')',
    re.IGNORECASE,
)

_READONLY_REFUSAL = (
    "Solo tengo acceso de lectura a la base de datos. "
    "No puedo eliminar, modificar ni insertar registros. "
    "Si necesitas hacer cambios en los datos, contacta al administrador del sistema."
)


def _check_sql_safety(sql: str) -> str | None:
    """Retorna un mensaje de error si el SQL es destructivo o sospechoso, None si es seguro."""
    destructive = _DESTRUCTIVE_SQL_RE.search(sql)
    if destructive:
        logger.error(
            "SQL DESTRUCTIVO BLOQUEADO (keyword: %s): %.200s",
            destructive.group(0).upper(), sql,
        )
        return f"Operación '{destructive.group(0).upper()}' bloqueada — TIARA es de solo lectura."

    injection = _INJECTION_RE.search(sql)
    if injection:
        logger.error(
            "POSIBLE INYECCIÓN SQL BLOQUEADA (patrón: %r): %.200s",
            injection.group(0), sql,
        )
        return "Consulta bloqueada por contener patrones no permitidos."

    return None


def _has_destructive_intent(question: str) -> bool:
    """Devuelve True si la pregunta expresa intención de modificar/eliminar datos."""
    return bool(_DESTRUCTIVE_INTENT_RE.search(question))


# ── VALIDACIÓN Y CORRECCIÓN DE SQL ───────────────────────────────────────────

_JOIN_ON_RE = re.compile(
    r'\bON\s+(?:\w+\.)?(\w+)\s*=\s*(?:\w+\.)?(\w+)',
    re.IGNORECASE,
)

# Para validación FK-aware: extrae alias→tabla y condiciones JOIN ON completas
_TABLE_ALIAS_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)\s+(\w+)\b',
    re.IGNORECASE,
)
_JOIN_ON_FULL_RE = re.compile(
    r'\bON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)',
    re.IGNORECASE,
)
_CTE_DEF_RE = re.compile(r'\b(\w+)\s+AS\s*\(', re.IGNORECASE)
_SQL_KW_SET = frozenset({
    'on', 'where', 'set', 'from', 'join', 'left', 'right', 'inner',
    'outer', 'full', 'cross', 'as', 'and', 'or', 'not', 'in', 'is',
    'null', 'like', 'between', 'exists', 'all', 'any', 'select',
    'order', 'group', 'by', 'having', 'union', 'with', 'insert',
    'update', 'delete', 'into', 'values', 'case', 'when', 'then',
    'else', 'end', 'top', 'distinct', 'over', 'partition',
})


def _keys_compatible(k1: str, k2: str) -> bool:
    """Dos *Key columns son compatibles si uno es sufijo del otro (ej. DateKey / OrderDateKey)."""
    a, b = k1.lower(), k2.lower()
    return a == b or a.endswith(b) or b.endswith(a)


def _validate_sql_joins(sql: str) -> list[str]:
    """Detecta JOINs semánticamente imposibles entre columnas *Key de tipos distintos."""
    errors = []
    for m in _JOIN_ON_RE.finditer(sql):
        left, right = m.group(1), m.group(2)
        if left.lower().endswith("key") and right.lower().endswith("key"):
            if not _keys_compatible(left, right):
                errors.append(
                    f"JOIN con columnas incompatibles: {left} = {right}"
                )

    # Detecta uso de alias DD. sin JOIN a DimDate (error frecuente del LLM)
    if re.search(r'\bDD\.', sql, re.IGNORECASE):
        if not re.search(r'\bDimDate\b', sql, re.IGNORECASE):
            errors.append(
                "SQL usa DD. (alias de DimDate) pero falta JOIN dbo.DimDate DD ON FIS.OrderDateKey = DD.DateKey"
            )

    # Validación FK-aware: detecta columnas *Key que no existen en la tabla referenciada
    if _FK_COL_SET:
        # Nombres de CTEs definidos en este SQL (para no validarlos como tablas reales)
        cte_names = {m.group(1).lower() for m in _CTE_DEF_RE.finditer(sql)}

        # Mapeo alias → nombre de tabla real
        alias_to_table: dict[str, str] = {}
        for m in _TABLE_ALIAS_RE.finditer(sql):
            table_name, alias = m.group(1), m.group(2)
            if alias.lower() not in _SQL_KW_SET:
                alias_to_table[alias.lower()] = table_name.lower()

        # Revisar cada JOIN ON alias1.col1 = alias2.col2
        for m in _JOIN_ON_FULL_RE.finditer(sql):
            alias1, col1, alias2, col2 = m.group(1), m.group(2), m.group(3), m.group(4)
            for alias, col in [(alias1, col1), (alias2, col2)]:
                if not col.lower().endswith('key'):
                    continue  # solo validar columnas *Key

                if alias.lower() in cte_names:
                    continue  # el alias es un CTE, no una tabla real

                table = alias_to_table.get(alias.lower())
                if not table:
                    continue  # alias no reconocido (probablemente CTE sin alias explícito)

                if table in cte_names:
                    continue  # el alias referencia a un CTE

                if (table, col.lower()) not in _FK_COL_SET:
                    # Buscar en qué tablas sí existe esa columna
                    tables_with_col = sorted(
                        t for (t, c) in _FK_COL_SET if c == col.lower() and t != table
                    )
                    hint = (
                        f" — la columna sí existe en: {', '.join(tables_with_col)}"
                        if tables_with_col else ""
                    )
                    errors.append(
                        f"La columna {col} no existe en {table}{hint}. "
                        f"Usa la tabla correcta para este JOIN."
                    )

    return errors


async def _get_corrected_sql(question: str, bad_sql: str, problem: str) -> Optional[str]:
    """Llama al LLM con el SQL problemático y el error para obtener una versión corregida."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None

    schema_context = ""
    if SCHEMA_STORE:
        try:
            hits = SCHEMA_STORE.query(question, k=6)
            schema_context = "\n".join(h.get("doc", "") for h in hits if h.get("doc"))
        except Exception:
            pass

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"SQL generado con error:\n{bad_sql}\n\n"
        f"Problema detectado: {problem}\n\n"
        f"Esquema disponible:\n{schema_context}\n\n"
        "REGLAS CRÍTICAS:\n"
        "- Si la pregunta menciona un nombre de VENDEDOR/EMPLEADO (quien hace la venta): "
        "usa DimEmployee DE + FactResellerSales FRS ON FRS.EmployeeKey = DE.EmployeeKey. "
        "NUNCA busques un vendedor en DimCustomer.\n"
        "- Si la pregunta menciona un nombre de CLIENTE (quien compra): "
        "usa DimCustomer DC + FactInternetSales FIS ON FIS.CustomerKey = DC.CustomerKey. "
        "NUNCA busques un cliente en DimEmployee.\n"
        "- NUNCA mezcles DimEmployee con DimCustomer en la misma consulta para una persona.\n\n"
        "Escribe ÚNICAMENTE el SQL corregido, sin explicaciones ni bloques markdown."
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            raw = response.json()["choices"][0]["message"]["content"].strip()
            # Eliminar bloques de código markdown si el LLM los incluyó
            raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            return raw.strip() or None
    except Exception:
        logger.exception("Error obteniendo SQL corregido del LLM")
        return None


def _validate_growth_structure(sql: str) -> list[str]:
    """Detecta LAG/LEAD cuyo OVER ordena por columna no-temporal (produce crecimiento=0).

    El error clásico: el CTE fuente agrega todos los años en una sola fila por
    (territorio, categoría), luego LAG ordena por ese string en vez de por año.
    Cada partición tiene exactamente una fila → LAG devuelve NULL → crecimiento = 0.
    """
    if not _WINDOW_REQUIRES_ORDER_BY.search(sql):
        return []

    errors: list[str] = []

    def _check(m: re.Match) -> str:
        content = m.group(1)
        order_m = re.search(r'\bORDER\s+BY\s+(\w+)', content, re.IGNORECASE)
        if not order_m:
            return m.group(0)  # sin ORDER BY → _fix_window_order_by ya lo maneja

        # Solo actuar si la función que precede a este OVER es LAG/LEAD/etc.
        preceding = sql[max(0, m.start() - 300): m.start()]
        if not _WINDOW_REQUIRES_ORDER_BY.search(preceding):
            return m.group(0)

        order_col = order_m.group(1)
        if not _DATE_LIKE_COL.match(order_col):
            errors.append(
                f"LAG/LEAD tiene ORDER BY '{order_col}' que no es una columna temporal. "
                "Para calcular crecimiento por período el ORDER BY del OVER debe ser una "
                "columna de año/fecha (CalendarYear, OrderDateKey, etc.) y el CTE fuente "
                "debe incluir esa columna en su GROUP BY. "
                "Reescribe la consulta agrupando por año, territorio y categoría antes de aplicar LAG."
            )
        return m.group(0)

    re.sub(r'\bOVER\s*\(([^()]*)\)', _check, sql, flags=re.IGNORECASE)
    return errors


_YEAR_IN_QUESTION_RE = re.compile(r'\b(20\d{2}|19\d{2})\b')
_YEAR_FILTER_IN_SQL_RE = re.compile(
    r'\b(CalendarYear|OrderDateKey|ShipDateKey|DueDateKey|YEAR\s*\(|DimDate)\b',
    re.IGNORECASE,
)


def _validate_temporal_filter(sql: str, question: str) -> list[str]:
    """Detecta cuando la pregunta menciona un año pero el SQL no filtra por él."""
    years = _YEAR_IN_QUESTION_RE.findall(question)
    if not years:
        return []
    if _YEAR_FILTER_IN_SQL_RE.search(sql):
        return []
    year = years[0]
    return [
        f"La pregunta menciona el año {year} pero el SQL no tiene filtro temporal. "
        f"Añade un JOIN a dbo.DimDate DD ON <FactTable>.OrderDateKey = DD.DateKey "
        f"y filtra con WHERE DD.CalendarYear = {year}. "
        "Sin este filtro la consulta suma todos los años y devuelve resultados incorrectos."
    ]


# ── FIN VALIDACIÓN Y CORRECCIÓN ───────────────────────────────────────────────


class TrackingSqlTool(RunSqlTool):
    """Instancia única compartida; el estado por-request vive en ContextVars."""

    def __init__(self, sql_runner):
        super().__init__(sql_runner=sql_runner, file_system=NullFileSystem())

    async def execute(self, context, args):
        sql = getattr(args, "sql", None)
        if sql:
            safety_error = _check_sql_safety(sql)
            if safety_error:
                raise RuntimeError(safety_error)

            sanitized = _sanitize_sql_aliases(sql)
            if sanitized != sql:
                logger.info("SQL sanitizado — alias reservados reemplazados")
                args = RunSqlToolArgs(sql=sanitized)
                sql = sanitized

            join_errors = _validate_sql_joins(sql)
            current_question = _ctx_sql_question.get()
            if join_errors and current_question:
                problem = "; ".join(join_errors)
                logger.warning("JOIN inválido detectado: %s — solicitando corrección al LLM", problem)
                corrected = await _get_corrected_sql(current_question, sql, problem)
                if corrected and corrected != sql:
                    logger.info("SQL corregido por LLM antes de ejecutar")
                    args = RunSqlToolArgs(sql=corrected)
                    sql = corrected

            growth_errors = _validate_growth_structure(sql)
            if growth_errors and current_question and not join_errors:
                problem = "; ".join(growth_errors)
                logger.warning("Estructura de crecimiento inválida: %s — solicitando corrección al LLM", problem)
                corrected = await _get_corrected_sql(current_question, sql, problem)
                if corrected and corrected != sql:
                    logger.info("SQL de crecimiento corregido por LLM antes de ejecutar")
                    args = RunSqlToolArgs(sql=corrected)
                    sql = corrected

            temporal_errors = _validate_temporal_filter(sql, current_question or "")
            if temporal_errors and current_question and not join_errors:
                problem = "; ".join(temporal_errors)
                logger.warning("Filtro temporal ausente: %s — solicitando corrección al LLM", problem)
                corrected = await _get_corrected_sql(current_question, sql, problem)
                if corrected and corrected != sql:
                    logger.info("SQL corregido con filtro temporal por LLM antes de ejecutar")
                    _evict_sql_cache(current_question)
                    args = RunSqlToolArgs(sql=corrected)
                    sql = corrected

        sql_key = (sql or "").strip()
        in_flight: set[str] | None = _ctx_sql_inflight.get()
        if in_flight is None:
            in_flight = set()
            _ctx_sql_inflight.set(in_flight)

        if sql_key and sql_key in in_flight:
            logger.warning("TrackingSqlTool — ejecución duplicada bloqueada")
            return None
        if sql_key:
            in_flight.add(sql_key)

        logger.info("TrackingSqlTool.execute() — SQL: %s", sql)
        result = await super().execute(context, args)
        if sql:
            try:
                cb = _ctx_sql_callback.get()
                if cb is not None:
                    cb(sql)
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
        "- NÚMEROS: usa SIEMPRE coma como separador de miles y punto como decimal: 16,351,550.34 (NO 16.351.550,34).\n"
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
            return _fix_eu_numbers(data["choices"][0]["message"]["content"].strip())
    except Exception:
        logger.exception("Error generando análisis fresco")
        return ""


async def _stream_analysis(question: str, data_rows: list, columns: list):
    """Genera análisis en streaming — yields tokens a medida que llegan del LLM."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return

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
        "- NÚMEROS: usa SIEMPRE coma como separador de miles y punto como decimal: 16,351,550.34.\n"
        "- Sin markdown, bullets ni encabezados. Solo texto narrativo directo."
    )

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 200,
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
        logger.exception("Error en stream de análisis")


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
            # Si quedaron placeholders sin reemplazar (ej: __YEAR2__ cuando la
            # pregunta actual solo tiene 1 año), el SQL sería inválido → ignorar cache
            if re.search(r'__\w+__', sql):
                logger.info("Cache HIT ignorado: SQL tiene placeholders sin resolver tras inyección")
                return None
            logger.info("Cache HIT con temporales sustituidos: %s", temporals)
        else:
            sql = sql_template

        # Validar que el valor de distance es usable
        if not isinstance(distance, (int, float)):
            return None

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


def _evict_sql_cache(question: str):
    """Elimina del cache las entradas asociadas a una pregunta (útil tras corrección de SQL)."""
    if not SQL_CACHE:
        return
    try:
        normalized = _normalize_question(question)
        results = SQL_CACHE.query(query_texts=[normalized], n_results=3)
        ids = (results.get("ids") or [[]])[0]
        if ids:
            SQL_CACHE.delete(ids=ids)
            logger.info("Cache purgado para '%s' (%d entradas eliminadas)", normalized, len(ids))
    except Exception:
        logger.exception("Error purgando cache")


# Helpers de texto

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


def _is_percent_col(col: str) -> bool:
    words = set(re.sub(r'([A-Z])', r' \1', col).lower().split())
    return bool(words & _PERCENT_KEYWORDS)


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

# ── PREDICCIÓN / PROYECCIÓN ───────────────────────────────────────────────────

# Años futuros respecto al dataset (AdventureWorks termina en 2014)
_FUTURE_YEAR_RE = re.compile(r'\b(201[5-9]|20[2-9]\d)\b')


_PREDICTION_SQL = """
SELECT DD.CalendarYear, SUM(S.SalesAmount) AS TotalSales
FROM (
    SELECT OrderDateKey, SalesAmount FROM dbo.FactInternetSales
    UNION ALL
    SELECT OrderDateKey, SalesAmount FROM dbo.FactResellerSales
) S
JOIN dbo.DimDate DD ON S.OrderDateKey = DD.DateKey
GROUP BY DD.CalendarYear
ORDER BY DD.CalendarYear
"""


async def _run_prediction(question: str) -> tuple[str, dict]:
    """Proyecta ventas futuras con regresión lineal sobre datos históricos.
    Retorna (narrativa, chart_payload)."""
    if not SQL_RUNNER:
        return "No hay conexión a la base de datos disponible.", {}

    year_match = _FUTURE_YEAR_RE.search(question)

    try:
        import numpy as np

        df = await SQL_RUNNER.run_sql(RunSqlToolArgs(sql=_PREDICTION_SQL), None)

        if df.empty or len(df) < 2:
            return "No hay suficientes datos históricos para proyectar.", {}

        df = df.dropna(subset=["CalendarYear", "TotalSales"])
        if len(df) < 2:
            return "No hay suficientes datos históricos para proyectar.", {}

        years = df["CalendarYear"].astype(int).tolist()
        sales = df["TotalSales"].astype(float).tolist()

        # Excluir el último año si parece incompleto (< 10% del año anterior)
        if len(sales) >= 2 and sales[-2] > 0 and (sales[-1] / sales[-2]) < 0.10:
            logger.warning(
                "Año %d parece incompleto ($%s vs $%s) — excluido de la proyección",
                years[-1], f"{sales[-1]:,.0f}", f"{sales[-2]:,.0f}",
            )
            years = years[:-1]
            sales = sales[:-1]

        if len(years) < 2:
            return "No hay suficientes años de datos para calcular una proyección confiable.", {}

        target_year = int(year_match.group()) if year_match else max(years) + 1

        x = np.array(years, dtype=float)
        y = np.array(sales, dtype=float)

        # Regresión lineal
        slope, intercept = np.polyfit(x, y, 1)
        projected_linear = float(slope * target_year + intercept)

        # Tasa de crecimiento mediana año-a-año (más robusta que CAGR extremo-a-extremo)
        yoy_rates = [
            (y[i] - y[i - 1]) / y[i - 1]
            for i in range(1, len(y))
            if y[i - 1] > 0
        ]
        median_yoy = float(np.median(yoy_rates)) if yoy_rates else 0.0
        # Limitar tasa a rango razonable para evitar proyecciones absurdas
        median_yoy = max(-0.9, min(median_yoy, 10.0))
        years_ahead = max(1, target_year - int(max(years)))
        projected_yoy = float(y[-1] * (1 + median_yoy) ** years_ahead)
        growth_pct = median_yoy * 100

        projected = (projected_linear + projected_yoy) / 2.0

        api_key = os.getenv("OPENAI_API_KEY")
        model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        narrative = ""
        if api_key:
            rows_txt = "\n".join(f"{yr}: ${sl:,.0f}" for yr, sl in zip(years, sales))
            prompt = (
                f"Datos de ventas históricas (todas las fuentes combinadas):\n{rows_txt}\n\n"
                f"Proyección para {target_year}:\n"
                f"  - Regresión lineal: ${projected_linear:,.0f}\n"
                f"  - Crecimiento mediano anual ({growth_pct:.1f}%): ${projected_yoy:,.0f}\n"
                f"  - Estimación promedio: ${projected:,.0f}\n\n"
                f"Pregunta del usuario: {question}\n\n"
                "Escribe UN párrafo en español (3-4 oraciones) respondiendo la pregunta con los datos de arriba. "
                "Si la pregunta es sobre ventas totales: menciona el crecimiento mediano anual, los años base "
                "y la estimación final. "
                "Si la pregunta es sobre un producto, región u otra dimensión específica: explica amablemente "
                "que esta proyección es para ventas totales y que no es posible predecir con precisión qué "
                "producto o dimensión específica liderará en el futuro, pero sí se puede ver cuál lideró "
                "históricamente. "
                "NÚMEROS: usa coma para miles y punto para decimal (1,234,567.89). Sin markdown."
            )
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": model,
                            "max_tokens": 250,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    narrative = resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                logger.exception("Error generando narrativa de proyección")

        if not narrative:
            narrative = (
                f"Con base en las ventas históricas de {years[0]} a {years[-1]} "
                f"(CAGR del {growth_pct:.1f}% anual), se proyecta que las ventas totales "
                f"para {target_year} alcanzarían aproximadamente ${projected:,.0f}. "
                "Esta es una proyección estadística basada en tendencias pasadas."
            )

        # Gráfico: barras históricas (azul) + barra de proyección (naranja)
        chart_years = [str(yr) for yr in years] + [str(target_year)]
        hist_data   = [round(s, 2) for s in sales] + [None]
        proj_data   = [None] * len(years) + [round(projected, 2)]

        chart_payload = {
            "backgroundColor": "transparent",
            "animation": True,
            "animationDuration": 900,
            "animationEasing": "cubicOut",
            "grid": {"left": "3%", "right": "4%", "bottom": "18%", "top": "8%", "containLabel": True},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"data": ["Histórico", f"Proyección {target_year}"], "bottom": 0, "textStyle": {"color": "#333"}},
            "xAxis": {
                "type": "category", "data": chart_years,
                "axisLabel": {"fontSize": 11, "color": "#333"},
                "axisLine": {"lineStyle": {"color": "#ccc"}},
            },
            "yAxis": {"type": "value", "axisLabel": {"fontSize": 11, "color": "#333"}},
            "series": [
                {
                    "name": "Histórico", "type": "bar", "data": hist_data,
                    "barMaxWidth": 52,
                    "itemStyle": {
                        "borderRadius": [5, 5, 0, 0],
                        "color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                                  "colorStops": [{"offset": 0, "color": "#4A90D9"},
                                                 {"offset": 1, "color": "#1a5276"}]},
                    },
                },
                {
                    "name": f"Proyección {target_year}", "type": "bar", "data": proj_data,
                    "barMaxWidth": 52,
                    "itemStyle": {
                        "borderRadius": [5, 5, 0, 0],
                        "color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                                  "colorStops": [{"offset": 0, "color": "#F39C12"},
                                                 {"offset": 1, "color": "#9a7d0a"}]},
                    },
                    "emphasis": {"itemStyle": {"opacity": 0.82}},
                },
            ],
        }

        logger.info(
            "Proyección para %d: $%s (lineal: $%s, mediana YoY: $%s)",
            target_year,
            f"{projected:,.0f}", f"{projected_linear:,.0f}", f"{projected_yoy:,.0f}",
        )
        return narrative, chart_payload

    except Exception:
        logger.exception("Error ejecutando proyección")
        return "No se pudo calcular la proyección. Intenta de nuevo.", {}


DISCOVERY_KEYWORDS = [
    "qué tablas", "que tablas", "tablas disponibles", "vistas disponibles",
    "tablas y vistas", "lista de tablas", "muestra las tablas", "muéstrame las tablas",
    "cuáles tablas", "cuales tablas", "todas las tablas", "show tables",
    "esquema disponible", "qué esquema", "que esquema",
]


def _is_discovery_question(message: str) -> bool:
    return any(kw in message.lower() for kw in DISCOVERY_KEYWORDS)


# ── CLASIFICADOR DE INTENCIÓN ─────────────────────────────────────────────────
# Determina si el mensaje requiere SQL o es conversación general.
# Usa el LLM con max_tokens=5 para ser robusto ante cualquier formulación.

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
    if not SQL_RUNNER:
        return "", "No hay conexión a la base de datos disponible."
    try:
        df = await SQL_RUNNER.run_sql(RunSqlToolArgs(sql=_DISCOVERY_SQL), None)
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


async def _classify_intent(message: str) -> str:
    """Retorna 'SQL', 'PREDICTION', 'DISCOVERY' o 'CHAT' según el tipo de mensaje."""
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
                                "Clasifica el mensaje del usuario en UNA de estas cuatro categorías:\n"
                                "DISCOVERY — el usuario quiere ver qué tablas existen en la base de datos, "
                                "cuántos registros tiene cada tabla, qué contiene la BD, qué datos hay disponibles, "
                                "la estructura o el contenido general de la base de datos.\n"
                                "SQL — el usuario quiere datos históricos reales: ventas pasadas, métricas, "
                                "rankings, comparaciones, productos, clientes, empleados, territorios, "
                                "tendencias pasadas, gráficos de datos existentes.\n"
                                "PREDICTION — el usuario pregunta por ventas TOTALES GLOBALES en el futuro "
                                "SIN filtrar por país, región, producto, categoría ni ninguna dimensión: "
                                "'¿cuánto se venderá en total en 2015?', '¿cuáles serán las ventas totales?'. "
                                "Si menciona un país (Australia, Francia, USA...), región, producto, empleado "
                                "o cualquier categoría específica → SQL, NO PREDICTION. "
                                "Si pregunta qué producto/región/empleado liderará en el futuro → SQL.\n"
                                "CHAT — el usuario saluda, agradece, se despide, pregunta qué eres, "
                                "qué puedes hacer, cómo funcionas, o hace preguntas fuera del dominio de datos.\n"
                                "Responde ÚNICAMENTE con la palabra SQL, PREDICTION, DISCOVERY o CHAT."
                            ),
                        },
                        {"role": "user", "content": message},
                    ],
                },
            )
            raw = resp.json()["choices"][0]["message"]["content"].strip().upper()
            # Extraer primera palabra — el LLM a veces añade contexto extra
            first_word = raw.split()[0] if raw.split() else ""
            if first_word in ("DISCOVERY", "SQL", "PREDICTION", "CHAT"):
                intent = first_word
            elif "DISCOVERY" in raw:
                intent = "DISCOVERY"
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


def _filter_and_deduplicate(hits: list) -> list:
    sorted_hits = sorted(hits, key=lambda h: h.get("distance", 999))

    # Limitar join_path/join_chain a máximo 3 para que los docs de esquema también aparezcan
    result: list = []
    join_doc_count = 0
    MAX_JOIN_DOCS = 3
    for h in sorted_hits:
        if len(result) >= RAG_K_FINAL:
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

    return (
        "Eres un experto en SQL Server y análisis de negocio. Reglas:\n"
        "1. Usa SOLO tablas y columnas del esquema dado.\n"
        "2. Llama a run_sql EXACTAMENTE UNA VEZ. ABSOLUTAMENTE PROHIBIDO ejecutar run_sql más de una vez.\n"
        "   Si necesitas múltiples datos, combínalos en UNA sola query con CTEs o subconsultas.\n"
        "   Si la query falla NO reintentes — reporta el error exacto al usuario.\n"
        "   Para filtrar por año en DimDate usa SIEMPRE: JOIN dbo.DimDate DD ON FIS.OrderDateKey = DD.DateKey — luego WHERE DD.CalendarYear IN (...)\n"
        "3. Para limitar filas usa TOP N al inicio del SELECT principal: SELECT TOP N ...\n"
        "   ORDER BY va SIEMPRE en el SELECT final, NUNCA dentro de una CTE (SQL Server devuelve error).\n"
        "   NUNCA uses FETCH FIRST N ROWS ONLY — SQL Server solo soporta SELECT TOP N.\n"
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
        "- Para 'último año disponible' → usa: (SELECT MAX(CalendarYear) FROM dbo.DimDate)\n"
        "- CRÍTICO: Si el usuario menciona un año futuro (ej. 2015, 2016) que puede no existir "
        "en los datos, NUNCA filtres directamente por ese año. Usa siempre el último año disponible:\n"
        "    WHERE DD.CalendarYear = (SELECT MAX(CalendarYear) FROM dbo.DimDate)\n"
        "  y menciona en tu respuesta que estás usando el último año con datos disponibles.\n\n"
        "FUENTE DE VENTAS (CRÍTICO — regla absoluta):\n"
        "Las ventas totales = FactInternetSales + FactResellerSales combinadas.\n"
        "SIEMPRE incluye en el UNION ALL TODAS las columnas que necesitarás para JOINs posteriores.\n"
        "REGLA CRÍTICA: si la pregunta filtra por territorio, producto o cliente, SalesTerritoryKey / ProductKey "
        "/ CustomerKey DEBEN estar en AllSales — sin ellos el JOIN posterior es un cross-join silencioso.\n"
        "  WITH AllSales AS (\n"
        "      SELECT OrderDateKey, SalesTerritoryKey, ProductKey,\n"
        "             SalesAmount, TotalProductCost, UnitPrice\n"
        "      FROM dbo.FactInternetSales\n"
        "      UNION ALL\n"
        "      SELECT OrderDateKey, SalesTerritoryKey, ProductKey,\n"
        "             SalesAmount, TotalProductCost, UnitPrice\n"
        "      FROM dbo.FactResellerSales\n"
        "  )\n"
        "JOIN CORRECTO (usa la clave de AllSales, no de la dimensión):\n"
        "  JOIN dbo.DimSalesTerritory DST ON AllSales.SalesTerritoryKey = DST.SalesTerritoryKey  ← CORRECTO\n"
        "  JOIN dbo.DimSalesTerritory DST ON DST.SalesTerritoryKey IN (subquery)                 ← INCORRECTO\n"
        "  JOIN dbo.DimSalesTerritory DST ON AllSales.OrderDateKey = DST.SalesTerritoryKey       ← INCORRECTO\n"
        "Si la pregunta pide totales globales sin desglose por canal, omite la columna Canal y agrupa directamente.\n"
        "Si la pregunta pide desglose 'por canal', incluye Canal en SELECT y GROUP BY.\n"
        "NUNCA intentes inferir el canal desde valores de columnas después del UNION ALL.\n"
        "CRÍTICO — GROUP BY en AllSales: cuando combinas FIS + FRS en AllSales y haces JOIN a una dimensión,\n"
        "el GROUP BY debe ser SOLO por la columna de la dimensión (ej: SalesTerritoryRegion).\n"
        "NUNCA incluyas columnas de AllSales en el GROUP BY final salvo que el usuario pida desglose por canal.\n"
        "Ejemplo CORRECTO (porcentaje por región):\n"
        "  SELECT DST.SalesTerritoryRegion,\n"
        "         SUM(AllSales.SalesAmount) * 100.0 / SUM(SUM(AllSales.SalesAmount)) OVER() AS Percentage\n"
        "  FROM AllSales\n"
        "  JOIN dbo.DimSalesTerritory DST ON AllSales.SalesTerritoryKey = DST.SalesTerritoryKey\n"
        "  GROUP BY DST.SalesTerritoryRegion  ← SOLO la dimensión\n"
        "  ORDER BY Percentage DESC\n"
        "CRÍTICO: NUNCA uses 'AS' como alias de tabla (es palabra reservada). Referencia AllSales directamente.\n"
        "Ejemplo INCORRECTO: GROUP BY DST.SalesTerritoryRegion, AllSales.SalesTerritoryKey ← produce filas duplicadas\n\n"
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
        "NOMBRES DE PERSONAS (CRÍTICO — error frecuente):\n"
        "Cuando la pregunta menciona un nombre de persona, determina si es VENDEDOR o CLIENTE:\n"
        "  VENDEDOR (ej: 'ventas de Linda Mitchell', 'cuánto vendió', 'mejor vendedor', 'rendimiento de'):\n"
        "    → DimEmployee DE + FactResellerSales FRS ON FRS.EmployeeKey = DE.EmployeeKey\n"
        "    → filtra: WHERE DE.FirstName = 'Linda' AND DE.LastName = 'Mitchell'\n"
        "  CLIENTE (ej: 'compras de', 'pedidos de', 'cuánto gastó', 'historial de compra'):\n"
        "    → DimCustomer DC + FactInternetSales FIS ON FIS.CustomerKey = DC.CustomerKey\n"
        "    → filtra: WHERE DC.FirstName = 'Linda' AND DC.LastName = 'Mitchell'\n"
        "NUNCA cruces DimEmployee con DimCustomer ni mezcles FactResellerSales con DimCustomer "
        "para buscar a una persona — son tablas independientes.\n\n"
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
        "- NUNCA uses formato | col | col | (markdown pipe).\n"
        "- NÚMEROS: usa SIEMPRE coma como separador de miles y punto como decimal: 16,351,550.34 (NO 16.351.550,34).\n\n"
        "ESQUEMA:\n"
        f"{schema_lines}\n\n"
        f"PREGUNTA: {message}\n\n"
        "INSTRUCCIÓN FINAL (obligatoria): Después de mostrar los datos, escribe SIEMPRE un párrafo "
        "en español (2-4 oraciones) analizando los resultados. Menciona valores específicos, "
        "tendencias o el dato más destacado. Este párrafo va DESPUÉS de la tabla, nunca antes."
    )


_PINNED_TABLES = {"dbo.FactInternetSales", "dbo.FactResellerSales"}


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

            # Pinear siempre las tablas Fact fundamentales
            hit_tables = {
                f"{h.get('meta', {}).get('schema', 'dbo')}.{h.get('meta', {}).get('table', '')}"
                for h in hits
                if h.get('meta', {}).get('type') not in ('join_path', 'join_chain')
            }
            for pinned in _PINNED_TABLES:
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
    retry: bool = False,
) -> AsyncGenerator[str, None]:

    original_question = message

    # 0. Bloquear intención destructiva antes de cualquier procesamiento
    if _has_destructive_intent(original_question):
        logger.warning("Intención destructiva detectada y bloqueada: %.100s", original_question)
        _ctx_intent.set("CHAT")
        yield _READONLY_REFUSAL
        return

    # 0. Clasificar intención — enruta a SQL, proyección, discovery o conversación
    if not retry:
        intent = await _classify_intent(original_question)
        _ctx_intent.set(intent)
        if intent == "CHAT":
            yield await _get_chat_response(original_question)
            return
        if intent == "PREDICTION":
            logger.info("Pregunta predictiva detectada — ejecutando proyección estadística")
            narrative, chart_payload = await _run_prediction(original_question)
            if narrative:
                yield narrative
            if chart_payload:
                yield CHART_SENTINEL + json.dumps(chart_payload)
            return
        if intent == "DISCOVERY":
            logger.info("Discovery detectado — consultando tablas y registros de la BD")
            table_html, summary = await _run_discovery(original_question)
            if table_html:
                yield table_html
            if summary:
                yield summary
            return

    # En reintento forzamos SQL fresco evictando la entrada cacheada
    if retry:
        _evict_sql_cache(original_question)
        logger.info("Reintento — cache evictado para '%s'", original_question)

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
                if '<table' in full_response.lower() and '</table>' in full_response.lower():
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
            temporal_errors = _validate_temporal_filter(cached_sql, original_question)
            if temporal_errors:
                logger.warning(
                    "Cache HIT descartado — SQL sin filtro temporal: %s", temporal_errors[0]
                )
                _evict_sql_cache(original_question)
                cached_sql = None

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
                    async for token in _stream_no_results(original_question):
                        yield token
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

    _ctx_sql_callback.set(on_sql_executed)
    _ctx_sql_question.set(original_question)
    _ctx_sql_inflight.set(set())

    # 4. Ejecutar agente
    pre_table_buffer: list[str] = []
    tabla_chunks: list[str] = []
    post_table_chunks: list[str] = []
    table_seen = False
    meaningful_post_analysis = False  # True cuando se emitió análisis sustancial (>20 chars)
    response_chunks: list[str] = []

    async def _step_agent(conv_id: Optional[str]):
        """Ejecuta el agente llenando los buffers. Lanza excepción si falla."""
        nonlocal table_seen, meaningful_post_analysis
        agen = agent.send_message(
            request_context=request_context,
            message=message,
            conversation_id=conv_id,
        )
        try:
            while True:
                try:
                    component = await asyncio.wait_for(agen.__anext__(), timeout=AGENT_STEP_TIMEOUT)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning("Agent timeout — paso excedió %.0f s", AGENT_STEP_TIMEOUT)
                    raise TimeoutError("timeout")

                text_output = extract_text_from_component(component)
                if not text_output:
                    continue

                if '<table' in text_output:
                    table_seen = True
                    tabla_chunks.append(text_output)
                elif table_seen:
                    if not _is_vague_analysis(text_output):
                        post_table_chunks.append(text_output)
                        response_chunks.append(text_output)
                        yield text_output
                        if len(text_output.strip()) > 20:
                            meaningful_post_analysis = True
                else:
                    pre_table_buffer.append(text_output)
        finally:
            await agen.aclose()

    try:
        async for chunk in _step_agent(conversation_id):
            yield chunk
    except TimeoutError:
        yield ERROR_RETRY_SENTINEL + "El agente tardó demasiado en responder."
        return
    except Exception as exc:
        # Historial corrupto por tool_call sin respuesta → limpiar y reintentar una vez
        if "tool_call" in str(exc).lower() and conversation_id and not retry:
            logger.warning(
                "Historial corrupto (tool_calls sin respuesta) — limpiando conv %s y reintentando",
                conversation_id,
            )
            try:
                await agent.conversation_store.delete_conversation(conversation_id)
            except Exception:
                pass
            try:
                async for chunk in _step_agent(conversation_id):
                    yield chunk
            except Exception:
                logger.exception("Error en reintento tras limpiar conversación")
                yield ERROR_RETRY_SENTINEL + "Ocurrió un error procesando tu pregunta."
                return
        else:
            logger.exception("Error en agent.send_message")
            yield ERROR_RETRY_SENTINEL + "Ocurrió un error procesando tu pregunta."
            return

    # 5. Emitir respuesta
    logger.info(
        "Buffers — pre_table:%d tabla:%d post_table:%d meaningful=%s",
        len(pre_table_buffer), len(tabla_chunks), len(post_table_chunks), meaningful_post_analysis,
    )
    if not post_table_chunks and table_seen:
        logger.warning("Tabla encontrada pero post_table_chunks está vacío — el LLM no generó análisis posterior")

    if table_seen and tabla_chunks:
        merged_table = _merge_multiple_tables(tabla_chunks) if len(tabla_chunks) > 1 else tabla_chunks[0]
        response_chunks.append(merged_table)

        # Si no viene gráfico, la tabla debe aparecer ANTES del análisis
        is_chart = _is_chart_question(original_question)
        if not is_chart:
            yield TABLE_FLUSH_SENTINEL  # señal para app.py: emitir tabla ya
        yield merged_table

        if meaningful_post_analysis:
            # Análisis sustancial ya emitido en el loop; solo registrar para cache
            pass
        elif captured_sql and SQL_RUNNER:
            # Sin análisis sustancial (vacío, trivial o vago): streaming fallback
            if post_table_chunks:
                logger.warning("Análisis del LLM trivial/vago — reemplazando con fallback streaming")
            best_sql = captured_sql[-1]
            try:
                df = await SQL_RUNNER.run_sql(RunSqlToolArgs(sql=best_sql), None)
                if not df.empty:
                    cols = df.columns.tolist()
                    rows = df.to_dict("records")
                    analysis_text = ""
                    async for token in _stream_analysis(original_question, rows, cols):
                        analysis_text += token
                        yield token
                    if analysis_text:
                        response_chunks.append(analysis_text)
                        logger.info("Análisis de respaldo (streaming) generado correctamente")
            except Exception:
                logger.exception("Error generando análisis de respaldo")
    elif captured_sql and SQL_RUNNER:
        # El stream produjo output vacío pero el SQL fue ejecutado — fallback completo
        logger.warning("Stream vacío con SQL capturado — activando fallback de renderizado")
        df = None
        last_sql_error: str = ""
        for sql_attempt in reversed(captured_sql):
            try:
                tool_args = RunSqlToolArgs(sql=sql_attempt)
                df = await SQL_RUNNER.run_sql(tool_args, None)
                logger.info("Fallback SQL exitoso (intento con SQL más reciente)")
                break
            except Exception as e:
                last_sql_error = str(e)
                logger.warning("Fallback SQL falló, probando anterior: %s", e)

        # Si todos los intentos fallaron con error SQL, pedir corrección al LLM
        if df is None and last_sql_error and captured_sql:
            logger.warning("SQL error — solicitando corrección LLM: %s", last_sql_error[:200])
            corrected = await _get_corrected_sql(
                original_question,
                captured_sql[-1],
                f"Error SQL de SQL Server: {last_sql_error[:400]}",
            )
            if corrected:
                try:
                    df = await SQL_RUNNER.run_sql(RunSqlToolArgs(sql=corrected), None)
                    captured_sql[-1] = corrected
                    _evict_sql_cache(original_question)
                    logger.info("Corrección LLM exitosa tras error SQL (%d filas)", len(df))
                except Exception as e:
                    logger.warning("SQL corregido también falló: %s", e)

        if df is None:
            yield ERROR_RETRY_SENTINEL + "Ocurrió un error al procesar los resultados."
            return
        try:
            if df.empty:
                logger.warning("Fallback devolvió 0 filas — intentando corrección LLM")
                corrected_sql = await _get_corrected_sql(
                    original_question,
                    captured_sql[-1],
                    "La consulta devolvió 0 resultados — revisa los JOINs y filtros",
                )
                if corrected_sql:
                    try:
                        df = await SQL_RUNNER.run_sql(RunSqlToolArgs(sql=corrected_sql), None)
                        if not df.empty:
                            captured_sql[-1] = corrected_sql
                            _evict_sql_cache(original_question)
                            logger.info("Retry exitoso tras corrección LLM (0 filas → %d filas)", len(df))
                    except Exception as e:
                        logger.warning("Retry SQL también falló: %s", e)

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
                    if not _is_chart_question(original_question):
                        yield TABLE_FLUSH_SENTINEL
                    yield table_html

                analysis_text = ""
                async for token in _stream_analysis(original_question, rows, cols):
                    analysis_text += token
                    yield token
                if analysis_text:
                    response_chunks.append(analysis_text)
                    logger.info("Fallback completo (streaming) emitido correctamente")
            else:
                async for token in _stream_no_results(original_question):
                    yield token
        except Exception:
            logger.exception("Error en fallback de renderizado")
            yield ERROR_RETRY_SENTINEL + "Ocurrió un error al procesar los resultados."
    else:
        combined_pre = " ".join(pre_table_buffer)
        if pre_table_buffer and _is_vague_analysis(combined_pre) and captured_sql and SQL_RUNNER:
            logger.warning("Texto del LLM detectado como vago (alias de columna) — regenerando con datos reales")
            try:
                tool_args = RunSqlToolArgs(sql=captured_sql[-1])
                df = await SQL_RUNNER.run_sql(tool_args, None)
                if not df.empty:
                    cols = df.columns.tolist()
                    rows = df.to_dict("records")
                    analysis_text = ""
                    async for token in _stream_analysis(original_question, rows, cols):
                        analysis_text += token
                        yield token
                    if analysis_text:
                        response_chunks.append(analysis_text)
                        pre_table_buffer[:] = [analysis_text]
                        logger.info("Análisis de reemplazo (streaming) generado correctamente")
            except Exception:
                logger.exception("Error regenerando análisis desde texto vago")
                for chunk in pre_table_buffer:
                    response_chunks.append(chunk)
                    yield chunk
        else:
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

    # 7. Guardar en cache — solo si la respuesta tiene contenido real (no "0 resultados")
    _NO_RESULTS_MSG = "La consulta no devolvió resultados."
    if captured_sql:
        combined_response = "\n".join(response_chunks)
        if combined_response.strip() and combined_response.strip() != _NO_RESULTS_MSG:
            _store_sql_cache(
                original_question,
                captured_sql[-1],
                full_response=combined_response,
            )
            logger.info("SQL guardado en cache para: '%s'", original_question)
        else:
            logger.info("No se guarda en cache: respuesta vacía o sin resultados")
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

    global SQL_RUNNER
    SQL_RUNNER = SqlServerRunner(odbc_str)

    llm = OpenAILlmService(model=model, api_key=api_key)

    tools = ToolRegistry()
    tracking_tool = TrackingSqlTool(sql_runner=SQL_RUNNER)
    tools.register_local_tool(tracking_tool, access_groups=["admin", "user"])
    logger.info("TrackingSqlTool registrado")

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

    # Indexar join paths solo si el número en el store difiere del esperado
    # (evita llamadas a la API de embeddings en cada recarga en desarrollo)
    try:
        from .ingest_schema import (
            get_connection, fetch_all_fk_relations, build_join_path_docs,
        )
        conn = get_connection()
        cursor = conn.cursor()
        all_fks = fetch_all_fk_relations(cursor)
        docs = build_join_path_docs(all_fks)
        conn.close()

        # Cargar mapa FK en memoria para validación dinámica de JOINs en runtime
        global _FK_COL_SET
        _FK_COL_SET.clear()
        for _, p_table, p_col, _, r_table, r_col in all_fks:
            _FK_COL_SET.add((p_table.lower(), p_col.lower()))
            _FK_COL_SET.add((r_table.lower(), r_col.lower()))
        logger.info("FK col set cargado: %d pares (tabla, columna) únicos", len(_FK_COL_SET))

        try:
            existing = SCHEMA_STORE.col.get(
                where={"type": {"$in": ["join_path", "join_chain"]}},
                limit=len(docs) + 1,
            )
            existing_count = len(existing.get("ids") or [])
        except Exception:
            existing_count = 0

        if existing_count < len(docs):
            for d in docs:
                SCHEMA_STORE.upsert(
                    ids=[d["id"]],
                    documents=[d["doc"]],
                    metadatas=[d["meta"]],
                )
            logger.info("Join paths indexados: %d nuevos — total store: %d",
                        len(docs), SCHEMA_STORE.count())
        else:
            logger.info("Join paths ya presentes (%d) — sin re-indexar", existing_count)
    except Exception:
        logger.exception("Error verificando join paths al arrancar")

    return Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=SimpleUserResolver(),
        agent_memory=agent_memory,
    )