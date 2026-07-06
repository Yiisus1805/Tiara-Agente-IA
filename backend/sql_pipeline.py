"""
Garantiza que el SQL que finalmente se ejecuta sea seguro, válido y esté cacheado.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Optional

import httpx
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from vanna.capabilities.sql_runner import RunSqlToolArgs

from . import agent_state as state
from .schema_analyzer import build_dynamic_prompt_sections, _EXCLUDED_SUFFIXES

logger = logging.getLogger(__name__)


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
    """Añade JOIN a la tabla de tiempo en CTEs que usan su alias pero olvidaron el JOIN."""
    schema_meta = state.SCHEMA_META
    if schema_meta and schema_meta.date_table:
        date_alias  = schema_meta.date_alias
        date_table  = schema_meta.date_table
        date_schema = schema_meta.date_schema
        date_key    = schema_meta.date_key_col
        dimdate_join = (
            f"JOIN {date_schema}.{date_table} {date_alias} "
            f"ON FIS.OrderDateKey = {date_alias}.{date_key}"
        )
        alias_pattern = re.escape(date_alias) + r'\.'
        table_pattern = re.escape(date_table)
    else:
        # Fallback seguro si no hay metadata
        dimdate_join  = "JOIN dbo.DimDate DD ON FIS.OrderDateKey = DD.DateKey"
        alias_pattern = r'DD\.'
        table_pattern = r'DimDate'

    result = sql
    offset = 0

    for start, end in _extract_cte_spans(sql):
        body = result[start + offset: end + offset]

        if not re.search(alias_pattern, body, re.IGNORECASE):
            continue
        if re.search(table_pattern, body, re.IGNORECASE):
            continue

        # Insertar después del último JOIN que referencie una tabla conocida
        last_join = None
        for m in re.finditer(
            r'JOIN\s+\w+\.\w+\s+\w+\s+ON\s+\w+\.\w+\s*=\s*\w+\.\w+',
            body, re.IGNORECASE,
        ):
            last_join = m

        if not last_join:
            continue

        insert_at = start + offset + last_join.end()
        addition = f"\n    {dimdate_join}"
        result = result[:insert_at] + addition + result[insert_at:]
        offset += len(addition)
        logger.info("SQL corregido — JOIN %s añadido a CTE", dimdate_join)

    return result


_LANG_PREFIX_FIXES = [
    # DimProductCategory: ProductCategoryName → EnglishProductCategoryName
    (re.compile(r'\b(?<!\bEnglish)(?<!\bSpanish)(?<!\bFrench)(ProductCategoryName)\b', re.IGNORECASE),
     'EnglishProductCategoryName'),
    # DimProductSubcategory: ProductSubcategoryName → EnglishProductSubcategoryName
    (re.compile(r'\b(?<!\bEnglish)(?<!\bSpanish)(?<!\bFrench)(ProductSubcategoryName)\b', re.IGNORECASE),
     'EnglishProductSubcategoryName'),
    # DimProduct: ProductDescription → EnglishDescription
    (re.compile(r'\b(?<!\bEnglish)(?<!\bSpanish)(?<!\bFrench)(ProductDescription)\b', re.IGNORECASE),
     'EnglishDescription'),
    # DimProduct: ProductName → EnglishProductName (columna real en DimProduct)
    (re.compile(r'\b(?<!\bEnglish)(?<!\bSpanish)(?<!\bFrench)(ProductName)\b', re.IGNORECASE),
     'EnglishProductName'),
    # DimDate alias DD: DD.Date / DD.CalendarDate → DD.FullDateAlternateKey
    (re.compile(r'\bDD\.(?:Date|CalendarDate)\b', re.IGNORECASE),
     'DD.FullDateAlternateKey'),
    # DimDate alias DD: DD.MonthName → DD.EnglishMonthName
    (re.compile(r'\bDD\.MonthName\b', re.IGNORECASE),
     'DD.EnglishMonthName'),
    # DimDate alias DD: DD.DayName → DD.EnglishDayNameOfWeek
    (re.compile(r'\bDD\.DayName\b', re.IGNORECASE),
     'DD.EnglishDayNameOfWeek'),
]


def _fix_lang_prefix_columns(sql: str) -> str:
    """Corrige columnas que requieren prefijo English o cuyo nombre es incorrecto."""
    fixed = sql
    for pattern, replacement in _LANG_PREFIX_FIXES:
        new = pattern.sub(replacement, fixed)
        if new != fixed:
            logger.info("SQL corregido — columna renombrada a %s", replacement)
            fixed = new
    return fixed


# Detecta SUM/COUNT/AVG * 100 / SUM/COUNT/AVG sin NULLIF — división por cero potencial
_DIV_ZERO_PCT_RE = re.compile(
    r'((?:SUM|COUNT|AVG)\s*\([^)]+\)\s*(?:\*\s*100(?:\.0)?)?)\s*/\s*((?:SUM|COUNT|AVG)\s*\([^)]+\))',
    re.IGNORECASE,
)


def _fix_division_by_zero(sql: str) -> str:
    """Envuelve denominadores en NULLIF(..., 0) para prevenir errores de división por cero."""
    def _wrap(m: re.Match) -> str:
        numerator = m.group(1)
        denominator = m.group(2)
        if 'NULLIF' in denominator.upper():
            return m.group(0)
        logger.info("SQL corregido — NULLIF añadido para prevenir división por cero")
        return f"{numerator} / NULLIF({denominator}, 0)"
    return _DIV_ZERO_PCT_RE.sub(_wrap, sql)


_KEYWORD_ALIAS_RE = re.compile(
    r'\bFROM\s+(\w+)\s+(AS|ON|IN|BY)\b(?=\s*(?:\n|\r|\Z|JOIN\b|WHERE\b|ON\b|GROUP\b|ORDER\b|HAVING\b|INNER\b|LEFT\b|RIGHT\b|FULL\b|CROSS\b))',
    re.IGNORECASE,
)
_KEYWORD_ALIAS_MAP = {'AS': 'ASales', 'ON': 'OnRef', 'IN': 'InRef', 'BY': 'ByRef'}

# Captura FROM table AS <keyword> donde el alias es la propia keyword (ej. FROM AllSales AS AS)
_KEYWORD_SELF_ALIAS_RE = re.compile(
    r'\bFROM\s+(\w+)\s+AS\s+(AS|ON|IN|BY)\b',
    re.IGNORECASE,
)


def _fix_keyword_self_alias(sql: str) -> str:
    """Corrige FROM table AS <keyword> donde el alias es también una keyword reservada.

    Ejemplo: FROM AllSales AS AS → FROM AllSales ASales
    y reemplaza todas las referencias AS.columna → ASales.columna en el SQL.
    """
    result = sql
    for m in _KEYWORD_SELF_ALIAS_RE.finditer(sql):
        bad_alias = m.group(2).upper()
        safe_alias = _KEYWORD_ALIAS_MAP.get(bad_alias, bad_alias + 'Ref')
        result = _KEYWORD_SELF_ALIAS_RE.sub(
            lambda x: f'FROM {x.group(1)} {_KEYWORD_ALIAS_MAP.get(x.group(2).upper(), x.group(2).upper() + "Ref")}',
            result,
        )
        result = re.sub(rf'\b{re.escape(bad_alias)}\.', f'{safe_alias}.', result)
        logger.info("SQL corregido — alias doble-keyword 'AS %s' → '%s'", bad_alias, safe_alias)
        break
    return result


def _fix_keyword_table_alias(sql: str) -> str:
    """Corrige aliases de tabla que son palabras reservadas de SQL Server.

    Ejemplo: FROM AllSales AS → FROM AllSales ASales
    y sustituye todas las referencias AS.columna → ASales.columna
    """
    result = sql
    for m in _KEYWORD_ALIAS_RE.finditer(sql):
        bad = m.group(2).upper()
        safe = _KEYWORD_ALIAS_MAP.get(bad, bad + 'Ref')
        result = _KEYWORD_ALIAS_RE.sub(
            lambda x: f'FROM {x.group(1)} {_KEYWORD_ALIAS_MAP.get(x.group(2).upper(), x.group(2) + "Ref")}',
            result,
        )
        result = re.sub(rf'\b{re.escape(bad)}\.', f'{safe}.', result)
        logger.info("SQL corregido — alias reservado '%s' reemplazado por '%s'", bad, safe)
        break  # _KEYWORD_ALIAS_RE.sub ya reemplazó todos; la iteración es solo para loggear
    return result


def _sanitize_sql_aliases(sql: str) -> str:
    fixed = _fix_keyword_self_alias(sql)
    fixed = _fix_keyword_table_alias(fixed)
    fixed = _fix_lang_prefix_columns(fixed)
    fixed = _fix_window_order_by(fixed)

    fixed, removed_order_by = _remove_cte_order_by(fixed)
    if removed_order_by:
        logger.info("SQL corregido — ORDER BY eliminado de CTE sin TOP")
        has_top_outer = bool(re.search(r'\bSELECT\s+TOP\b', fixed, re.IGNORECASE))
        has_order_outer = bool(re.search(r'\bORDER\s+BY\b', fixed, re.IGNORECASE))
        if has_top_outer and not has_order_outer:
            fixed = fixed.rstrip().rstrip(';').rstrip() + '\n' + removed_order_by + ';'
            logger.info("SQL corregido — ORDER BY movido al SELECT final")

    fixed = _fix_fetch_first(fixed)
    fixed = _fix_missing_dimdate_join(fixed)
    fixed = _fix_division_by_zero(fixed)

    # Protege contextos donde las palabras coinciden con keywords válidos (no reservados)
    guarded = fixed
    for guard_pattern, placeholder in _SQL_KEYWORD_GUARDS:
        guarded = re.sub(guard_pattern, placeholder, guarded, flags=re.IGNORECASE)

    sanitized = guarded
    for pattern, replacement in _RESERVED_ALIAS_REPLACEMENTS.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    sanitized = sanitized.replace('FETCH __NEXT__', 'FETCH NEXT')
    sanitized = sanitized.replace('__CURRENT_', 'CURRENT_')
    return sanitized


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

# Intención destructiva sobre la BD
# "eliminar/borrar/suprimir" + artículo se bloquea siempre (verbos inherentemente destructivos).
# "modificar/actualizar" requiere objeto de BD explícito para evitar falsos positivos.
_DB_OBJECT = (
    r'(?:registro|dato|fila|entrada|tabla|record'
    r'|venta|ventas|pedido|pedidos|orden|ordenes|órdenes'
    r'|cliente|clientes|producto|productos|empleado|empleados'
    r'|precio|precios|transaccion|transacciones|transacción|transacciones'
    r'|compra|compras|factura|facturas|inventario)'
)
_DESTRUCTIVE_INTENT_RE = re.compile(
    r'\b('
    # Eliminar/borrar/suprimir + artículo → siempre destructivo en contexto de datos
    r'elimin[ae]r?\s+(?:el|la|los|las|este|ese|un|una|todos?|todas?)\s+\w+|'
    r'borr[ae]r?\s+(?:el|la|los|las|este|ese|un|una|todos?|todas?)\s+\w+|'
    r'suprim[ei]r?\s+(?:el|la|los|las|este|ese|un|una|todos?|todas?)\s+\w+|'
    # SQL keywords escritos en lenguaje natural
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

READONLY_REFUSAL = (
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


_JOIN_ON_RE = re.compile(
    r'\bON\s+(?:\w+\.)?(\w+)\s*=\s*(?:\w+\.)?(\w+)',
    re.IGNORECASE,
)

# Para validación FK-aware: extrae alias→tabla y condiciones JOIN ON completas.
# El "(?:AS\s+)?" opcional es crítico: sin él, "FROM AllSales AS ASales" captura
# "AS" como alias (se descarta por ser keyword) y nunca aprende que ASales→AllSales,
# dejando ese alias sin resolver en toda validación posterior.
_TABLE_ALIAS_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)\s+(?:AS\s+)?(\w+)\b',
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
    schema_meta = state.SCHEMA_META

    for m in _JOIN_ON_RE.finditer(sql):
        left, right = m.group(1), m.group(2)
        if left.lower().endswith("key") and right.lower().endswith("key"):
            if not _keys_compatible(left, right):
                errors.append(
                    f"JOIN con columnas incompatibles: {left} = {right}"
                )

    # Detecta uso del alias de la tabla de tiempo sin el JOIN correspondiente
    if schema_meta and schema_meta.date_table:
        _da = schema_meta.date_alias
        _dt = schema_meta.date_table
        _ds = schema_meta.date_schema
        _dk = schema_meta.date_key_col
    else:
        _da, _dt, _ds, _dk = "DD", "DimDate", "dbo", "DateKey"
    if re.search(re.escape(_da) + r'\.', sql, re.IGNORECASE):
        if not re.search(re.escape(_dt), sql, re.IGNORECASE):
            errors.append(
                f"SQL usa {_da}. (alias de {_dt}) pero falta "
                f"JOIN {_ds}.{_dt} {_da} ON <Fact>.OrderDateKey = {_da}.{_dk}"
            )

    # Validación FK-aware: detecta columnas *Key que no existen en la tabla referenciada
    if state._FK_COL_SET:
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

                if (table, col.lower()) not in state._FK_COL_SET:
                    # Buscar en qué tablas sí existe esa columna
                    tables_with_col = sorted(
                        t for (t, c) in state._FK_COL_SET if c == col.lower() and t != table
                    )
                    hint = (
                        f" — la columna sí existe en: {', '.join(tables_with_col)}"
                        if tables_with_col else ""
                    )
                    errors.append(
                        f"La columna {col} no existe en {table}{hint}. "
                        f"Usa la tabla correcta para este JOIN."
                    )

        # Validación de fan-out: un CTE de ventas (ej. AllSales) o una tabla Fact
        # joineada a una dimensión por una columna que NO es la clave primaria de
        # esa dimensión puede multiplicar las filas del fact si esa dimensión tiene
        # varias filas por ese valor de clave (ej. DimGeography tiene varias filas —
        # una por ciudad/código postal — para cada SalesTerritoryKey).
        if state._PK_GRAIN_TABLE and schema_meta:
            # alias_to_table guarda los nombres de tabla en minúsculas, pero
            # schema_meta.tables usa el nombre real (mixed-case) como clave.
            _tables_lower = {t.lower(): t for t in schema_meta.tables}

            def _is_fact_like(alias_lower: str) -> bool:
                if alias_lower in cte_names:
                    return True
                t = alias_to_table.get(alias_lower)
                if not t:
                    return False
                if t in cte_names:
                    return True  # alias de un CTE (ej. "ASales" → "AllSales")
                tmeta = schema_meta.tables.get(_tables_lower.get(t, ""))
                return bool(tmeta and tmeta.table_type == "Fact")

            # Cada JOIN por una *Key cuyo grano se conoce, con su tabla de grano.
            key_edges = []
            for m3 in _JOIN_ON_FULL_RE.finditer(sql):
                a1, c1, a2, c2 = m3.group(1), m3.group(2), m3.group(3), m3.group(4)
                if c1.lower() == c2.lower() and c1.lower().endswith("key"):
                    grain = state._PK_GRAIN_TABLE.get(c1.lower())
                    if grain:
                        key_edges.append((a1, a2, c1, grain))

            all_join_aliases = {
                a.lower()
                for m3 in _JOIN_ON_FULL_RE.finditer(sql)
                for a in (m3.group(1), m3.group(3))
            }
            fact_connected = {a for a in all_join_aliases if _is_fact_like(a)}

            # El fan-out se propaga transitivamente: si una dimensión YA unida 1:1 al
            # fact (ej. DimSalesTerritory) se vuelve a unir a OTRA tabla por la misma
            # *Key (ej. DimSalesTerritory.SalesTerritoryKey = DimGeography.SalesTerritoryKey),
            # esa segunda tabla multiplica igual las filas del fact aunque no esté unida
            # DIRECTAMENTE a él. Pero solo se propaga a través de saltos SEGUROS (el
            # destino es exactamente la tabla de grano de esa *Key) — un salto inseguro
            # es justo el fan-out que se quiere detectar, no hay que seguir más allá de
            # él (si no, ej. en un snowflake producto→subcategoría→categoría, cada tabla
            # terminaría "fact_connected" igual y dispararía falsos positivos cruzados
            # entre claves que no tienen relación entre sí).
            changed = True
            while changed:
                changed = False
                for a1, a2, col, grain in key_edges:
                    for src, dst in ((a1.lower(), a2.lower()), (a2.lower(), a1.lower())):
                        if src not in fact_connected or dst in fact_connected:
                            continue
                        if dst in cte_names or _is_fact_like(dst):
                            fact_connected.add(dst)
                            changed = True
                            continue
                        dst_table = alias_to_table.get(dst)
                        if dst_table and dst_table == grain:
                            fact_connected.add(dst)
                            changed = True

            # Reportar los saltos inseguros: un *Key edge donde un lado está
            # fact_connected y el otro NO es ni la tabla de grano ni fact-like.
            reported: set[tuple[str, str]] = set()
            for a1, a2, col1, grain_table in key_edges:
                for fact_alias, dim_alias in ((a1, a2), (a2, a1)):
                    if fact_alias.lower() not in fact_connected:
                        continue
                    if dim_alias.lower() in cte_names or _is_fact_like(dim_alias.lower()):
                        continue
                    # Si dim_alias YA está fact_connected (verificado seguro vía su
                    # PROPIA *Key en otro edge — ej. PS es grano correcto de
                    # ProductSubcategoryKey), no se le puede acusar de "riesgoso"
                    # solo porque no coincide con el grano de ESTA OTRA *Key — esa
                    # comparación no tiene relación con esa tabla.
                    if dim_alias.lower() in fact_connected:
                        continue
                    dim_table = alias_to_table.get(dim_alias.lower())
                    if not dim_table or dim_table == grain_table:
                        continue
                    dedup_key = (dim_table, col1.lower())
                    if dedup_key in reported:
                        continue
                    reported.add(dedup_key)

                    # Si la consulta usa un atributo de dim_table que NO existe en
                    # grain_table (ej. City solo está en DimGeography, no en
                    # DimSalesTerritory), "usa grain_table en su lugar" es un consejo
                    # imposible de seguir — el LLM queda atascado repitiendo el mismo
                    # JOIN. La causa real: si esta consulta combina canales con UNION
                    # ALL, ya perdió la clave de cliente/reseller individual al UNIONar
                    # por esa *Key, así que el JOIN a dim_table debe hacerse POR
                    # SEPARADO en cada rama, antes del UNION.
                    dim_meta = schema_meta.tables.get(_tables_lower.get(dim_table, ""))
                    grain_meta = schema_meta.tables.get(_tables_lower.get(grain_table, ""))
                    exclusive_cols = [
                        c for c in (dim_meta.col_names if dim_meta else [])
                        if grain_meta and c not in grain_meta.col_names
                        and re.search(rf'\b{re.escape(dim_alias)}\.{re.escape(c)}\b', sql, re.IGNORECASE)
                    ]
                    if exclusive_cols:
                        # En vez de una descripción genérica (que el LLM no siempre logra
                        # ejecutar — requiere reestructurar el UNION ALL con un puente
                        # distinto por rama), se calcula la ruta real de FKs desde CADA
                        # tabla fact hasta dim_table y se la da explícita, con nombres
                        # reales de tabla/columna — sin asumir ningún esquema específico.
                        real_dim_table = _tables_lower.get(dim_table, dim_table)
                        bridge_lines = []
                        for fact_name in (schema_meta.union_fact_tables or []):
                            path = _find_bridge_path(fact_name, real_dim_table, schema_meta, max_hops=3)
                            if path:
                                hops = " → ".join(
                                    f"{frm}.{c} → {to}.{tc}" for frm, c, to, tc in path
                                )
                                bridge_lines.append(f"  - Desde {fact_name}: JOIN {hops}")
                        bridge_text = (
                            "\nRutas reales para llegar a " + real_dim_table + " desde cada fuente:\n"
                            + "\n".join(bridge_lines)
                        ) if bridge_lines else ""

                        errors.append(
                            f"JOIN riesgoso: {dim_table} (alias {dim_alias}) puede tener "
                            f"múltiples filas por cada valor de {col1} — su clave primaria no "
                            f"es {col1}. Además necesitas {', '.join(exclusive_cols)}, que NO "
                            f"existe en {grain_table}, así que no puedes simplemente cambiar "
                            f"de tabla. Si esta consulta combina varias fuentes de venta con "
                            f"UNION ALL, {col1} ya perdió la granularidad necesaria para llegar "
                            f"a {dim_table} de forma única — debes hacer ese JOIN POR SEPARADO "
                            f"dentro de cada rama del UNION, seleccionando ya el atributo que "
                            f"necesitas ({', '.join(exclusive_cols)}) en cada rama, y recién "
                            f"después UNION ALL de los resultados ya enriquecidos. "
                            f"NUNCA combines primero con UNION ALL y unas {dim_table} después "
                            f"por {col1}.{bridge_text}"
                        )
                    else:
                        errors.append(
                            f"JOIN riesgoso: {dim_table} (alias {dim_alias}) puede tener "
                            f"múltiples filas por cada valor de {col1} — su clave primaria no "
                            f"es {col1}, así que uniría de más y duplicaría los totales. "
                            f"Usa {grain_table} en su lugar para este JOIN por {col1}, esa sí "
                            f"tiene exactamente una fila por cada valor."
                        )

    return errors


def _build_person_correction_rules() -> str:
    """Genera las reglas de corrección para referencias a personas según la metadata actual."""
    schema_meta = state.SCHEMA_META
    if not schema_meta:
        return (
            "- Para VENDEDORES: usa DimEmployee + tabla de ventas reseller ON EmployeeKey.\n"
            "- Para CLIENTES: usa DimCustomer + tabla de ventas internet ON CustomerKey.\n"
            "- NUNCA mezcles tablas de empleados con tablas de clientes."
        )
    emp_dim  = schema_meta.employee_dim  or "DimEmployee"
    emp_fact = schema_meta.employee_fact or "FactResellerSales"
    cust_dim  = schema_meta.customer_dim  or "DimCustomer"
    cust_fact = schema_meta.customer_fact or "FactInternetSales"
    emp_alias  = schema_meta.aliases.get(emp_dim,  "DE")
    ef_alias   = schema_meta.aliases.get(emp_fact, "FRS")
    cust_alias = schema_meta.aliases.get(cust_dim,  "DC")
    cf_alias   = schema_meta.aliases.get(cust_fact, "FIS")
    return (
        f"- Si la pregunta menciona un VENDEDOR/EMPLEADO: "
        f"usa {emp_dim} {emp_alias} + {emp_fact} {ef_alias} "
        f"ON {ef_alias}.EmployeeKey = {emp_alias}.EmployeeKey. "
        f"NUNCA busques un vendedor en {cust_dim}.\n"
        f"- Si la pregunta menciona un CLIENTE: "
        f"usa {cust_dim} {cust_alias} + {cust_fact} {cf_alias} "
        f"ON {cf_alias}.CustomerKey = {cust_alias}.CustomerKey. "
        f"NUNCA busques un cliente en {emp_dim}.\n"
        f"- NUNCA mezcles {emp_dim} con {cust_dim} en la misma consulta para una persona."
    )


def _diagnose_zero_rows(sql: str, question: str) -> str:
    """
    Analiza el SQL y la pregunta para generar un diagnóstico específico de por qué
    la consulta devolvió 0 filas. Le da al LLM de corrección una hipótesis concreta
    en lugar de "revisa los JOINs y filtros" genérico.
    """
    hints: list[str] = []
    q = question.lower()
    sql_lower = sql.lower()

    schema_meta = state.SCHEMA_META
    if schema_meta:
        ed = schema_meta.employee_dim or ""
        ef = schema_meta.employee_fact or ""
        cd = schema_meta.customer_dim or ""
        cf = schema_meta.customer_fact or ""

        uses_customer = cd and cd.lower() in sql_lower
        uses_employee = ed and ed.lower() in sql_lower

        sold_verbs    = {"vendió", "vendieron", "realizó ventas", "cuánto vendió", "vendedor"}
        bought_verbs  = {"compró", "gastó", "cuánto gastó", "pagó", "cliente"}

        asks_employee = any(v in q for v in sold_verbs)
        asks_customer = any(v in q for v in bought_verbs)

        if uses_customer and asks_employee and ed:
            hints.append(
                f"CAUSA PROBABLE — tabla incorrecta: el SQL usa {cd} (clientes) pero la pregunta "
                f"habla de alguien que VENDIÓ, lo que corresponde a un empleado. "
                f"Reescribe usando {ed} ({ef}) con JOIN en EmployeeKey."
            )
        elif uses_employee and asks_customer and cd:
            hints.append(
                f"CAUSA PROBABLE — tabla incorrecta: el SQL usa {ed} (empleados) pero la pregunta "
                f"habla de alguien que COMPRÓ, lo que corresponde a un cliente. "
                f"Reescribe usando {cd} ({cf}) con JOIN en CustomerKey."
            )

    # Nombres propios con posible problema de mayúsculas/ortografía
    name_matches = re.findall(
        r"(?:FirstName|LastName|FullName|Name)\s*=\s*'([^']+)'", sql, re.IGNORECASE
    )
    if name_matches:
        hints.append(
            f"Verifica que los valores '{', '.join(name_matches)}' existen exactamente así "
            f"en la base de datos (distinción mayúsculas/minúsculas y ortografía)."
        )

    if not hints:
        hints.append(
            "Posibles causas: (1) tabla incorrecta para la entidad buscada, "
            "(2) valor del filtro WHERE no existe en la BD, "
            "(3) JOIN mal definido que elimina todas las filas."
        )

    return "\n".join(hints)


_INVALID_COLUMN_RE = re.compile(r"Invalid column name '(\w+)'", re.IGNORECASE)
_TABLE_REF_RE = re.compile(r'\b(?:FROM|JOIN)\s+(?:\w+\.)?(\w+)\b', re.IGNORECASE)
_DRIVER_ERROR_RE = re.compile(r'\[SQL Server\](.+?\(\d+\))', re.IGNORECASE)
_UNBOUND_IDENTIFIER_RE = re.compile(
    r'multi-part identifier "(\w+)\.\w+" could not be bound', re.IGNORECASE
)
_ALIAS_DEF_RE = re.compile(
    r'\b(?:FROM|JOIN)\s+(?:\w+\.)?\w+\s+(?:AS\s+)?(\w+)\b', re.IGNORECASE
)
_CTE_NAME_RE = re.compile(r'\b(\w+)\s+AS\s*\(', re.IGNORECASE)
_SQL_ALIAS_STOPWORDS = {
    "on", "where", "group", "order", "join", "inner", "left", "right", "outer",
    "select", "union", "as",
}


def _extract_sql_error_detail(exc_text: str, fallback_len: int = 400) -> str:
    """pandas/SQLAlchemy incrustan el SQL completo DOS veces en el mensaje de
    excepción (al inicio en 'Execution failed on sql ...' y al final en
    '[SQL: ...]'), con el detalle real del driver (ej. "Invalid column name")
    en el medio — si el SQL es largo, truncar por los primeros o los últimos
    N caracteres pierde igual ese detalle. Se busca el patrón del driver
    primero; si no aparece, se recurre al truncado simple como respaldo."""
    m = _DRIVER_ERROR_RE.search(exc_text)
    if m:
        return m.group(1).strip()
    return exc_text[-fallback_len:]


def _suggest_real_columns(bad_sql: str, problem: str) -> str:
    """Si el error es 'Invalid column name X' de SQL Server, lista las columnas
    REALES de las tablas que la consulta ya referencia. Es más confiable que
    adivinar por parecido de texto con el nombre inventado (ej. el LLM podría
    probar 'SalesTerritory', 'Country' o 'SalesTerritoryName' en distintos
    intentos — ninguno comparte substring con la columna real
    'SalesTerritoryCountry' — pero la tabla DimSalesTerritory sí está bien
    identificada en el FROM/JOIN, así que mostrar su lista de columnas real
    funciona sin importar qué nombre haya inventado el LLM)."""
    schema_meta = state.SCHEMA_META
    if not schema_meta or not _INVALID_COLUMN_RE.search(problem):
        return ""

    referenced = {
        m.group(1) for m in _TABLE_REF_RE.finditer(bad_sql)
        if m.group(1) in schema_meta.tables
    }
    if not referenced:
        return ""

    lines = [
        f"{tname}: {', '.join(sorted(schema_meta.tables[tname].col_names))}"
        for tname in sorted(referenced)
    ]
    return "Columnas REALES de las tablas que ya usa la consulta:\n" + "\n".join(lines)


def _suggest_valid_aliases(bad_sql: str, problem: str) -> str:
    """Si el error es 'multi-part identifier X.Y could not be bound' (alias no
    definido), lista los alias/nombres de tabla y CTE REALMENTE definidos en la
    consulta. Ocurre cuando el corrector reescribe el FROM/JOIN (ej. quita un
    alias) pero olvida actualizar una referencia que quedó apuntando al alias
    viejo — decirle explícitamente cuáles son válidos evita que repita la
    misma inconsistencia en el siguiente intento."""
    m = _UNBOUND_IDENTIFIER_RE.search(problem)
    if not m:
        return ""
    bad_alias = m.group(1)

    defined = {t.group(1) for t in _TABLE_REF_RE.finditer(bad_sql)}
    for a in _ALIAS_DEF_RE.finditer(bad_sql):
        if a.group(1).lower() not in _SQL_ALIAS_STOPWORDS:
            defined.add(a.group(1))
    for c in _CTE_NAME_RE.finditer(bad_sql):
        defined.add(c.group(1))
    defined.discard(bad_alias)
    if not defined:
        return ""

    return (
        f"El alias/nombre '{bad_alias}' que usas en la consulta NO está definido en ningún "
        f"FROM/JOIN de ESTA consulta — probablemente lo cambiaste o eliminaste al reescribir "
        f"el FROM/JOIN pero dejaste una referencia vieja sin actualizar. "
        f"Alias/nombres de tabla o CTE realmente definidos aquí: {', '.join(sorted(defined))}. "
        f"Usa uno de estos (el que corresponda a la tabla correcta) en TODAS sus referencias — "
        f"no dejes ninguna mención de '{bad_alias}'."
    )


async def _get_corrected_sql(question: str, bad_sql: str, problem: str) -> Optional[str]:
    """Llama al LLM con el SQL problemático y el error para obtener una versión corregida."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None

    schema_context = ""
    if state.SCHEMA_STORE:
        try:
            hits = state.SCHEMA_STORE.query(question, k=6)
            schema_context = "\n".join(h.get("doc", "") for h in hits if h.get("doc"))
        except Exception:
            pass

    # Si el problema es de 0 filas, agregar diagnóstico específico de causa probable
    is_zero_rows = "0 resultados" in problem or "0 filas" in problem
    diagnosis_section = (
        f"\nDIAGNÓSTICO (úsalo para orientar la corrección):\n{_diagnose_zero_rows(bad_sql, question)}\n"
        if is_zero_rows else ""
    )

    # Si el error es de columna inexistente, mostrar las columnas reales de las
    # tablas ya referenciadas, en vez de dejar que el LLM vuelva a adivinar a ciegas
    column_suggestions = _suggest_real_columns(bad_sql, problem)
    if column_suggestions:
        diagnosis_section += f"\n{column_suggestions}\n"

    # Si el error es de alias no vinculado ("multi-part identifier ... could not
    # be bound"), mostrar los alias/nombres realmente definidos en la consulta
    alias_suggestions = _suggest_valid_aliases(bad_sql, problem)
    if alias_suggestions:
        diagnosis_section += f"\n{alias_suggestions}\n"

    # Mismas reglas de fuente de ventas (UNION ALL, tablas XL_CCI/PageCompressed
    # prohibidas) que usa la generación original — sin esto, el RAG genérico de
    # schema_context puede surgir con una tabla variante (ej. FactResellerSalesXL_CCI)
    # y el corrector la usa por error en vez de la tabla canónica.
    sales_source = ""
    if state.SCHEMA_META:
        try:
            sales_source = build_dynamic_prompt_sections(state.SCHEMA_META).get("sales_source", "")
        except Exception:
            pass

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"SQL generado con error:\n{bad_sql}\n\n"
        f"Problema detectado: {problem}"
        f"{diagnosis_section}\n\n"
        f"{sales_source}\n\n"
        f"Esquema disponible:\n{schema_context}\n\n"
        "REGLAS CRÍTICAS:\n"
        "- ALIAS: NUNCA uses AS, ON, IN, BY, FROM, WHERE, JOIN, GROUP, ORDER, SELECT como alias de tabla. "
        "Usa alias descriptivos cortos del esquema. "
        "Incorrecto: FROM AllSales AS AS. Correcto: FROM AllSales.\n"
        "- CTE: Si defines un CTE llamado AllSales, refiérelo directamente como AllSales sin alias adicional.\n"
        "- CONSISTENCIA DE ALIAS: si cambias, agregas o quitas un alias de tabla/CTE en el FROM/JOIN "
        "(ej. quitar 'AS ASales' y dejar solo 'AllSales'), debes actualizar TODAS las demás referencias "
        "a esa tabla en el resto de la consulta (SELECT, WHERE, ON, GROUP BY) para usar el mismo nombre. "
        "Antes de responder, revisa que cada alias/nombre de tabla usado en SELECT/WHERE/ON esté "
        "definido en un FROM o JOIN de esa misma consulta — nunca dejes una referencia a un alias "
        "que ya no existe.\n"
        + _build_person_correction_rules() +
        "\n\n"
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
                    "temperature": 0,
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
    schema_meta = state.SCHEMA_META
    if schema_meta and schema_meta.date_table:
        dta = schema_meta.date_alias
        dt  = schema_meta.date_table
        ds  = schema_meta.date_schema
        dk  = schema_meta.date_key_col
        yc  = schema_meta.date_year_col
        hint = (
            f"Añade JOIN {ds}.{dt} {dta} ON <Fact>.OrderDateKey = {dta}.{dk} "
            f"y filtra con WHERE {dta}.{yc} = {year}."
        )
    else:
        hint = (
            f"Añade un JOIN a dbo.DimDate DD ON <FactTable>.OrderDateKey = DD.DateKey "
            f"y filtra con WHERE DD.CalendarYear = {year}."
        )
    return [
        f"La pregunta menciona el año {year} pero el SQL no tiene filtro temporal. "
        f"{hint} "
        "Sin este filtro la consulta suma todos los años y devuelve resultados incorrectos."
    ]


_CHANNEL_KEYWORD_RE = re.compile(
    r'\b(online|internet|canal\s+directo|solo\s+online|solo\s+internet'
    r'|reseller|distribuidor|canal\s+indirecto|solo\s+reseller)\b',
    re.IGNORECASE,
)
_CUSTOMER_SQL_RE = re.compile(r'\b(DimCustomer|CustomerKey)\b', re.IGNORECASE)
_CUSTOMER_QUESTION_RE = re.compile(
    r'\b(cliente|clientes|customer|customers)\b',
    re.IGNORECASE,
)
# Patrón "Nombre Apellido" — detecta nombres propios en la pregunta
PERSON_NAME_RE = re.compile(r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b')


def _find_bridge_path(
    start_table: str, target_table: str, schema_meta, max_hops: int = 2
) -> Optional[list[tuple]]:
    """BFS hacia adelante para hallar la ruta de FKs más corta (<=max_hops) desde
    start_table hasta target_table. Retorna la lista de (tabla_origen, columna,
    tabla_destino, columna_destino) que forman la ruta, o None si no existe
    dentro de max_hops. Sirve para decirle al LLM, con nombres reales, por dónde
    debe pasar para llegar a una dimensión riesgosa SIN usar la *Key que causa
    fan-out (ej. FactResellerSales -> DimReseller -> DimGeography)."""
    if start_table == target_table:
        return []
    frontier: list[tuple[str, list[tuple]]] = [(start_table, [])]
    visited = {start_table}
    for _ in range(max_hops):
        next_frontier: list[tuple[str, list[tuple]]] = []
        for table, path in frontier:
            tmeta = schema_meta.tables.get(table)
            if not tmeta:
                continue
            for col, ref_schema, ref_table, ref_col in tmeta.fk_refs:
                if ref_table not in schema_meta.tables or ref_table in visited:
                    continue
                new_path = path + [(table, col, ref_table, ref_col)]
                if ref_table == target_table:
                    return new_path
                visited.add(ref_table)
                next_frontier.append((ref_table, new_path))
        frontier = next_frontier
    return None


def _reachable_tables(start_table: str, schema_meta, max_hops: int = 2) -> set[str]:
    """BFS SOLO hacia adelante (siguiendo fk_refs propios de cada tabla, nunca al
    revés) para hallar qué tablas son alcanzables desde start_table en máximo
    max_hops saltos. Cada salto hacia adelante (hijo.fk -> padre.pk) es siempre
    N:1, así que el JOIN resultante es determinista y seguro.

    Bidireccional sería incorrecto: ej. DimEmployee tiene su PROPIO FK hacia
    DimSalesTerritory (para territorio asignado), y FactInternetSales también
    llega a DimSalesTerritory — pero eso NO significa que FactInternetSales
    pueda darte datos de empleado/vendedor (no los tiene). Tratar ese FK como
    bidireccional haría parecer que DimEmployee es "alcanzable" desde
    FactInternetSales, lo cual es falso."""
    adjacency: dict[str, set[str]] = {}
    for tname, tmeta in schema_meta.tables.items():
        for _, _, ref_table, _ in tmeta.fk_refs:
            if ref_table in schema_meta.tables:
                adjacency.setdefault(tname, set()).add(ref_table)

    visited = {start_table}
    frontier = {start_table}
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for t in frontier:
            next_frontier |= adjacency.get(t, set()) - visited
        visited |= next_frontier
        frontier = next_frontier
    visited.discard(start_table)
    return visited


def _validate_sales_source(sql: str, question: str) -> list[str]:
    """Detecta cuando el SQL usa solo una tabla de ventas en consultas de totales generales.

    Con SCHEMA_META se usa la lista real de tablas unificables; sin ella,
    la validación se omite para evitar falsos positivos.
    """
    schema_meta = state.SCHEMA_META
    if not schema_meta or len(schema_meta.union_fact_tables) < 2:
        return []

    tables_in_sql = [
        n for n in schema_meta.union_fact_tables
        if re.search(rf'\b{re.escape(n)}\b', sql, re.IGNORECASE)
    ]
    tables_absent = [
        n for n in schema_meta.union_fact_tables if n not in tables_in_sql
    ]

    if not tables_in_sql or not tables_absent:
        return []
    if _CHANNEL_KEYWORD_RE.search(question):
        return []

    # Si las dimensiones que la consulta realmente une (excluyendo la tabla de
    # tiempo, que cuelga de prácticamente cualquier fact table y no indica nada)
    # son alcanzables por FK desde alguna tabla ausente, entonces esa tabla SÍ
    # podría aportar las mismas filas — no se exime. Esto evita el falso negativo
    # de antes: ver "CustomerKey" en el SQL y asumir que la pregunta es "sobre
    # clientes" cuando en realidad CustomerKey es solo el puente de JOIN hacia una
    # dimensión compartida (ej. DimGeography, alcanzable también desde
    # FactResellerSales vía DimReseller.GeographyKey).
    referenced_tables = {
        m.group(1) for m in _TABLE_REF_RE.finditer(sql)
        if m.group(1) in schema_meta.tables
    }
    skip = set(tables_in_sql) | set(tables_absent)
    if schema_meta.date_table:
        skip.add(schema_meta.date_table)
    relevant_dims = referenced_tables - skip

    if relevant_dims:
        exempt = all(
            not (relevant_dims & _reachable_tables(absent_table, schema_meta, max_hops=2))
            for absent_table in tables_absent
        )
        if exempt:
            logger.debug(
                "Validación sales_source omitida: dimensiones %s no alcanzables desde %s",
                relevant_dims, tables_absent,
            )
            return []

    cols = schema_meta.union_common_cols or [
        "OrderDateKey", "SalesTerritoryKey", "ProductKey", "SalesAmount", "TotalProductCost"
    ]
    cols_str = ", ".join(cols)
    union_parts = [
        f"SELECT {cols_str} FROM {schema_meta.tables[n].schema}.{n}"
        for n in schema_meta.union_fact_tables
        if n in schema_meta.tables
    ]
    union_sql = " UNION ALL ".join(union_parts)

    present_str = ", ".join(tables_in_sql)
    absent_str  = ", ".join(tables_absent)
    return [
        f"El SQL usa solo {present_str} pero la pregunta no especifica canal. "
        f"Las ventas totales requieren UNION ALL de todas las tablas de venta. "
        f"Reescribe usando: WITH AllSales AS ({union_sql}) "
        f"y aplica el JOIN y filtros sobre AllSales. "
        f"Sin {absent_str} los totales son parciales e incorrectos."
    ]


_INTERNET_CHANNEL_RE = re.compile(
    r'\b(online|internet|canal\s+directo|solo\s+online|solo\s+internet)\b', re.IGNORECASE,
)
_RESELLER_CHANNEL_RE = re.compile(
    r'\b(reseller|distribuidor|canal\s+indirecto|solo\s+reseller)\b', re.IGNORECASE,
)


def _validate_channel_scope(sql: str, question: str) -> list[str]:
    """Caso inverso de _validate_sales_source: si la pregunta pide EXPLÍCITAMENTE
    un solo canal de venta (ej. "ventas de internet"), pero el SQL combina con
    UNION ALL también la(s) tabla(s) del OTRO canal, el resultado incluye datos
    que el usuario no pidió — infla la cifra igual que el caso de fuente
    incompleta, solo que al revés (de más, no de menos)."""
    schema_meta = state.SCHEMA_META
    if not schema_meta or len(schema_meta.union_fact_tables) < 2:
        return []

    wants_internet = bool(_INTERNET_CHANNEL_RE.search(question))
    wants_reseller = bool(_RESELLER_CHANNEL_RE.search(question))
    if wants_internet == wants_reseller:  # ninguno o ambos a la vez — ambiguo
        return []

    keyword = "internet" if wants_internet else "reseller|distributor"
    target_tables = [n for n in schema_meta.union_fact_tables if re.search(keyword, n, re.IGNORECASE)]
    other_tables = [n for n in schema_meta.union_fact_tables if n not in target_tables]
    if not target_tables or not other_tables:
        return []

    used_other = [
        n for n in other_tables
        if re.search(rf'\b{re.escape(n)}\b', sql, re.IGNORECASE)
    ]
    if not used_other:
        return []

    canal = "Internet" if wants_internet else "Reseller"
    return [
        f"La pregunta pide específicamente el canal {canal}, pero el SQL combina con "
        f"UNION ALL también {', '.join(used_other)} — eso incluye ventas de un canal que "
        f"el usuario no pidió, inflando el resultado. Usa SOLO "
        f"{', '.join(target_tables)}, sin UNION ALL de otras fuentes de venta."
    ]


def _validate_no_excluded_tables(sql: str) -> list[str]:
    """Detecta referencias a tablas de benchmark/demo incluidas en AdventureWorksDW
    para probar columnstore indexes (sufijos en _EXCLUDED_SUFFIXES, ej. XL_CCI,
    XL_PageCompressed) — tienen datos sintéticos con volumen y montos inflados
    cientos de veces respecto a la tabla real (ej. FactResellerSalesXL_CCI tiene
    11.6M filas y $17,270M vs. los 60,855 filas y $80.4M reales de
    FactResellerSales) y arruinan cualquier resultado si se usan por error.
    Corre siempre, sin importar qué otro validador ya haya disparado en esta
    pasada — una corrección de JOIN o de otra regla puede reintroducir o dejar
    intacta una referencia a estas tablas sin que el validador correspondiente
    llegue a revisarlo."""
    schema_meta = state.SCHEMA_META
    if not schema_meta:
        return []

    excluded_in_sql = [
        n for n in schema_meta.fact_tables
        if any(n.endswith(s) for s in _EXCLUDED_SUFFIXES)
        and re.search(rf'\b{re.escape(n)}\b', sql, re.IGNORECASE)
    ]
    if not excluded_in_sql:
        return []

    suggestions = []
    for n in excluded_in_sql:
        for suffix in _EXCLUDED_SUFFIXES:
            if n.endswith(suffix):
                canonical = n[: -len(suffix)]
                if canonical in schema_meta.tables:
                    suggestions.append(f"{n} → {canonical}")
                break

    return [
        f"El SQL usa {', '.join(excluded_in_sql)} — son tablas de demostración/benchmark "
        f"de AdventureWorksDW con datos sintéticos inflados (cientos de veces más volumen "
        f"que los datos reales), NUNCA deben usarse para responder preguntas de negocio. "
        f"Usa en su lugar la tabla real: "
        f"{', '.join(suggestions) if suggestions else 'la misma tabla sin el sufijo de benchmark'}."
    ]


_SELECT_COLS_RE = re.compile(
    r'\bSELECT\b\s+(?:TOP\s+\d+\s+)?(?:DISTINCT\s+)?(.*?)\bFROM\b',
    re.IGNORECASE | re.DOTALL,
)
_BARE_KEY_REF_RE = re.compile(
    r'^\(?\s*(?:\[?\w+\]?\.)?\[?(\w*Key)\]?\s*\)?$', re.IGNORECASE
)


def _split_top_level_commas(s: str) -> list[str]:
    """Separa por comas de nivel superior, ignorando comas dentro de paréntesis
    (ej. SUM(a, b)), para poder analizar cada columna del SELECT por separado."""
    parts, depth, current = [], 0, []
    for ch in s:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return parts


def _get_final_select_segment(sql: str) -> str:
    """Devuelve el texto desde el SELECT final (después de los CTEs, si hay)."""
    spans = _extract_cte_spans(sql)
    if not spans:
        return sql
    last_close_paren = max(end for _, end in spans)
    return sql[last_close_paren + 1:]


def _validate_no_raw_keys(sql: str) -> list[str]:
    """Detecta columnas *Key expuestas directamente en el SELECT final, sin JOIN a
    la dimensión correspondiente — el LLM a veces agrupa/ordena por la clave numérica
    y se olvida de mostrar el nombre legible (ej. 'ProductKey 477' en vez del nombre
    del producto). Las claves son identificadores internos, nunca deben ser el
    resultado visible para el usuario."""
    final_segment = _get_final_select_segment(sql)
    m = _SELECT_COLS_RE.search(final_segment)
    if not m:
        return []

    raw_keys = []
    for item in _split_top_level_commas(m.group(1)):
        item = item.strip()
        if not item:
            continue
        expr = re.split(r'\s+AS\s+\w+\s*$', item, flags=re.IGNORECASE)[0].strip()
        match = _BARE_KEY_REF_RE.match(expr)
        if match:
            raw_keys.append(match.group(1))

    if not raw_keys:
        return []

    return [
        f"El SELECT final expone la(s) columna(s) clave {', '.join(raw_keys)} "
        f"directamente como resultado visible para el usuario — son identificadores "
        f"numéricos internos sin significado de negocio. Haz JOIN a la tabla de "
        f"dimensión correspondiente y muestra el nombre/atributo descriptivo en su "
        f"lugar (ej. EnglishProductName, o FirstName + ' ' + LastName), nunca la "
        f"columna *Key cruda."
    ]


def _sql_validation_problems(sql: str, question: str) -> list[str]:
    """Corre los 7 validadores semánticos de SQL (sin corregir, sin LLM) y
    devuelve la lista combinada de problemas detectados — fan-out de JOINs,
    estructura de crecimiento, filtro temporal, fuente de ventas incompleta o
    de más, columnas *Key expuestas, tablas de benchmark. Usado tanto por
    _validate_and_fix_sql (que sí corrige vía LLM) como por el chequeo de SQL
    cacheado (que solo necesita decidir si una entrada sigue siendo confiable,
    sin gastar una llamada al LLM en el camino rápido de caché)."""
    return [
        *_validate_sql_joins(sql),
        *_validate_growth_structure(sql),
        *_validate_temporal_filter(sql, question),
        *_validate_sales_source(sql, question),
        *_validate_channel_scope(sql, question),
        *_validate_no_raw_keys(sql),
        *_validate_no_excluded_tables(sql),
    ]


async def _validate_and_fix_sql(sql: str, question: str, extra_instructions: str = "") -> str:
    """Corre el mismo pipeline de validación + corrección semántica que usa
    TrackingSqlTool.execute() (fan-out de JOINs, fuente de ventas incompleta,
    tablas de benchmark, columnas *Key expuestas, etc.) — para que CUALQUIER
    SQL generado por LLM en TIARA pase por las mismas reglas, sin importar si
    se ejecuta vía TrackingSqlTool o directo (ej. STATS, PREDICTION).

    extra_instructions se reenvía al corrector junto con cada problema detectado
    — sin esto, un llamador con una regla adicional (ej. STATS exige datos A
    NIVEL DE FILA) la pierde en cuanto el corrector arregla un problema
    genérico (ej. fuente de ventas incompleta) sin saber de esa regla, y
    "resuelve" agregando con GROUP BY/SUM, rompiendo el análisis posterior.

    Lanza RuntimeError si tras 2 rondas de corrección el SQL sigue con
    problemas conocidos — preferible fallar la consulta a ejecutar SQL que
    sabemos roto (ej. el fan-out de DimGeography duplicando totales)."""
    if not question:
        return sql

    for _round in range(2):
        problems = _sql_validation_problems(sql, question)
        if not problems:
            return sql

        problem = "; ".join(problems)
        if extra_instructions:
            problem += f" Regla adicional a respetar al corregir: {extra_instructions}"
        logger.warning("SQL con problemas detectados: %s — solicitando corrección al LLM", problem)
        corrected = await _get_corrected_sql(question, sql, problem)
        if not corrected or corrected == sql:
            break
        logger.info("SQL corregido por LLM antes de ejecutar")
        sql = corrected

    final_problems = _sql_validation_problems(sql, question)
    if final_problems:
        logger.error(
            "SQL sigue con problemas tras agotar reintentos de corrección — bloqueando ejecución: %s",
            "; ".join(final_problems),
        )
        raise RuntimeError(
            "No pude generar una consulta SQL confiable para esta pregunta "
            "después de varios intentos de corrección. Por favor reformula "
            "la pregunta o intenta de nuevo."
        )
    return sql


# _run_stats (en agent_features.py) ejecuta SQL directo via SQL_RUNNER (sin pasar
# por TrackingSqlTool), así que debe aplicar las mismas validaciones de seguridad
# manualmente antes de correr cualquier query generada por el LLM.

async def _run_llm_generated_sql(sql: str):
    """Ejecuta SQL generado por LLM aplicando las mismas validaciones de
    seguridad que TrackingSqlTool.execute. Lanza ValueError si el SQL es inseguro."""
    if not state.SQL_RUNNER:
        raise RuntimeError("No hay conexión a la base de datos disponible.")
    safety_error = _check_sql_safety(sql)
    if safety_error:
        raise ValueError(safety_error)
    sanitized = _sanitize_sql_aliases(sql)
    return await state.SQL_RUNNER.run_sql(RunSqlToolArgs(sql=sanitized), None)


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


def _init_sql_cache(base_persist_dir: str):
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

    state.SQL_CACHE = client.get_or_create_collection(
        name="tiara_sql_cache",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine", "embedding_function": "openai"},
    )
    logger.info("SQL Cache inicializado (%d entradas)", state.SQL_CACHE.count())


def _search_sql_cache(question: str) -> Optional[dict]:
    if not state.SQL_CACHE:
        return None
    try:
        normalized = _normalize_question(question)

        results = state.SQL_CACHE.query(query_texts=[normalized], n_results=1)
        if (
            not results
            or not results.get("documents", [[]])[0]
            or not results.get("distances", [[]])[0]
            or not results.get("metadatas", [[]])[0]
        ):
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - (distance / 2.0)
        logger.info("Cache similarity: %.4f (threshold: %.2f)", similarity, state.SQL_CACHE_THRESHOLD)

        if similarity < state.SQL_CACHE_THRESHOLD:
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
    if not state.SQL_CACHE:
        return

    if any(f in sql.lower() for f in ["current_date", "now()", "getdate()"]):
        logger.info("SQL no cacheado: contiene función de fecha dinámica")
        return

    if full_response and len(full_response) > state.MAX_RESPONSE_CACHE_LEN:
        full_response = full_response[:state.MAX_RESPONSE_CACHE_LEN]

    normalized_question = _normalize_question(question)
    temporals = _extract_temporals(question)
    normalized_sql = _normalize_with_temporals(sql, temporals)

    try:
        state.SQL_CACHE.add(
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
    if not state.SQL_CACHE:
        return
    try:
        normalized = _normalize_question(question)
        results = state.SQL_CACHE.query(query_texts=[normalized], n_results=3)
        ids = (results.get("ids") or [[]])[0]
        if ids:
            state.SQL_CACHE.delete(ids=ids)
            logger.info("Cache purgado para '%s' (%d entradas eliminadas)", normalized, len(ids))
    except Exception:
        logger.exception("Error purgando cache")
