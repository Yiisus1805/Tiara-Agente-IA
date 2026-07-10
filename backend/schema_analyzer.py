"""
Analiza el esquema real de la base de datos y genera contexto dinámico para el agente.
Soporta múltiples dialectos: SQL Server (SQLSERVER_ODBC) y PostgreSQL (DATABASE_URL).
Para agregar otro motor, subclasear SchemaIntrospector e implementar los 5 métodos.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_NUMERIC_TYPES = frozenset({
    "int", "bigint", "smallint", "tinyint",
    "decimal", "numeric", "float", "real",
    "money", "smallmoney",
    # PostgreSQL equivalents
    "integer", "double precision", "real",
})
_DATE_TYPES = frozenset({
    "date", "datetime", "datetime2", "smalldatetime", "datetimeoffset",
    # PostgreSQL
    "timestamp", "timestamp without time zone", "timestamp with time zone",
})
_TEXT_TYPES = frozenset({
    "varchar", "nvarchar", "char", "nchar", "text", "ntext",
    # PostgreSQL
    "character varying", "character", "bpchar",
})

_SQL_RESERVED = frozenset({
    "IS", "AS", "IN", "ON", "BY", "OR", "AND", "NOT", "TO", "AT", "GO", "IF", "DO",
    "CURRENT", "PREVIOUS", "NEXT", "KEY", "SET", "VALUE", "USER", "TABLE", "VIEW",
    "INDEX", "ORDER", "GROUP", "SELECT", "WHERE", "FROM", "JOIN", "CASE", "WHEN",
    "ALL", "WITH", "TOP", "INTO", "OVER", "FOR", "ROW", "END", "NEW", "OLD", "NO",
})

# Indicadores de que una tabla contiene ventas agregables
_SALES_INDICATOR_COLS = {"SalesAmount", "OrderDateKey", "SalesTerritoryKey", "ProductKey"}
# Sufijos de tablas auxiliares que nunca deben usarse en UNION ALL
_EXCLUDED_SUFFIXES = ("XL_CCI", "XL_PageCompressed", "_CCI", "_PageCompressed")


# ── Alias helper ─────────────────────────────────────────────────────────────

def _make_alias(table_name: str, used: set[str]) -> str:
    """Genera un alias corto y seguro de 2-4 caracteres a partir del nombre de la tabla."""
    name = table_name
    prefix_char = ""
    for prefix in ("Fact", "Dim", "Bridge"):
        if name.startswith(prefix) and len(name) > len(prefix):
            prefix_char = prefix[0].upper()
            name = name[len(prefix):]
            break

    words = re.findall(r'[A-Z][a-z0-9]*', name) or [name[:1].upper()]

    if len(words) == 1:
        alias = (prefix_char + words[0][0].upper()) if prefix_char else words[0][:4].upper()
    else:
        initials = "".join(w[0].upper() for w in words)
        alias = (prefix_char + initials)[:4] if prefix_char else initials[:4]

    if not alias or alias in _SQL_RESERVED:
        alias = (name[:3] if len(name) >= 3 else name * 2).upper()[:3]

    base = alias
    n = 2
    while alias in used or alias in _SQL_RESERVED:
        alias = base[:3] + str(n)
        n += 1

    used.add(alias)
    return alias


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TableMeta:
    schema: str
    name: str
    table_type: str   # "Fact" | "Dimension" | "Bridge" | "Table" | "View"
    columns: list     # [(col_name, col_type, max_length, is_nullable), ...]
    pks: list[str]
    fk_refs: list     # [(col, ref_schema, ref_table, ref_col), ...]
    alias: str = ""

    @property
    def col_names(self) -> set[str]:
        return {c[0] for c in self.columns}


@dataclass
class SchemaMetadata:
    tables: dict[str, TableMeta] = field(default_factory=dict)
    schema_hash: str = ""
    dialect: str = "mssql"          # "mssql" | "postgresql"

    fact_tables: list[str] = field(default_factory=list)
    dim_tables:  list[str] = field(default_factory=list)
    view_tables: list[str] = field(default_factory=list)

    date_table:    Optional[str] = None
    date_schema:   str = "dbo"
    date_alias:    str = "DD"
    date_year_col: str = "CalendarYear"
    date_key_col:  str = "DateKey"

    union_fact_tables:  list[str] = field(default_factory=list)
    union_common_cols:  list[str] = field(default_factory=list)

    employee_dim:  Optional[str] = None
    employee_fact: Optional[str] = None
    customer_dim:  Optional[str] = None
    customer_fact: Optional[str] = None

    aliases: dict[str, str] = field(default_factory=dict)

    def pinned_tables(self) -> set[str]:
        """Tablas que siempre deben incluirse en el contexto RAG."""
        pinned: set[str] = set()
        for t in self.union_fact_tables:
            tbl = self.tables.get(t)
            schema = tbl.schema if tbl else "dbo"
            pinned.add(f"{schema}.{t}")
        return pinned


# ── Introspector abstracto ────────────────────────────────────────────────────

class SchemaIntrospector(ABC):
    """
    Encapsula las queries de introspección específicas de cada motor de base de datos.
    Para soportar un nuevo motor: subclasear e implementar los 5 métodos abstractos.
    """

    @abstractmethod
    def compute_hash(self, cursor) -> str:
        """Hash estable de (schema, tabla, columna, tipo) para detectar cambios."""

    @abstractmethod
    def fetch_tables(self, cursor) -> list:
        """Retorna lista de (schema_name, table_name, kind) de tablas y vistas de usuario.
        kind es 'BASE TABLE' o 'VIEW'."""

    @abstractmethod
    def fetch_columns(self, cursor) -> dict[tuple, list]:
        """Retorna {(schema, table): [(col_name, col_type, max_length, is_nullable), ...]}."""

    @abstractmethod
    def fetch_pks(self, cursor) -> dict[tuple, list]:
        """Retorna {(schema, table): [pk_col_name, ...]}."""

    @abstractmethod
    def fetch_fks(self, cursor) -> dict[tuple, list]:
        """Retorna {(parent_schema, parent_table): [(col, ref_schema, ref_table, ref_col), ...]}."""


# ── SQL Server ────────────────────────────────────────────────────────────────

class SqlServerIntrospector(SchemaIntrospector):
    """Introspección via sys.* (SQL Server / Azure SQL)."""

    def compute_hash(self, cursor) -> str:
        cursor.execute("""
        SELECT s.name, o.name, c.name, ty.name
        FROM sys.columns c
        JOIN sys.types   ty ON c.user_type_id  = ty.user_type_id
        JOIN sys.objects o  ON c.object_id     = o.object_id AND o.type IN ('U', 'V')
        JOIN sys.schemas s  ON o.schema_id     = s.schema_id
        ORDER BY s.name, o.name, c.column_id
        """)
        rows = cursor.fetchall()
        raw = json.dumps([[r[0], r[1], r[2], r[3]] for r in rows])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def fetch_tables(self, cursor) -> list:
        cursor.execute("""
        SELECT s.name, t.name, 'BASE TABLE'
        FROM sys.tables t
        JOIN sys.schemas s ON t.schema_id = s.schema_id
        UNION ALL
        SELECT s.name, v.name, 'VIEW'
        FROM sys.views v
        JOIN sys.schemas s ON v.schema_id = s.schema_id
        ORDER BY 1, 2
        """)
        return cursor.fetchall()

    def fetch_columns(self, cursor) -> dict[tuple, list]:
        cursor.execute("""
        SELECT s.name, o.name, c.name, ty.name, c.max_length, c.is_nullable
        FROM sys.columns c
        JOIN sys.types   ty ON c.user_type_id  = ty.user_type_id
        JOIN sys.objects o  ON c.object_id     = o.object_id AND o.type IN ('U', 'V')
        JOIN sys.schemas s  ON o.schema_id     = s.schema_id
        ORDER BY s.name, o.name, c.column_id
        """)
        result: dict[tuple, list] = {}
        for row in cursor.fetchall():
            result.setdefault((row[0], row[1]), []).append(
                (row[2], row[3], row[4], row[5])
            )
        return result

    def fetch_pks(self, cursor) -> dict[tuple, list]:
        cursor.execute("""
        SELECT s.name, t.name, c.name
        FROM sys.indexes i
        JOIN sys.index_columns ic ON i.object_id  = ic.object_id
                                 AND i.index_id   = ic.index_id
        JOIN sys.columns c        ON ic.object_id = c.object_id
                                 AND ic.column_id = c.column_id
        JOIN sys.tables  t        ON i.object_id  = t.object_id
        JOIN sys.schemas s        ON t.schema_id  = s.schema_id
        WHERE i.is_primary_key = 1
        """)
        result: dict[tuple, list] = {}
        for row in cursor.fetchall():
            result.setdefault((row[0], row[1]), []).append(row[2])
        return result

    def fetch_fks(self, cursor) -> dict[tuple, list]:
        cursor.execute("""
        SELECT ps.name, pt.name, pc.name, rs.name, rt.name, rc.name
        FROM sys.foreign_key_columns fk
        JOIN sys.tables  pt ON fk.parent_object_id     = pt.object_id
        JOIN sys.columns pc ON fk.parent_object_id     = pc.object_id
                           AND fk.parent_column_id     = pc.column_id
        JOIN sys.tables  rt ON fk.referenced_object_id = rt.object_id
        JOIN sys.columns rc ON fk.referenced_object_id = rc.object_id
                           AND fk.referenced_column_id = rc.column_id
        JOIN sys.schemas ps ON pt.schema_id = ps.schema_id
        JOIN sys.schemas rs ON rt.schema_id = rs.schema_id
        """)
        result: dict[tuple, list] = {}
        for row in cursor.fetchall():
            result.setdefault((row[0], row[1]), []).append(
                (row[2], row[3], row[4], row[5])
            )
        return result


# ── PostgreSQL ────────────────────────────────────────────────────────────────

class PostgresIntrospector(SchemaIntrospector):
    """Introspección via information_schema + pg_catalog (PostgreSQL)."""

    _EXCLUDED_SCHEMAS = "('pg_catalog', 'information_schema', 'pg_toast')"

    def compute_hash(self, cursor) -> str:
        cursor.execute(f"""
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN {self._EXCLUDED_SCHEMAS}
        ORDER BY table_schema, table_name, ordinal_position
        """)
        rows = cursor.fetchall()
        raw = json.dumps([[r[0], r[1], r[2], r[3]] for r in rows])
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def fetch_tables(self, cursor) -> list:
        cursor.execute(f"""
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_type IN ('BASE TABLE', 'VIEW')
          AND table_schema NOT IN {self._EXCLUDED_SCHEMAS}
        ORDER BY table_schema, table_name
        """)
        return cursor.fetchall()

    def fetch_columns(self, cursor) -> dict[tuple, list]:
        cursor.execute(f"""
        SELECT table_schema, table_name, column_name, data_type,
               COALESCE(character_maximum_length, -1),
               CASE WHEN is_nullable = 'YES' THEN 1 ELSE 0 END
        FROM information_schema.columns
        WHERE table_schema NOT IN {self._EXCLUDED_SCHEMAS}
        ORDER BY table_schema, table_name, ordinal_position
        """)
        result: dict[tuple, list] = {}
        for row in cursor.fetchall():
            result.setdefault((row[0], row[1]), []).append(
                (row[2], row[3], row[4], row[5])
            )
        return result

    def fetch_pks(self, cursor) -> dict[tuple, list]:
        cursor.execute(f"""
        SELECT kcu.table_schema, kcu.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema    = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND kcu.table_schema NOT IN {self._EXCLUDED_SCHEMAS}
        ORDER BY kcu.table_schema, kcu.table_name, kcu.ordinal_position
        """)
        result: dict[tuple, list] = {}
        for row in cursor.fetchall():
            result.setdefault((row[0], row[1]), []).append(row[2])
        return result

    def fetch_fks(self, cursor) -> dict[tuple, list]:
        cursor.execute(f"""
        SELECT
            kcu.table_schema, kcu.table_name, kcu.column_name,
            ccu.table_schema, ccu.table_name, ccu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema    = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema    = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND kcu.table_schema NOT IN {self._EXCLUDED_SCHEMAS}
        """)
        result: dict[tuple, list] = {}
        for row in cursor.fetchall():
            result.setdefault((row[0], row[1]), []).append(
                (row[2], row[3], row[4], row[5])
            )
        return result


# ── Factory ───────────────────────────────────────────────────────────────────

def _detect_dialect() -> str:
    """Detecta el dialecto desde DATABASE_URL o SQLSERVER_ODBC."""
    db_url = os.getenv("DATABASE_URL", "")
    if "postgresql" in db_url or "postgres://" in db_url:
        return "postgresql"
    return "mssql"


def get_introspector() -> SchemaIntrospector:
    """Retorna el introspector correcto para el motor configurado en .env."""
    dialect = _detect_dialect()
    if dialect == "postgresql":
        return PostgresIntrospector()
    return SqlServerIntrospector()


def _get_connection():
    """Abre una conexión nativa al motor configurado en .env."""
    dialect = _detect_dialect()
    if dialect == "postgresql":
        db_url = os.getenv("DATABASE_URL", "")
        # psycopg2 no acepta el prefijo +psycopg2
        clean = db_url.replace("postgresql+psycopg2://", "postgresql://")
        try:
            import psycopg2
            return psycopg2.connect(clean)
        except ImportError:
            raise RuntimeError(
                "psycopg2 no instalado. Ejecuta: pip install psycopg2-binary"
            )
    # SQL Server via pyodbc
    odbc = os.getenv("SQLSERVER_ODBC")
    if not odbc:
        raise RuntimeError(
            "Configura SQLSERVER_ODBC (SQL Server) o DATABASE_URL (PostgreSQL) en .env"
        )
    import pyodbc
    return pyodbc.connect(odbc)


# ── API pública ───────────────────────────────────────────────────────────────

def analyze_schema(cursor=None) -> SchemaMetadata:
    """
    Consulta la BD activa y construye un SchemaMetadata con análisis relacional completo.
    Si no se provee cursor, abre su propia conexión.
    """
    intro = get_introspector()
    dialect = _detect_dialect()

    own_conn = cursor is None
    conn = None
    if own_conn:
        conn = _get_connection()
        cursor = conn.cursor()

    try:
        meta = SchemaMetadata(dialect=dialect)
        meta.schema_hash = intro.compute_hash(cursor)

        table_rows = intro.fetch_tables(cursor)
        all_cols   = intro.fetch_columns(cursor)
        all_pks    = intro.fetch_pks(cursor)
        all_fks    = intro.fetch_fks(cursor)

        # Construir TableMeta
        aliases_used: set[str] = set()
        for p_schema, p_table, p_kind in table_rows:
            key     = (p_schema, p_table)
            columns = all_cols.get(key, [])
            pks     = all_pks.get(key, [])
            fk_refs = all_fks.get(key, [])

            nl = p_table.lower()
            if p_kind == "VIEW":
                ttype = "View"
            elif nl.startswith("fact"):
                ttype = "Fact"
            elif nl.startswith("dim"):
                ttype = "Dimension"
            elif "bridge" in nl:
                ttype = "Bridge"
            else:
                ttype = "Table"

            alias = _make_alias(p_table, aliases_used)
            tm = TableMeta(
                schema=p_schema, name=p_table, table_type=ttype,
                columns=columns, pks=pks, fk_refs=fk_refs, alias=alias,
            )
            meta.tables[p_table] = tm
            meta.aliases[p_table] = alias

        meta.fact_tables = [n for n, t in meta.tables.items() if t.table_type == "Fact"]
        meta.dim_tables  = [n for n, t in meta.tables.items() if t.table_type == "Dimension"]
        meta.view_tables = [n for n, t in meta.tables.items() if t.table_type == "View"]

        # Detectar tabla de tiempo
        for n in meta.dim_tables:
            t = meta.tables[n]
            cols = t.col_names
            if "CalendarYear" in cols and ("DateKey" in cols or "FullDateAlternateKey" in cols):
                meta.date_table    = n
                meta.date_schema   = t.schema
                meta.date_alias    = t.alias
                meta.date_year_col = "CalendarYear"
                meta.date_key_col  = "DateKey" if "DateKey" in cols else "FullDateAlternateKey"
                break

        # Detectar tablas Fact combinables para UNION ALL
        main_facts = [
            n for n in meta.fact_tables
            if (
                len(_SALES_INDICATOR_COLS.intersection(meta.tables[n].col_names)) >= 3
                and not any(n.endswith(s) for s in _EXCLUDED_SUFFIXES)
            )
        ]
        meta.union_fact_tables = main_facts

        if main_facts:
            common: set[str] | None = None
            for n in main_facts:
                cols = meta.tables[n].col_names
                common = cols if common is None else common.intersection(cols)
            ORDERED = [
                "OrderDateKey", "SalesTerritoryKey", "ProductKey", "CustomerKey",
                "EmployeeKey", "SalesAmount", "TotalProductCost", "UnitPrice", "OrderQuantity",
            ]
            meta.union_common_cols = [c for c in ORDERED if c in (common or set())]

        # Detectar tablas de empleados y clientes
        for n in meta.dim_tables:
            t = meta.tables[n]
            cols = t.col_names
            if not meta.employee_dim and (
                "HireDate" in cols
                or "SalesPersonFlag" in cols
                or ("EmployeeKey" in cols and "FirstName" in cols)
            ):
                meta.employee_dim = n
            elif not meta.customer_dim and (
                "YearlyIncome" in cols
                or ("CustomerKey" in cols and "FirstName" in cols)
            ):
                meta.customer_dim = n

        for n in (meta.union_fact_tables or meta.fact_tables):
            t = meta.tables[n]
            if meta.employee_dim and not meta.employee_fact:
                emp = meta.tables[meta.employee_dim]
                emp_key = emp.pks[0] if emp.pks else "EmployeeKey"
                if emp_key in t.col_names:
                    meta.employee_fact = n
            if meta.customer_dim and not meta.customer_fact:
                cust = meta.tables[meta.customer_dim]
                cust_key = cust.pks[0] if cust.pks else "CustomerKey"
                if cust_key in t.col_names:
                    meta.customer_fact = n

        logger.info(
            "Schema analizado (%s): %d tablas, %d fact, %d dim | "
            "fecha=%s(%s) | union_facts=%s | emp=%s/%s cust=%s/%s",
            dialect,
            len(meta.tables), len(meta.fact_tables), len(meta.dim_tables),
            meta.date_table, meta.date_alias, meta.union_fact_tables,
            meta.employee_dim, meta.employee_fact,
            meta.customer_dim, meta.customer_fact,
        )
        return meta

    finally:
        if own_conn and conn:
            conn.close()


# ── Generador de prompt dinámico ──────────────────────────────────────────────

def build_dynamic_prompt_sections(meta: SchemaMetadata) -> dict[str, str]:
    """
    Genera las secciones dinámicas del prompt del agente a partir del esquema real.
    Retorna un dict con claves: aliases, columns, date_rules, sales_source, persons, syntax.
    """
    # ── Sintaxis SQL según dialecto ────────────────────────────────────────────
    if meta.dialect == "postgresql":
        syntax_section = (
            "SINTAXIS SQL (PostgreSQL):\n"
            "- Paginación: LIMIT N (NO uses TOP N)\n"
            "- Fecha actual: NOW() o CURRENT_TIMESTAMP (NO GETDATE())\n"
            "- Nulos: COALESCE(col, valor) (NO ISNULL)\n"
            "- Identificadores especiales: comillas dobles \"col\" (NO corchetes [col])\n"
            "- Concatenación: col || ' ' || col2  o  CONCAT(col, ' ', col2)\n"
            "- Esquema por defecto: public (si no se especifica otro)\n"
        )
    else:
        syntax_section = (
            "SINTAXIS SQL (SQL Server / T-SQL):\n"
            "- Paginación: TOP N (NO uses LIMIT)\n"
            "- Fecha actual: NUNCA uses GETDATE() — los datos son históricos\n"
            "- Nulos: ISNULL(col, valor) o COALESCE(col, valor)\n"
            "- Identificadores especiales: corchetes [col] si son palabras reservadas\n"
            "- Concatenación: col + ' ' + col2  o  CONCAT(col, ' ', col2)\n"
            "- Esquema por defecto: dbo\n"
        )

    # ── Aliases ────────────────────────────────────────────────────────────────
    alias_lines = ["Usa SIEMPRE estos alias seguros para las tablas:"]
    for tname in meta.fact_tables + meta.dim_tables:
        t = meta.tables.get(tname)
        if t:
            alias_lines.append(f"  {tname:<38} → {t.alias}")
    if meta.view_tables:
        alias_lines.append("Vistas disponibles (VIEW, ya combinan datos de varias tablas):")
        for tname in meta.view_tables:
            t = meta.tables.get(tname)
            if t:
                alias_lines.append(f"  {tname:<38} → {t.alias}")
        alias_lines.append(
            "Si una vista ya contiene las columnas necesarias para responder la pregunta, "
            "prefiérela sobre reconstruir el JOIN manualmente con las tablas base."
        )
    aliases_section = "\n".join(alias_lines)

    # ── Columnas ───────────────────────────────────────────────────────────────
    col_hints: list[str] = []
    for tname, t in meta.tables.items():
        cols = t.col_names
        english_cols = sorted(c for c in cols if c.startswith("English") and "Name" in c)
        for ecol in english_cols:
            col_hints.append(
                f"- {tname}: usa {ecol} "
                f"(existen versiones Spanish/French; el inglés es el estándar)."
            )
        if "FirstName" in cols and "LastName" in cols:
            col_hints.append(
                f"- {tname}: combina FirstName + ' ' + LastName AS Nombre "
                f"para mostrar personas (no exponer columnas *Key)."
            )
        if not english_cols:
            name_cols = [
                c for c in cols
                if c.endswith("Name")
                and not any(c.startswith(p) for p in ("English", "Spanish", "French"))
            ]
            if name_cols:
                col_hints.append(
                    f"- {tname}: usa {name_cols[0]} como etiqueta visible en SELECT."
                )
    col_hints.append(
        "REGLA GENERAL: en el SELECT final NUNCA expongas columnas *Key como resultado visible — "
        "los *Key son solo para JOIN."
    )
    columns_section = "NOMBRES DE COLUMNAS (derivados del esquema real):\n" + "\n".join(col_hints)

    # ── Reglas de fechas ───────────────────────────────────────────────────────
    if meta.date_table:
        dta = meta.date_alias
        yc  = meta.date_year_col
        dk  = meta.date_key_col
        ds  = meta.date_schema
        date_rules = (
            f"AÑOS Y FECHAS (tabla de tiempo detectada: {meta.date_table} alias {dta}):\n"
            f"NUNCA uses funciones de fecha del sistema — los datos tienen un rango histórico fijo.\n"
            f"- Contexto de conversación: si la pregunta actual no menciona año pero el historial sí,\n"
            f"  aplica ese mismo período.\n"
            f"- Para filtrar por año: JOIN {ds}.{meta.date_table} {dta} "
            f"ON <Fact>.OrderDateKey = {dta}.{dk}\n"
            f"  luego WHERE {dta}.{yc} IN (...)\n"
            f"- Si la pregunta NO especifica año: "
            f"WHERE {dta}.{yc} IN (SELECT DISTINCT {yc} FROM {ds}.{meta.date_table})\n"
            f"- Para 'último año disponible': (SELECT MAX({yc}) FROM {ds}.{meta.date_table})\n"
            f"- Para comparaciones año a año: LAG() OVER (ORDER BY {dta}.{yc})\n"
            f"- Si el usuario menciona un año que puede no existir en los datos:\n"
            f"    WHERE {dta}.{yc} = (SELECT MAX({yc}) FROM {ds}.{meta.date_table})\n"
            f"  y menciona en tu respuesta que usas el último año con datos disponibles.\n"
        )
    else:
        date_rules = (
            "AÑOS Y FECHAS:\n"
            "NUNCA uses funciones de fecha del sistema — los datos tienen un rango histórico fijo.\n"
            "Filtra por columnas de fecha disponibles en las tablas del esquema.\n"
        )

    # ── Fuente de ventas totales (UNION ALL) ───────────────────────────────────
    if len(meta.union_fact_tables) >= 2:
        cols = meta.union_common_cols or ["SalesAmount", "OrderDateKey"]
        cols_str = ",\n             ".join(cols)

        union_blocks: list[str] = []
        for i, n in enumerate(meta.union_fact_tables):
            t = meta.tables[n]
            prefix = "      UNION ALL\n      " if i > 0 else "      "
            union_blocks.append(f"{prefix}SELECT {cols_str}\n      FROM {t.schema}.{n}")

        internet_like = [n for n in meta.union_fact_tables if "internet" in n.lower()]
        reseller_like = [n for n in meta.union_fact_tables
                         if "reseller" in n.lower() or "distributor" in n.lower()]
        exceptions: list[str] = []
        if internet_like:
            exceptions.append(
                f"Excepción: usa SOLO {internet_like[0]} "
                f"si el usuario dice 'online', 'internet' o 'canal directo'."
            )
        if reseller_like:
            exceptions.append(
                f"Excepción: usa SOLO {reseller_like[0]} "
                f"si el usuario dice 'reseller', 'distribuidor' o 'canal indirecto'."
            )

        excluded = [n for n in meta.fact_tables
                    if any(n.endswith(s) for s in _EXCLUDED_SUFFIXES)]
        if excluded:
            exceptions.append(f"NUNCA uses: {', '.join(excluded)}.")

        cust_only = [n for n in meta.union_fact_tables
                     if "CustomerKey" in meta.tables[n].col_names]
        cust_note = ""
        if cust_only and len(cust_only) < len(meta.union_fact_tables):
            cust_note = (
                f"\nNOTA: CustomerKey solo existe en {', '.join(cust_only)}. "
                "Para consultas de clientes no incluyas en el UNION ALL tablas que no tengan CustomerKey.\n"
            )

        sales_source = (
            f"FUENTE DE VENTAS TOTALES ({' + '.join(meta.union_fact_tables)}):\n"
            f"Para totales globales SIEMPRE combina con UNION ALL:\n"
            f"  WITH AllSales AS (\n"
            f"{''.join(union_blocks)}\n"
            f"  )\n"
            f"REGLA CRÍTICA: incluye en el UNION ALL todas las columnas que necesitarás para JOINs posteriores.\n"
            f"JOIN correcto (usa clave de AllSales, no de la dimensión):\n"
            f"  JOIN dbo.DimXxx D ON AllSales.XxxKey = D.XxxKey  ← CORRECTO\n"
            f"  JOIN dbo.DimXxx D ON D.XxxKey IN (subquery)      ← INCORRECTO\n"
            f"GROUP BY: solo por columnas de dimensión (nunca por columnas clave de AllSales).\n"
            + ("\n".join(exceptions) + "\n" if exceptions else "")
            + cust_note
        )
    elif len(meta.union_fact_tables) == 1:
        n = meta.union_fact_tables[0]
        t = meta.tables[n]
        sales_source = f"FUENTE DE VENTAS: tabla principal es {t.schema}.{n}.\n"
    else:
        sales_source = ""

    # ── Personas (empleados / clientes) ────────────────────────────────────────
    person_lines: list[str] = []
    if meta.employee_dim and meta.employee_fact:
        et  = meta.tables[meta.employee_dim]
        eft = meta.tables[meta.employee_fact]
        emp_pk = et.pks[0] if et.pks else "EmployeeKey"
        person_lines.append(
            f"VENDEDOR/EMPLEADO (quien hace la venta): "
            f"usa {meta.employee_dim} ({et.alias}) + {meta.employee_fact} ({eft.alias}) "
            f"ON {eft.alias}.{emp_pk} = {et.alias}.{emp_pk}\n"
            f"  Filtra por: WHERE {et.alias}.FirstName = 'X' AND {et.alias}.LastName = 'Y'"
        )
    if meta.customer_dim and meta.customer_fact:
        ct  = meta.tables[meta.customer_dim]
        cft = meta.tables[meta.customer_fact]
        cust_pk = ct.pks[0] if ct.pks else "CustomerKey"
        person_lines.append(
            f"CLIENTE (quien compra): "
            f"usa {meta.customer_dim} ({ct.alias}) + {meta.customer_fact} ({cft.alias}) "
            f"ON {cft.alias}.{cust_pk} = {ct.alias}.{cust_pk}\n"
            f"  Filtra por: WHERE {ct.alias}.FirstName = 'X' AND {ct.alias}.LastName = 'Y'"
        )
    if meta.employee_dim and meta.customer_dim:
        person_lines.append(
            f"NUNCA cruces {meta.employee_dim} con {meta.customer_dim} "
            f"ni uses la tabla de empleados para buscar clientes ni viceversa."
        )

    verb_hint = ""
    if meta.employee_dim and meta.customer_dim:
        ed  = meta.employee_dim
        ef  = meta.employee_fact or ""
        cd  = meta.customer_dim
        cf  = meta.customer_fact or ""
        verb_hint = (
            f"SEÑALES DE CONTEXTO PARA ELEGIR TABLA:\n"
            f"  'vendió', 'realizó ventas', 'cuánto vendió', 'ventas de [nombre]' "
            f"→ VENDEDOR: usar {ed} + {ef}\n"
            f"  'compró', 'gastó', 'cuánto gastó', 'cuánto pagó', 'cuánto compró', "
            f"'cliente [nombre]' → CLIENTE: usar {cd} + {cf}\n"
            f"  NUNCA usar {cd} cuando la pregunta habla de ventas realizadas por alguien.\n"
        )

    persons_section = (
        "NOMBRES DE PERSONAS:\n"
        + verb_hint
        + "\n".join(person_lines)
        if person_lines else ""
    )

    return {
        "aliases":      aliases_section,
        "columns":      columns_section,
        "date_rules":   date_rules,
        "sales_source": sales_source,
        "persons":      persons_section,
        "syntax":       syntax_section,
    }
