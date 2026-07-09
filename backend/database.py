from __future__ import annotations

import asyncio
import re
import sys
import time
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from vanna.capabilities.sql_runner import SqlRunner, RunSqlToolArgs
from vanna.core.tool import ToolContext


class SqlServerRunner(SqlRunner):
    def __init__(self, odbc_conn_str: str):
        if not odbc_conn_str:
            raise ValueError("ODBC connection string vacío")

        if "Encrypt=" not in odbc_conn_str:
            odbc_conn_str += ";Encrypt=no"

        if "Connect Timeout=" not in odbc_conn_str and "Connection Timeout=" not in odbc_conn_str:
            odbc_conn_str += ";Connect Timeout=30"

        self.engine = create_engine(
            "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_conn_str),
            poolclass=NullPool,
            future=True,
        )

        # Smoke test con reintentos: en el primer arranque en Render, la red o
        # la base de datos pueden tardar unos segundos en estar disponibles.
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                break
            except Exception:
                if attempt == max_attempts:
                    raise
                wait = 2 ** attempt
                print(
                    f"[SQL] Intento {attempt}/{max_attempts} de conexión falló, reintentando en {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)

        print("[SQL] SqlServerRunner inicializado correctamente", file=sys.stderr)

    def _normalize_sql(self, sql: str) -> str:
        sql = (sql or "").strip().rstrip(";")
        if not sql:
            return sql

        lower = sql.lower().strip()

        if lower.startswith("show tables"):
            return """
            SELECT s.name AS schema_name, t.name AS table_name
            FROM sys.tables t
            JOIN sys.schemas s ON s.schema_id = t.schema_id
            ORDER BY s.name, t.name
            """.strip()

        if lower.startswith("show columns from "):
            table = sql.split("from", 1)[1].strip().strip("`[]")
            return f"""
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table}'
            ORDER BY ORDINAL_POSITION
            """.strip()

        if " limit " in lower:
            try:
                before, after = sql.rsplit("LIMIT", 1)
                n = int(after.strip())
                before = before.strip()
                if before.lower().startswith("select "):
                    return f"SELECT TOP {n} {before[7:].strip()}"
            except Exception:
                pass

        return sql

    _BLOCKED_STARTS = (
        "insert", "update", "delete", "drop", "alter", "create",
        "truncate", "exec", "execute", "merge", "grant", "revoke",
    )
    _BLOCKED_INLINE = {
        "xp_cmdshell", "sp_executesql", "openrowset", "opendatasource",
        "bulk insert", "sys.server_principals", "sys.credentials",
        "sys.login_token", "sys.sql_logins", "sys.asymmetric_keys",
    }
    _STACKED_RE = re.compile(
        r";\s*(insert|update|delete|drop|alter|create|truncate|exec|execute|merge|grant|revoke)\b",
        re.IGNORECASE,
    )

    def _is_allowed(self, sql: str) -> tuple[bool, str]:
        if not sql:
            return False, "SQL vacío"
        lower = sql.lower().lstrip()

        if lower.startswith(self._BLOCKED_STARTS):
            return False, "Solo consultas SELECT están permitidas"

        if not (lower.startswith("select") or lower.startswith("with")):
            return False, "Solo consultas SELECT están permitidas"

        if self._STACKED_RE.search(sql):
            return False, "Consultas apiladas (stacked queries) no están permitidas"

        if any(kw in lower for kw in self._BLOCKED_INLINE):
            return False, "La consulta contiene funciones o tablas del sistema no permitidas"

        return True, ""

    async def run_sql(self, args: RunSqlToolArgs, context: ToolContext) -> pd.DataFrame:
        sql = self._normalize_sql(args.sql)
        allowed, reason = self._is_allowed(sql)
        if not allowed:
            raise ValueError(reason)

        def _query() -> pd.DataFrame:
            with self.engine.connect() as conn:
                return pd.read_sql(text(sql), conn)

        try:
            return await asyncio.wait_for(asyncio.to_thread(_query), timeout=75.0)
        except asyncio.TimeoutError:
            raise ValueError("La consulta tardó demasiado (>75s). Intenta reformular la pregunta.")

