from __future__ import annotations

import asyncio
import sys
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

        self.engine = create_engine(
            "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_conn_str),
            poolclass=NullPool,
            future=True,
        )

        # Smoke test
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

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

    def _is_allowed(self, sql: str) -> bool:
        if not sql:
            return False
        lower = sql.lower().lstrip()

        blocked_starts = (
            "insert", "update", "delete", "drop", "alter", "create",
            "truncate", "exec", "execute", "merge", "grant", "revoke"
        )
        if lower.startswith(blocked_starts):
            return False

        return lower.startswith("select") or lower.startswith("with")

    async def run_sql(self, args: RunSqlToolArgs, context: ToolContext) -> pd.DataFrame:
        sql = self._normalize_sql(args.sql)
        if not self._is_allowed(sql):
            raise ValueError("Solo consultas SELECT (y WITH ... SELECT) permitidas")

        def _query() -> pd.DataFrame:
            with self.engine.connect() as conn:
                return pd.read_sql(text(sql), conn)

        return await asyncio.to_thread(_query)

