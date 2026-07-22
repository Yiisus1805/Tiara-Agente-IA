from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from threading import Lock


_DB_PATH = os.path.join(os.path.dirname(__file__), "audit.db")
_lock = Lock()

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id           TEXT PRIMARY KEY,
    timestamp    TEXT NOT NULL,
    username     TEXT NOT NULL,
    question     TEXT NOT NULL,
    intent       TEXT NOT NULL,
    duration_ms  INTEGER NOT NULL,
    is_retry     INTEGER NOT NULL DEFAULT 0,
    success      INTEGER NOT NULL DEFAULT 1
)
"""


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _init_db() -> None:
    with _conn() as con:
        con.execute(_CREATE_SQL)


_init_db()


def log(
    username: str,
    question: str,
    intent: str,
    duration_ms: int,
    is_retry: bool = False,
    success: bool = True,
) -> None:
    row_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    with _lock:
        with _conn() as con:
            con.execute(
                "INSERT INTO audit_log VALUES (?,?,?,?,?,?,?,?)",
                (row_id, ts, username, question, intent, duration_ms, int(is_retry), int(success)),
            )


def get_summary() -> dict:
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        avg_row = con.execute("SELECT AVG(duration_ms) FROM audit_log").fetchone()[0]
        avg_ms = round(avg_row) if avg_row else 0

        # PREDICTION se excluye del desglose: no es una funcionalidad real (el
        # agente no hace predicciones), esas preguntas solo reciben un rechazo.
        intent_rows = con.execute(
            "SELECT intent, COUNT(*) as cnt FROM audit_log "
            "WHERE intent != 'PREDICTION' GROUP BY intent ORDER BY cnt DESC"
        ).fetchall()
        intent_counts = {r["intent"]: r["cnt"] for r in intent_rows}

        success_count = con.execute(
            "SELECT COUNT(*) FROM audit_log WHERE success = 1"
        ).fetchone()[0]
        success_rate = round(success_count / total * 100) if total else 0

        top_intent = intent_rows[0]["intent"] if intent_rows else "—"

    return {
        "total": total,
        "avg_duration_ms": avg_ms,
        "intent_counts": intent_counts,
        "top_intent": top_intent,
        "success_rate": success_rate,
    }
