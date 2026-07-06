from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from threading import Lock

SHARE_TTL = timedelta(hours=2)

_DB_PATH = os.path.join(os.path.dirname(__file__), "share.db")
_lock = Lock()

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS shared_results (
    id            TEXT PRIMARY KEY,
    created_by    TEXT NOT NULL,
    question      TEXT NOT NULL,
    html_content  TEXT,
    narrative     TEXT,
    chart_data    TEXT,
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL
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


def _purge_expired(con: sqlite3.Connection) -> None:
    now = datetime.now(timezone.utc).isoformat()
    con.execute("DELETE FROM shared_results WHERE expires_at < ?", (now,))


def create_share(
    created_by: str,
    question: str,
    html_content: str = "",
    narrative: str = "",
    chart_data: dict | None = None,
) -> dict:
    share_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expires_at = now + SHARE_TTL

    with _lock:
        with _conn() as con:
            _purge_expired(con)
            con.execute(
                "INSERT INTO shared_results VALUES (?,?,?,?,?,?,?,?)",
                (
                    share_id,
                    created_by,
                    question,
                    html_content or "",
                    narrative or "",
                    json.dumps(chart_data) if chart_data else "",
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )

    return {"share_id": share_id, "created_at": now.isoformat(), "expires_at": expires_at.isoformat()}


def get_share(share_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM shared_results WHERE id = ?", (share_id,)
        ).fetchone()

    if not row:
        return None

    expires_at = datetime.fromisoformat(row["expires_at"])
    if expires_at < datetime.now(timezone.utc):
        return None

    return {
        "question": row["question"],
        "html_content": row["html_content"] or "",
        "narrative": row["narrative"] or "",
        "chart_data": json.loads(row["chart_data"]) if row["chart_data"] else None,
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
    }
