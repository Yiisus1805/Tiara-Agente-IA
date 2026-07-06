from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth import require_auth
from . import audit

router = APIRouter(prefix="/api/admin", dependencies=[Depends(require_auth)])


def _get_sql_cache():
    from .agent_state import SQL_CACHE
    return SQL_CACHE


def _get_schema_store():
    from .agent_state import SCHEMA_STORE
    return SCHEMA_STORE


# SQL Cache 

@router.get("/sql-cache")
async def list_sql_cache():
    cache = _get_sql_cache()
    if not cache:
        return {"entries": []}
    results = cache.get(include=["documents", "metadatas"])
    entries = [
        {
            "id":            results["ids"][i],
            "question":      (results["documents"] or [""])[i],
            "sql":           ((results["metadatas"] or [{}])[i] or {}).get("sql", ""),
            "full_response": ((results["metadatas"] or [{}])[i] or {}).get("full_response", ""),
        }
        for i in range(len(results.get("ids") or []))
    ]
    return {"entries": entries, "total": len(entries)}


@router.delete("/sql-cache/{entry_id}")
async def delete_sql_cache(entry_id: str):
    cache = _get_sql_cache()
    if not cache:
        raise HTTPException(status_code=503, detail="Cache no disponible")
    cache.delete(ids=[entry_id])
    return {"status": "deleted", "id": entry_id}


class SqlCacheEntry(BaseModel):
    question: str
    sql: str


@router.post("/sql-cache")
async def add_sql_cache(body: SqlCacheEntry):
    cache = _get_sql_cache()
    if not cache:
        raise HTTPException(status_code=503, detail="Cache no disponible")
    if not body.question.strip() or not body.sql.strip():
        raise HTTPException(status_code=400, detail="question y sql son requeridos")

    from .sql_pipeline import _normalize_question
    normalized = _normalize_question(body.question.strip())
    cache.add(
        ids=[str(uuid.uuid4())],
        documents=[normalized],
        metadatas=[{"sql": body.sql.strip(), "full_response": ""}],
    )
    return {"status": "added"}


# Schema Store 

@router.get("/schema-store")
async def list_schema_store():
    store = _get_schema_store()
    if not store:
        return {"entries": []}
    results = store.col.get(include=["documents", "metadatas"])
    entries = [
        {
            "id":   results["ids"][i],
            "doc":  (results["documents"] or [""])[i],
            "meta": (results["metadatas"] or [{}])[i] or {},
        }
        for i in range(len(results.get("ids") or []))
    ]
    return {"entries": entries, "total": len(entries)}


@router.delete("/schema-store/{entry_id}")
async def delete_schema_entry(entry_id: str):
    store = _get_schema_store()
    if not store:
        raise HTTPException(status_code=503, detail="Schema store no disponible")
    store.col.delete(ids=[entry_id])
    return {"status": "deleted", "id": entry_id}


# Logs & Auditoría 

@router.get("/logs/summary")
async def logs_summary():
    return audit.get_summary()


# Auth info 

@router.get("/me")
async def me(user: dict = Depends(require_auth)):
    return {"username": user.get("sub")}
