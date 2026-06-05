from __future__ import annotations

import json
import os
import traceback

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

from vanna.core.user import RequestContext

from .agent_logic import build_agent, run_agent_stream_text, CHART_SENTINEL, ERROR_RETRY_SENTINEL, TABLE_FLUSH_SENTINEL
from .auth import check_credentials, create_token, require_auth
from .admin import router as admin_router


agent = build_agent()

app = FastAPI(title="TIARA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def build_request_context(request: Request) -> RequestContext:
    return RequestContext(
        headers=dict(request.headers),
        cookies=dict(request.cookies),
        remote_addr=request.client.host if request.client else None,
    )


# ── Auth ──────────────────────────────────────────────────────────────────────

class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def login(body: LoginBody):
    if not check_credentials(body.username, body.password):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    token = create_token(body.username)
    return {"access_token": token, "token_type": "bearer"}


# ── Páginas ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "TIARA"}


@app.get("/api/test-db")
async def test_db():
    from .agent_logic import SQL_RUNNER
    try:
        if SQL_RUNNER is None:
            return JSONResponse({"status": "error", "message": "SQL_RUNNER no inicializado"}, status_code=500)
        with SQL_RUNNER.engine.connect() as conn:
            rows = conn.execute(text("SELECT TOP 5 name FROM sys.tables ORDER BY name")).fetchall()
        return {"status": "ok", "tables": [r[0] for r in rows]}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/login")
async def login_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "login.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin.html"))


@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Chat (protegido con JWT) ───────────────────────────────────────────────────

@app.post("/api/tiara/chat_stream")
async def tiara_chat_stream(request: Request, _user: dict = Depends(require_auth)):
    try:
        body = await request.json()
        question = (body.get("question") or "").strip()
        conversation_id = body.get("conversation_id")
        is_retry = bool(body.get("retry", False))

        if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)

        ctx = build_request_context(request)

        async def sse():
            try:
                yield f"data: {json.dumps({'type': 'start'})}\n\n"

                # Tablas se retienen hasta saber si viene un gráfico.
                # TABLE_FLUSH_SENTINEL indica que no hay chart: emitir tabla inmediatamente.
                # Si llega CHART_SENTINEL, descartar buffer.
                table_buffer: list[str] = []
                flush_tables = False  # True → emitir tablas ahora, no al final
                chart_seen = False

                async for chunk in run_agent_stream_text(
                    agent=agent,
                    request_context=ctx,
                    message=question,
                    conversation_id=conversation_id,
                    retry=is_retry,
                ):
                    if chunk.startswith(ERROR_RETRY_SENTINEL):
                        msg = chunk[len(ERROR_RETRY_SENTINEL):]
                        yield f"data: {json.dumps({'type': 'error_retry', 'message': msg})}\n\n"
                    elif chunk == TABLE_FLUSH_SENTINEL:
                        flush_tables = True
                    elif chunk.startswith(CHART_SENTINEL):
                        chart_seen = True
                        table_buffer.clear()
                        chart_data = json.loads(chunk[len(CHART_SENTINEL):])
                        yield f"data: {json.dumps({'type': 'chart', 'data': chart_data})}\n\n"
                    elif '<table' in chunk:
                        if flush_tables:
                            yield f"data: {json.dumps({'type': 'table', 'content': chunk})}\n\n"
                        else:
                            table_buffer.append(chunk)
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"

                if not chart_seen:
                    for t in table_buffer:
                        yield f"data: {json.dumps({'type': 'table', 'content': t})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error_retry', 'message': 'Ocurrió un error inesperado.'})}\n\n"
                traceback.print_exc()

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e), "type": type(e).__name__},
            status_code=500,
        )


@app.delete("/api/tiara/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, _user: dict = Depends(require_auth)):
    return JSONResponse({"status": "deleted", "conversation_id": conversation_id})
