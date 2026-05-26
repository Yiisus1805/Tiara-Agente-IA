from __future__ import annotations

import json
import os
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from vanna.core.user import RequestContext

from .agent_logic import build_agent, run_agent_stream_text, CHART_SENTINEL


agent = build_agent()

app = FastAPI(title="TIARA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/")
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path)


# 🔥 SOLO STREAMING
@app.post("/api/tiara/chat_stream")
async def tiara_chat_stream(request: Request):
    try:
        body = await request.json()
        question = (body.get("question") or "").strip()
        conversation_id = body.get("conversation_id")

        if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)

        ctx = build_request_context(request)

        async def sse():
            try:
                yield f"data: {json.dumps({'type': 'start'})}\n\n"

                async for chunk in run_agent_stream_text(
                    agent=agent,
                    request_context=ctx,
                    message=question,
                    conversation_id=conversation_id,
                ):
                    if chunk.startswith(CHART_SENTINEL):
                        chart_data = json.loads(chunk[len(CHART_SENTINEL):])
                        yield f"data: {json.dumps({'type': 'chart', 'data': chart_data})}\n\n"
                    elif '<table' in chunk:
                        yield f"data: {json.dumps({'type': 'table', 'content': chunk})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'error_type': type(e).__name__})}\n\n"
                traceback.print_exc()

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e), "type": type(e).__name__},
            status_code=500,
        )


@app.delete("/api/tiara/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    return JSONResponse(
        {"status": "deleted", "conversation_id": conversation_id}
    )
