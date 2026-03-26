#!/bin/bash
# ── start.sh ─────────────────────────────────────────────────────────
# Arranca el backend: ingesta el schema 1 vez y luego lanza uvicorn.

set -e

SCHEMA_DIR="/app/backend/vanna_chromadb/schema_store"

if [ -d "$SCHEMA_DIR" ] && [ "$(ls -A "$SCHEMA_DIR" 2>/dev/null)" ]; then
    echo "📦 Schema ya ingested — saltando."
else
    echo "🚀 Corriendo ingest_schema..."
    cd /app && python -m backend.ingest_schema
    echo "✅ Ingest completado."
fi

echo "🌐 Arrancando FastAPI..."
exec uvicorn backend.app:app --host 0.0.0.0 --port 8000