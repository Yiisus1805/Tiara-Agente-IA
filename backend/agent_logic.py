from __future__ import annotations

import logging
import os
import uuid
import re
from typing import Any, Optional, AsyncGenerator

from dotenv import load_dotenv

from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.core.user import UserResolver, User, RequestContext
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.tools import RunSqlTool
from vanna.capabilities.sql_runner import RunSqlToolArgs
from vanna.capabilities.file_system import FileSystem

from chromadb import PersistentClient
from chromadb.utils import embedding_functions

from .schema_store import SchemaVectorStore
from .database import SqlServerRunner

logger = logging.getLogger(__name__)

SCHEMA_STORE: SchemaVectorStore | None = None
SQL_RUNNER: SqlServerRunner | None = None
SQL_CACHE = None
TRACKING_TOOL: "TrackingSqlTool | None" = None

RAG_K_FETCH = 30
RAG_K_FINAL = 15
RAG_MIN_SCORE = 0.45
SQL_CACHE_THRESHOLD = 0.92
MAX_ROWS_LIMIT = 500
MAX_RESPONSE_CACHE_LEN = 100_000  # ~100 KB por respuesta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class NullFileSystem(FileSystem):
    """Descarta todos los archivos — evita la generación de CSVs por RunSqlTool."""

    async def write_file(self, filename, content, context=None, overwrite=False):
        pass

    async def read_file(self, filename, context=None):
        return ""

    async def list_files(self, context=None):
        return []

    async def delete_file(self, filename, context=None):
        pass

    async def exists(self, filename, context=None):
        return False

    async def is_directory(self, path, context=None):
        return False

    async def run_bash(self, command, context=None):
        return ""

    async def search_files(self, pattern, context=None):
        return []


class TrackingSqlTool(RunSqlTool):
    """
    Intercepta execute(context, args) — el método que _LocalToolWrapper delega.
    args es RunSqlToolArgs con campo .sql que contiene el query string directamente.
    El callback se muta por request via set_callback() sin re-registrar el tool.
    """

    def __init__(self, sql_runner, on_sql_executed=None):
        super().__init__(sql_runner=sql_runner, file_system=NullFileSystem())
        self._on_sql_executed = on_sql_executed or (lambda sql: None)

    def set_callback(self, callback):
        self._on_sql_executed = callback

    async def execute(self, context, args):
        sql = getattr(args, "sql", None)
        logger.info("TrackingSqlTool.execute() — SQL: %s", sql)

        result = await super().execute(context, args)

        if sql:
            try:
                self._on_sql_executed(sql)
            except Exception:
                logger.exception("Error en TrackingSqlTool callback")

        return result


# USER RESOLVER

class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie("vanna_email") or "guest@example.com"
        group = "admin" if user_email == "admin@example.com" else "user"
        return User(id=user_email, email=user_email, group_memberships=[group])


# SQL CACHE

def _init_sql_cache(base_persist_dir: str):
    global SQL_CACHE

    cache_path = os.path.join(base_persist_dir, "sql_cache")
    os.makedirs(cache_path, exist_ok=True)

    client = PersistentClient(path=cache_path)

    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_key_env_var=None,
        model_name="text-embedding-3-small",
    )

    try:
        existing = client.get_collection(name="tiara_sql_cache")
        existing_space = (existing.metadata or {}).get("hnsw:space", "l2")
        if existing_space != "cosine":
            logger.warning(
                "Cache existente usa space='%s'. Eliminando y recreando con cosine.",
                existing_space,
            )
            client.delete_collection("tiara_sql_cache")
    except Exception:
        pass

    SQL_CACHE = client.get_or_create_collection(
        name="tiara_sql_cache",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(
        "SQL Cache inicializado (cosine space, %d entradas existentes)",
        SQL_CACHE.count(),
    )


def _search_sql_cache(question: str) -> Optional[dict]:
    """
    Retorna dict con 'sql' y opcionalmente 'full_response', o None si no hay hit.
    """
    if not SQL_CACHE:
        logger.warning("SQL_CACHE no inicializado al consultar")
        return None

    try:
        results = SQL_CACHE.query(
            query_texts=[question],
            n_results=1
        )
        logger.info("Resultado cache raw: %s", results)

        if (
            not results
            or not results.get("documents")
            or not results["documents"][0]
            or not results.get("distances")
            or not results["distances"][0]
            or not results.get("metadatas")
            or not results["metadatas"][0]
        ):
            logger.info("Cache miss: resultado vacío o incompleto")
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - distance

        logger.info("Cache similarity calculada: %.4f (threshold: %.2f)", similarity, SQL_CACHE_THRESHOLD)

        if similarity < SQL_CACHE_THRESHOLD:
            logger.info("Cache miss: similarity %.4f < threshold %.2f", similarity, SQL_CACHE_THRESHOLD)
            return None

        metadata = results["metadatas"][0][0]
        sql = metadata.get("sql")
        full_response = metadata.get("full_response") or ""

        logger.info(
            "Cache HIT (similarity=%.4f): sql=%s | full_response=%s",
            similarity, sql, "sí" if full_response else "no"
        )

        return {"sql": sql, "full_response": full_response}

    except Exception:
        logger.exception("Error consultando SQL cache")
        return None


def _store_sql_cache(question: str, sql: str, full_response: str = ""):
    """
    Guarda en cache el SQL y la respuesta completa (texto + tabla HTML).
    """
    if not SQL_CACHE:
        logger.warning("SQL_CACHE no inicializado al guardar")
        return

    sql_lower = sql.lower()
    forbidden = ["current_date", "now()", "getdate()"]

    if any(f in sql_lower for f in forbidden):
        logger.info("SQL no cacheado: contiene función de fecha dinámica")
        return

    # Truncar si es demasiado grande para ChromaDB metadata
    if full_response and len(full_response) > MAX_RESPONSE_CACHE_LEN:
        logger.warning("full_response truncada de %d a %d chars", len(full_response), MAX_RESPONSE_CACHE_LEN)
        full_response = full_response[:MAX_RESPONSE_CACHE_LEN]

    try:
        SQL_CACHE.add(
            ids=[str(uuid.uuid4())],
            documents=[question],
            metadatas=[{
                "sql": sql,
                "full_response": full_response or "",
            }]
        )
        logger.info("SQL + respuesta guardados en cache exitosamente")
    except Exception:
        logger.exception("Error guardando SQL en cache")


# FILTROS Y HELPERS

def _clean_markdown(text: str) -> str:
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    return text


def _safe_str(v: Any) -> str:
    try:
        return "" if v is None else str(v)
    except Exception:
        return ""


def _should_hide_text(text: str) -> bool:
    lower = text.lower()

    technical_indicators = [
        "tool completed successfully",
        "running tool",
        "executing tool",
        "run_sql",
        "similarity:",
        "**arguments:",
        "**timestamp:",
        "guardados en un archivo csv",
        "**id:",
        "query executed successfully. no rows returned",
        "**retrieved memories",
        "tool failed:",
        "error executing query:",
        "pyodbc",
        "sqlexecdirectw",
        "background on this error",
    ]

    return any(indicator in lower for indicator in technical_indicators)


def _is_row_enumeration(text: str) -> bool:
    return bool(re.match(r"^\d+\.\s", text.strip()))


def extract_text_from_component(component: Any) -> str:
    if component is None:
        return ""

    if isinstance(component, str):
        text = component.strip()
        if _should_hide_text(text) or _is_row_enumeration(text):
            return ""
        return text

    rc = getattr(component, "rich_component", None)
    if rc is not None:

        if hasattr(rc, "tool_name"):
            return ""

        # Tabla — si hay tabla, ignorar simple_component (evita mostrar razonamiento)
        if hasattr(rc, "rows") and hasattr(rc, "columns"):
            rows = getattr(rc, "rows", [])
            cols = getattr(rc, "columns", [])

            if rows and cols:
                max_rows = min(len(rows), MAX_ROWS_LIMIT)
                html_parts = ['<table class="data-table">']
                html_parts.append('<thead><tr>')
                for col in cols:
                    html_parts.append(f'<th>{_safe_str(col)}</th>')
                html_parts.append('</tr></thead><tbody>')

                for r in rows[:max_rows]:
                    html_parts.append('<tr>')
                    for c in cols:
                        v = r.get(c, "") if isinstance(r, dict) else ""
                        html_parts.append(f'<td>{_safe_str(v)}</td>')
                    html_parts.append('</tr>')

                html_parts.append('</tbody></table>')
                return "\n".join(html_parts)

        content = getattr(rc, "content", None)
        if isinstance(content, str) and content.strip():
            text = content.strip()
            if _should_hide_text(text) or _is_row_enumeration(text):
                return ""
            return _clean_markdown(text)

        # rich_component existe pero no tiene tabla ni content útil
        # → ignorar simple_component para no mostrar el razonamiento interno
        return ""

    sc = getattr(component, "simple_component", None)
    if sc is not None:
        text = getattr(sc, "text", None)
        if isinstance(text, str):
            text = text.strip()
            if _should_hide_text(text) or _is_row_enumeration(text):
                return ""
            return _clean_markdown(text)

    return ""


# RAG

def _filter_and_deduplicate(hits: list) -> list:
    filtered = [
        h for h in hits
        if h.get("score", 1.0) >= RAG_MIN_SCORE
    ]
    filtered.sort(key=lambda h: h.get("score", 0.0), reverse=True)
    return filtered[:RAG_K_FINAL]


def _build_schema_prompt(message: str, hits: list) -> str:
    schema_lines = "\n".join([f"- {h['doc']}" for h in hits if h.get("doc")])

    return (
        "Eres un experto senior en SQL Server y analisis de datos.\n"
        "Reglas estrictas:\n"
        "- Usa SOLO las tablas y columnas del esquema proporcionado.\n"
        "- Si la pregunta involucra fechas, busca columnas de tipo fecha en TODAS las tablas del esquema.\n"
        "- Si necesitas hacer JOIN entre tablas para responder, hazlo.\n"
        "- Si necesitas combinar datos de múltiples tablas similares usa UNION ALL en una sola consulta. NUNCA ejecutes múltiples queries separadas.\n"
        "- No menciones que necesitaste combinar información para mostrar algo, ni tampoco muestres la consulta en sql realizada.\n"
        "- Para limitar filas usa TOP, nunca LIMIT.\n"
        "- No menciones que puedes acceder al archivo resultante para visualizar todos los detalles.\n"
        "- Evita mostrar ** y ## en tus respuestas y cualquier cosa que genere ruido visual.\n"
        "- Si con el esquema disponible no puedes responder, explica qué tabla o columna falta.\n\n"
        "ESQUEMA DISPONIBLE (todas las tablas relevantes):\n"
        f"{schema_lines}\n\n"
        f"PREGUNTA: {message}"
    )


DISCOVERY_KEYWORDS = [
    "qué tablas", "que tablas", "tablas disponibles", "vistas disponibles",
    "tablas y vistas", "lista de tablas", "muestra las tablas", "muéstrame las tablas",
    "cuáles tablas", "cuales tablas", "todas las tablas", "show tables",
    "esquema disponible", "qué esquema", "que esquema",
]

def _is_discovery_question(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in DISCOVERY_KEYWORDS)


def _inject_schema_rag(message: str) -> str:
    if not SCHEMA_STORE:
        return message

    try:
        # Para preguntas de discovery, inyectar TODO el schema sin filtrar
        if _is_discovery_question(message):
            logger.info("Discovery question detectada — inyectando schema completo")
            total = SCHEMA_STORE.count()
            all_hits = SCHEMA_STORE.query(message, k=total) if total > 0 else []
            hits = sorted(all_hits, key=lambda h: h.get("score", 0.0), reverse=True)
        else:
            raw_hits = SCHEMA_STORE.query(message, k=RAG_K_FETCH)
            hits = _filter_and_deduplicate(raw_hits)

        if not hits:
            return message

        return _build_schema_prompt(message, hits)

    except Exception:
        logger.exception("RAG falló")
        return message


# STREAM PRINCIPAL

async def run_agent_stream_text(
    agent: Agent,
    request_context: RequestContext,
    message: str,
    conversation_id: Optional[str],
) -> AsyncGenerator[str, None]:

    original_question = message

    # 1. Intentar responder desde cache
    cache_hit = _search_sql_cache(original_question)
    if cache_hit:
        full_response = cache_hit.get("full_response")
        cached_sql = cache_hit.get("sql")

        # Tenemos respuesta completa guardada → devolverla directamente
        if full_response:
            logger.info("Cache HIT con full_response — devolviendo respuesta guardada")
            yield full_response
            return

        # Fallback: entradas antiguas solo con SQL → re-ejecutar
        if cached_sql and SQL_RUNNER:
            try:
                tool_args = RunSqlToolArgs(sql=cached_sql)
                df = await SQL_RUNNER.run_sql(tool_args, None)
                logger.info("Cache HIT (solo SQL) ejecutado correctamente (%d filas)", len(df))
                if not df.empty:
                    cols = df.columns.tolist()
                    rows = df.to_dict("records")
                    html_parts = ['<table class="data-table"><thead><tr>']
                    for col in cols:
                        html_parts.append(f'<th>{_safe_str(col)}</th>')
                    html_parts.append('</tr></thead><tbody>')
                    for r in rows[:MAX_ROWS_LIMIT]:
                        html_parts.append('<tr>')
                        for c in cols:
                            html_parts.append(f'<td>{_safe_str(r.get(c, ""))}</td>')
                        html_parts.append('</tr>')
                    html_parts.append('</tbody></table>')
                    yield "\n".join(html_parts)
                else:
                    yield "La consulta no devolvió resultados."
                return
            except Exception:
                logger.exception("Error ejecutando SQL desde cache, continuando con flujo normal")

    # 2. Enriquecer mensaje con RAG de esquema
    message = _inject_schema_rag(message)

    # 3. Actualizar callback para esta request
    captured_sql: list[str] = []

    def on_sql_executed(sql: str):
        if sql not in captured_sql:
            captured_sql.append(sql)
            logger.info("SQL capturado: %s", sql)

    if TRACKING_TOOL:
        TRACKING_TOOL.set_callback(on_sql_executed)
        logger.info("TrackingSqlTool callback actualizado (id=%s)", id(TRACKING_TOOL))
    else:
        logger.warning("TRACKING_TOOL no inicializado")

    # 4. Ejecutar agente, hacer stream y acumular respuesta completa
    # Estrategia: buffer de texto pre-tabla.
    # - Texto antes de tabla: acumular pero NO emitir todavía.
    # - Cuando llega una tabla: descartar buffer pre-tabla, emitir tabla.
    # - Texto después de tabla: emitir normalmente.
    response_chunks: list[str] = []
    pre_table_buffer: list[str] = []
    table_seen = False

    try:
        async for component in agent.send_message(
            request_context=request_context,
            message=message,
            conversation_id=conversation_id,
        ):
            text_output = extract_text_from_component(component)
            if not text_output:
                continue

            is_table = '<table' in text_output

            if is_table:
                # Descartar buffer pre-tabla y emitir la tabla
                pre_table_buffer.clear()
                table_seen = True
                response_chunks.append(text_output)
                yield text_output
            elif table_seen:
                # Post-tabla: emitir normalmente
                response_chunks.append(text_output)
                yield text_output
            else:
                # Pre-tabla: guardar en buffer sin emitir
                pre_table_buffer.append(text_output)

    except Exception:
        logger.exception("Error en agent.send_message")
        yield "Ocurrió un error procesando tu pregunta. Por favor intenta de nuevo."
        return

    # Si nunca hubo tabla, emitir el texto acumulado en el buffer
    if not table_seen:
        for chunk in pre_table_buffer:
            response_chunks.append(chunk)
            yield chunk

    # 5. Guardar en cache: SQL + respuesta completa acumulada
    if captured_sql:
        executed_sql = captured_sql[-1]
        full_response = "\n".join(response_chunks)
        logger.info("Guardando SQL + respuesta en cache para: '%s'", original_question)
        _store_sql_cache(original_question, executed_sql, full_response)
    else:
        logger.info("No se capturó SQL, no se guarda en cache")


# BUILD AGENT

def build_agent() -> Agent:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    project_root = os.path.dirname(os.path.dirname(__file__))
    base_persist_dir = os.path.join(project_root, "backend", "vanna_chromadb")

    path_memory = os.path.join(base_persist_dir, "agent_memory")
    path_schema = os.path.join(base_persist_dir, "schema_store")

    os.makedirs(path_memory, exist_ok=True)
    os.makedirs(path_schema, exist_ok=True)

    _init_sql_cache(base_persist_dir)

    api_key = os.getenv("OPENAI_API_KEY")
    odbc_str = os.getenv("SQLSERVER_ODBC")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    global SQL_RUNNER, TRACKING_TOOL
    SQL_RUNNER = SqlServerRunner(odbc_str)

    llm = OpenAILlmService(model=model, api_key=api_key)

    tools = ToolRegistry()

    TRACKING_TOOL = TrackingSqlTool(sql_runner=SQL_RUNNER)
    tools.register_local_tool(
        TRACKING_TOOL,
        access_groups=["admin", "user"],
    )
    logger.info("TrackingSqlTool registrado (id=%s)", id(TRACKING_TOOL))

    agent_memory = ChromaAgentMemory(
        persist_directory=path_memory,
        collection_name=os.getenv("CHROMA_COLLECTION") or "vanna_tool_memory",
    )

    global SCHEMA_STORE
    SCHEMA_STORE = SchemaVectorStore(
        persist_dir=path_schema,
        collection_name=os.getenv("SCHEMA_COLLECTION") or "tiara_schema",
        embedding_mode="default",
    )

    user_resolver = SimpleUserResolver()

    return Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=user_resolver,
        agent_memory=agent_memory,
    )