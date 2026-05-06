from __future__ import annotations

import logging
import os
import uuid
import re
from typing import Any, Optional, AsyncGenerator

import httpx
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

RAG_K_FETCH = 10
RAG_K_FINAL = 6
SQL_CACHE_THRESHOLD = 0.97
MAX_ROWS_LIMIT = 500
MAX_RESPONSE_CACHE_LEN = 100_000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# NullFileSystem para no generar archivos csv

class NullFileSystem(FileSystem):
    async def write_file(self, filename, content, context=None, overwrite=False): pass
    async def read_file(self, filename, context=None): return ""
    async def list_files(self, context=None): return []
    async def delete_file(self, filename, context=None): pass
    async def exists(self, filename, context=None): return False
    async def is_directory(self, path, context=None): return False
    async def run_bash(self, command, context=None): return ""
    async def search_files(self, pattern, context=None): return []


# TrackingSqlTool

class TrackingSqlTool(RunSqlTool):
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


# UserResolver

class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie("vanna_email") or "guest@example.com"
        group = "admin" if user_email == "admin@example.com" else "user"
        return User(id=user_email, email=user_email, group_memberships=[group])

# Normalización temporal para cache

def _normalize_question(question: str) -> str:
    """Reemplaza años, meses y trimestres por placeholders numerados."""
    years = re.findall(r'\b(20\d{2}|19\d{2})\b', question)
    normalized = question
    for i, year in enumerate(years):
        normalized = normalized.replace(year, f'__YEAR{i+1}__', 1)
    normalized = re.sub(
        r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b',
        '__MONTH__', normalized, flags=re.IGNORECASE
    )
    normalized = re.sub(
        r'\b(q1|q2|q3|q4|primer trimestre|segundo trimestre|tercer trimestre|cuarto trimestre)\b',
        '__QUARTER__', normalized, flags=re.IGNORECASE
    )
    return normalized.strip().lower()


def _extract_temporals(question: str) -> dict:
    """Extrae los valores temporales reales de la pregunta original."""
    temporals = {}
    years = re.findall(r'\b(20\d{2}|19\d{2})\b', question)
    for i, year in enumerate(years):
        temporals[f"__YEAR{i+1}__"] = year

    month_match = re.search(
        r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\b',
        question, re.IGNORECASE
    )
    if month_match:
        temporals["__MONTH__"] = month_match.group(1).lower()

    quarter_match = re.search(
        r'\b(q1|q2|q3|q4|primer trimestre|segundo trimestre|tercer trimestre|cuarto trimestre)\b',
        question, re.IGNORECASE
    )
    if quarter_match:
        temporals["__QUARTER__"] = quarter_match.group(1).lower()

    return temporals


def _inject_temporals(text: str, temporals: dict) -> str:
    """Reemplaza placeholders por los valores reales."""
    for placeholder, value in temporals.items():
        text = text.replace(placeholder, value)
    return text


def _normalize_with_temporals(text: str, temporals: dict) -> str:
    """Reemplaza valores reales por placeholders."""
    normalized = text
    for placeholder, value in temporals.items():
        normalized = normalized.replace(value, placeholder)
    return normalized


# Generar análisis fresco con LLM

async def _generate_analysis(question: str, data_rows: list, columns: list) -> str:
    """Llama al LLM para generar un párrafo de análisis basado en los datos frescos."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return ""

    # Construir resumen de datos para el LLM
    rows_preview = data_rows[:10]
    data_str = ", ".join(columns) + "\n"
    for r in rows_preview:
        data_str += ", ".join(str(r.get(c, "")) for c in columns) + "\n"

    prompt = (
        f"Pregunta del usuario: {question}\n\n"
        f"Datos obtenidos:\n{data_str}\n\n"
        "Escribe UN párrafo breve (2-3 oraciones) en español analizando estos datos. "
        "Menciona los valores específicos. No uses markdown, bullets ni encabezados. "
        "Solo texto narrativo directo."
    )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Error generando análisis fresco")
        return ""


# SQL Cache

def _init_sql_cache(base_persist_dir: str):
    global SQL_CACHE

    cache_path = os.path.join(base_persist_dir, "sql_cache")
    os.makedirs(cache_path, exist_ok=True)

    client = PersistentClient(path=cache_path)
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    try:
        existing = client.get_collection(name="tiara_sql_cache")
        existing_meta = existing.metadata or {}
        existing_space = existing_meta.get("hnsw:space", "l2")
        existing_ef = existing_meta.get("embedding_function", "")

        needs_recreate = existing_space != "cosine" or "openai" in existing_ef.lower()
        if needs_recreate:
            logger.warning("Cache incompatible (space=%s, ef=%s). Recreando.", existing_space, existing_ef)
            client.delete_collection("tiara_sql_cache")
    except Exception:
        pass

    SQL_CACHE = client.get_or_create_collection(
        name="tiara_sql_cache",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine", "embedding_function": "default"},
    )
    logger.info("SQL Cache inicializado (%d entradas)", SQL_CACHE.count())


def _search_sql_cache(question: str) -> Optional[dict]:
    if not SQL_CACHE:
        return None
    try:
        normalized = _normalize_question(question)

        results = SQL_CACHE.query(query_texts=[normalized], n_results=1)
        if (
            not results
            or not results.get("documents", [[]])[0]
            or not results.get("distances", [[]])[0]
            or not results.get("metadatas", [[]])[0]
        ):
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 - (distance / 2.0)
        logger.info("Cache similarity: %.4f (threshold: %.2f)", similarity, SQL_CACHE_THRESHOLD)

        if similarity < SQL_CACHE_THRESHOLD:
            return None

        metadata = results["metadatas"][0][0]
        sql_template = metadata.get("sql", "")
        temporals = _extract_temporals(question)
        has_temporals = bool(temporals)

        if has_temporals:
            sql = _inject_temporals(sql_template, temporals)
            logger.info("Cache HIT con temporales sustituidos: %s", temporals)
        else:
            sql = sql_template

        return {
            "sql": sql,
            "full_response": "" if has_temporals else (metadata.get("full_response") or ""),
            "has_temporals": has_temporals,
        }
    except Exception:
        logger.exception("Error consultando SQL cache")
        return None


def _store_sql_cache(
    question: str,
    sql: str,
    full_response: str = "",
):
    if not SQL_CACHE:
        return

    if any(f in sql.lower() for f in ["current_date", "now()", "getdate()"]):
        logger.info("SQL no cacheado: contiene función de fecha dinámica")
        return

    if full_response and len(full_response) > MAX_RESPONSE_CACHE_LEN:
        full_response = full_response[:MAX_RESPONSE_CACHE_LEN]

    normalized_question = _normalize_question(question)
    temporals = _extract_temporals(question)
    normalized_sql = _normalize_with_temporals(sql, temporals)

    try:
        SQL_CACHE.add(
            ids=[str(uuid.uuid4())],
            documents=[normalized_question],
            metadatas=[{
                "sql": normalized_sql,
                # Para preguntas temporales no cacheamos full_response
                # porque el análisis contiene valores numéricos específicos
                "full_response": "" if temporals else (full_response or ""),
            }],
        )
        logger.info("SQL normalizado guardado en cache: '%s'", normalized_question)
    except Exception:
        logger.exception("Error guardando en cache")


# Helpers de texto

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
    indicators = [
        "tool completed successfully", "running tool", "executing tool",
        "run_sql", "similarity:", "**arguments:", "**timestamp:",
        "guardados en un archivo csv", "**id:",
        "query executed successfully. no rows returned",
        "**retrieved memories", "tool failed:", "error executing query:",
        "pyodbc", "sqlexecdirectw", "background on this error",
    ]
    return any(i in lower for i in indicators)


def _is_row_enumeration(text: str) -> bool:
    return bool(re.match(r"^\d+\.\s", text.strip()))


def _is_markdown_table_text(text: str) -> bool:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False
    pipe_lines = sum(1 for l in lines if l.startswith("|") and l.endswith("|"))
    return pipe_lines >= 2


# extract_text_from_component

def extract_text_from_component(component: Any) -> str:
    if component is None:
        return ""

    if isinstance(component, str):
        text = component.strip()
        if _should_hide_text(text) or _is_row_enumeration(text) or _is_markdown_table_text(text):
            return ""
        return text

    rc = getattr(component, "rich_component", None)
    if rc is not None:
        if hasattr(rc, "tool_name"):
            return ""

        if hasattr(rc, "rows") and hasattr(rc, "columns"):
            rows = getattr(rc, "rows", [])
            cols = getattr(rc, "columns", [])
            if rows and cols:
                max_rows = min(len(rows), MAX_ROWS_LIMIT)
                html = ['<table class="data-table"><thead><tr>']
                for col in cols:
                    html.append(f'<th>{_safe_str(col)}</th>')
                html.append('</tr></thead><tbody>')
                for r in rows[:max_rows]:
                    html.append('<tr>')
                    for c in cols:
                        v = r.get(c, "") if isinstance(r, dict) else ""
                        html.append(f'<td>{_safe_str(v).strip()}</td>')
                    html.append('</tr>')
                html.append('</tbody></table>')
                return "\n".join(html)

        content = getattr(rc, "content", None)
        if isinstance(content, str) and content.strip():
            text = content.strip()
            if _should_hide_text(text) or _is_row_enumeration(text) or _is_markdown_table_text(text):
                return ""
            return _clean_markdown(text)

        return ""

    sc = getattr(component, "simple_component", None)
    if sc is not None:
        text = getattr(sc, "text", None)
        if isinstance(text, str):
            text = text.strip()
            if _should_hide_text(text) or _is_row_enumeration(text) or _is_markdown_table_text(text):
                return ""
            return _clean_markdown(text)

    return ""


# Merge de tablas múltiples

def _merge_multiple_tables(html_chunks: list[str]) -> str:
    all_html = "\n".join(html_chunks)
    theads = re.findall(r'<thead>.*?</thead>', all_html, re.DOTALL | re.IGNORECASE)
    tbodies = re.findall(r'<tbody>(.*?)</tbody>', all_html, re.DOTALL | re.IGNORECASE)

    if not theads or not tbodies:
        return all_html

    all_rows = []
    for tbody in tbodies:
        all_rows.extend(re.findall(r'<tr>.*?</tr>', tbody, re.DOTALL | re.IGNORECASE))

    if not all_rows:
        return all_html

    merged = f'<table class="data-table">{theads[0]}<tbody>'
    merged += "\n".join(all_rows)
    merged += '</tbody></table>'
    logger.info("Tablas fusionadas: %d tbody → %d filas", len(tbodies), len(all_rows))
    return merged


# RAG

DISCOVERY_KEYWORDS = [
    "qué tablas", "que tablas", "tablas disponibles", "vistas disponibles",
    "tablas y vistas", "lista de tablas", "muestra las tablas", "muéstrame las tablas",
    "cuáles tablas", "cuales tablas", "todas las tablas", "show tables",
    "esquema disponible", "qué esquema", "que esquema",
]


def _is_discovery_question(message: str) -> bool:
    return any(kw in message.lower() for kw in DISCOVERY_KEYWORDS)


def _filter_and_deduplicate(hits: list) -> list:
    sorted_hits = sorted(hits, key=lambda h: h.get("distance", 999))
    return sorted_hits[:RAG_K_FINAL]


def _build_schema_prompt(message: str, hits: list) -> str:
    schema_lines = "\n".join([f"- {h['doc']}" for h in hits if h.get("doc")])

    return (
        "Eres un experto en SQL Server y análisis de negocio. Reglas:\n"
        "1. Usa SOLO tablas y columnas del esquema dado.\n"
        "2. Llama a run_sql UNA SOLA VEZ. Si la query falla NO reintentes — reporta el error.\n"
        "   Para filtrar por año en DimDate usa SIEMPRE: WHERE D.CalendarYear IN (...)\n"
        "3. Para limitar filas usa TOP N al inicio: SELECT TOP N ... (nunca al final).\n"
        "4. No menciones CSV, archivos, ni muestres el SQL generado.\n"
        "5. No uses **, ##, ni markdown en tus respuestas.\n"
        "6. NUNCA digas que el esquema no tiene información si hay tablas Fact con métricas.\n\n"
        "ANÁLISIS DE NEGOCIO (importante):\n"
        "- Si la pregunta es analítica o estratégica, infiere la métrica más relevante "
        "(SalesAmount, OrderQuantity) y responde con datos reales.\n"
        "- Para preguntas de regiones → usa DimSalesTerritory + FactInternetSales.\n"
        "- Para preguntas de productos → usa DimProduct + FactInternetSales.\n"
        "- Para preguntas de clientes → usa DimCustomer + FactInternetSales.\n"
        "- Para tendencias temporales → usa DimDate + FactInternetSales, filtra por CalendarYear.\n\n"
        "USO DE RELACIONES:\n"
        "- Úsalas para hacer JOIN entre tablas correctamente.\n"
        "- Para ventas sin canal específico usa SOLO FactInternetSales.\n"
        "- Nunca uses FactResellerSalesXL_PageCompressed ni FactResellerSalesXL_CCI.\n\n"
        "FORMATO (obligatorio):\n"
        "- Pregunta analítica simple (total, promedio, conteo, mejor/peor) → SOLO texto narrativo, sin tabla.\n"
        "- Múltiples filas con múltiples columnas (ranking, comparación, top N) → tabla seguida de un párrafo breve de análisis.\n"
        "- Resultado de 1 sola fila O 1 sola columna → SOLO texto narrativo, SIN tabla.\n"
        "- Respeta exactamente el N de filas pedido (TOP N en el SQL).\n"
        "- NUNCA uses formato | col | col | (markdown pipe).\n\n"
        "ESQUEMA:\n"
        "RECUERDA: Tu respuesta DEBE terminar con un párrafo de texto analizando los resultados.\n\n"
        f"{schema_lines}\n\n"
        f"PREGUNTA: {message}"
    )


def _inject_schema_rag(message: str) -> str:
    if not SCHEMA_STORE:
        return message
    try:
        if _is_discovery_question(message):
            total = SCHEMA_STORE.count()
            all_hits = SCHEMA_STORE.query(message, k=total) if total > 0 else []
            hits = sorted(all_hits, key=lambda h: h.get("distance", 999))
        else:
            raw_hits = SCHEMA_STORE.query(message, k=RAG_K_FETCH)
            hits = _filter_and_deduplicate(raw_hits)

        if not hits:
            return message

        logger.info("RAG hits para '%s':", message)
        for h in hits:
            logger.info("  score=%.4f tabla=%s", h.get("score", 0), h.get("meta", {}).get("table"))

        return _build_schema_prompt(message, hits)
    except Exception:
        logger.exception("RAG falló")
        return message


# Stream principal

async def run_agent_stream_text(
    agent: Agent,
    request_context: RequestContext,
    message: str,
    conversation_id: Optional[str],
) -> AsyncGenerator[str, None]:

    original_question = message

    # 1. Cache
    cache_hit = _search_sql_cache(original_question)
    if cache_hit:
        full_response = cache_hit.get("full_response")
        cached_sql = cache_hit.get("sql")
        has_temporals = cache_hit.get("has_temporals", False)

        # Sin temporales: devolver respuesta cacheada completa
        if full_response and not has_temporals:
            logger.info("Cache HIT con full_response")
            yield full_response
            return

        # Con temporales: re-ejecutar SQL y generar análisis fresco
        if cached_sql and SQL_RUNNER:
            try:
                tool_args = RunSqlToolArgs(sql=cached_sql)
                df = await SQL_RUNNER.run_sql(tool_args, None)
                logger.info("Cache HIT (SQL re-ejecutado) — %d filas", len(df))

                if not df.empty:
                    cols = df.columns.tolist()
                    rows = df.to_dict("records")

                    # Emitir tabla
                    html = ['<table class="data-table"><thead><tr>']
                    for col in cols:
                        html.append(f'<th>{_safe_str(col)}</th>')
                    html.append('</tr></thead><tbody>')
                    for r in rows[:MAX_ROWS_LIMIT]:
                        html.append('<tr>')
                        for c in cols:
                            html.append(f'<td>{_safe_str(r.get(c, "")).strip()}</td>')
                        html.append('</tr>')
                    html.append('</tbody></table>')
                    yield "\n".join(html)

                    # Generar análisis fresco con los datos reales
                    analysis = await _generate_analysis(original_question, rows, cols)
                    if analysis:
                        yield analysis
                else:
                    yield "La consulta no devolvió resultados."

                return
            except Exception:
                logger.exception("Error ejecutando SQL desde cache, continuando con flujo normal")

    # 2. RAG
    message = _inject_schema_rag(message)

    # 3. Callback SQL
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

    # 4. Ejecutar agente
    pre_table_buffer: list[str] = []
    tabla_chunks: list[str] = []
    post_table_chunks: list[str] = []
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
                table_seen = True
                tabla_chunks.append(text_output)
            elif table_seen:
                post_table_chunks.append(text_output)
            else:
                pre_table_buffer.append(text_output)

    except Exception:
        logger.exception("Error en agent.send_message")
        yield "Ocurrió un error procesando tu pregunta. Por favor intenta de nuevo."
        return

    # 5. Emitir respuesta
    response_chunks: list[str] = []

    if table_seen and tabla_chunks:
        merged_table = _merge_multiple_tables(tabla_chunks) if len(tabla_chunks) > 1 else tabla_chunks[0]
        response_chunks.append(merged_table)
        yield merged_table
        for chunk in post_table_chunks:
            response_chunks.append(chunk)
            yield chunk
    else:
        for chunk in pre_table_buffer:
            response_chunks.append(chunk)
            yield chunk

    # 6. Guardar en cache
    if captured_sql:
        _store_sql_cache(
            original_question,
            captured_sql[-1],
            full_response="\n".join(response_chunks),
        )
        logger.info("SQL guardado en cache para: '%s'", original_question)
    else:
        logger.info("No se capturó SQL, no se guarda en cache")


# Build agent

def build_agent() -> Agent:
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)

    project_root = os.path.dirname(os.path.dirname(__file__))
    base_persist_dir = os.path.join(project_root, "backend", "vanna_chromadb")

    path_memory = os.path.join(base_persist_dir, "agent_memory")
    path_schema = os.path.join(base_persist_dir, "schema_store")

    os.makedirs(path_memory, exist_ok=True)
    os.makedirs(path_schema, exist_ok=True)

    _init_sql_cache(base_persist_dir)

    api_key = os.getenv("OPENAI_API_KEY")
    odbc_str = os.getenv("SQLSERVER_ODBC")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    global SQL_RUNNER, TRACKING_TOOL
    SQL_RUNNER = SqlServerRunner(odbc_str)

    llm = OpenAILlmService(model=model, api_key=api_key)

    tools = ToolRegistry()
    TRACKING_TOOL = TrackingSqlTool(sql_runner=SQL_RUNNER)
    tools.register_local_tool(TRACKING_TOOL, access_groups=["admin", "user"])
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

    return Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=SimpleUserResolver(),
        agent_memory=agent_memory,
    )