from __future__ import annotations

import asyncio
import json
import logging
import os
from contextvars import ContextVar
from typing import Optional, AsyncGenerator

from dotenv import load_dotenv

from vanna import Agent, AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.core.user import UserResolver, User, RequestContext
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.tools import RunSqlTool
from vanna.capabilities.sql_runner import RunSqlToolArgs
from vanna.capabilities.file_system import FileSystem

from .schema_store import SchemaVectorStore
from .database import SqlServerRunner
from .schema_analyzer import analyze_schema

from . import agent_state as state

from .sql_pipeline import (
    _check_sql_safety,
    _sanitize_sql_aliases,
    _sql_validation_problems,
    _validate_and_fix_sql,
    _get_corrected_sql,
    _extract_sql_error_detail,
    _has_destructive_intent,
    _init_sql_cache,
    _search_sql_cache,
    _store_sql_cache,
    _evict_sql_cache,
    READONLY_REFUSAL as _READONLY_REFUSAL,
)
from .agent_features import (
    _classify_intent,
    _get_chat_response,
    _run_stats,
    _run_discovery,
    _is_chart_question,
    _is_historical_sales_question,
    _is_discovery_question,
    _is_vague_analysis,
    _build_chart_payload,
    _generate_analysis,
    _stream_no_results,
    extract_text_from_component,
    _merge_multiple_tables,
    _safe_str,
    _format_cell,
    _inject_schema_rag,
)

logger = logging.getLogger(__name__)

# Contexto por-request: aislamiento total entre requests concurrentes
_ctx_sql_callback: ContextVar = ContextVar("_tiara_sql_cb", default=None)
_ctx_sql_question: ContextVar = ContextVar("_tiara_sql_q", default="")
_ctx_sql_inflight: ContextVar = ContextVar("_tiara_sql_inflight", default=None)
_ctx_intent: ContextVar = ContextVar("_tiara_intent", default="")

CHART_SENTINEL = "\x00CHART\x00"
ERROR_RETRY_SENTINEL = "\x00ERROR_RETRY\x00"
TABLE_FLUSH_SENTINEL = "\x00TABLE_FLUSH\x00"  # indica que no viene chart: emitir tabla ya

_PREDICTION_REFUSAL = (
    "Soy un agente conversacional de consulta de datos (texto a SQL): puedo responder "
    "preguntas sobre los datos históricos que ya existen en la base de datos, pero no "
    "estoy capacitado para hacer predicciones o proyecciones a futuro."
)
AGENT_STEP_TIMEOUT = float(os.getenv("AGENT_STEP_TIMEOUT", "90"))

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


class TrackingSqlTool(RunSqlTool):
    """Instancia única compartida; el estado por-request vive en ContextVars."""

    def __init__(self, sql_runner):
        super().__init__(sql_runner=sql_runner, file_system=NullFileSystem())

    async def execute(self, context, args):
        sql = getattr(args, "sql", None)
        if sql:
            safety_error = _check_sql_safety(sql)
            if safety_error:
                raise RuntimeError(safety_error)

            sanitized = _sanitize_sql_aliases(sql)
            if sanitized != sql:
                logger.info("SQL sanitizado — alias reservados reemplazados")
                args = RunSqlToolArgs(sql=sanitized)
                sql = sanitized

            current_question = _ctx_sql_question.get()

            # _validate_and_fix_sql corre el mismo pipeline de validación/corrección
            # que usa STATS/PREDICTION (ver sql_pipeline.py) — combina todos los
            # problemas detectados en una sola llamada de corrección por ronda (no
            # una por categoría, que llegó a encadenar tantas llamadas al LLM dentro
            # de una sola ejecución de la tool que corrompió el historial de la
            # conversación vía error 400 de OpenAI) y lanza RuntimeError si tras 2
            # rondas el SQL sigue con problemas — preferible fallar la tool a
            # ejecutar SQL que sabemos roto (ej. el fan-out de DimGeography).
            corrected_sql = await _validate_and_fix_sql(sql, current_question or "")
            if corrected_sql != sql:
                _evict_sql_cache(current_question)
                args = RunSqlToolArgs(sql=corrected_sql)
                sql = corrected_sql

        sql_key = (sql or "").strip()
        in_flight: set[str] | None = _ctx_sql_inflight.get()
        if in_flight is None:
            in_flight = set()
            _ctx_sql_inflight.set(in_flight)

        if sql_key and sql_key in in_flight:
            logger.warning("TrackingSqlTool — ejecución duplicada bloqueada")
            return None
        if sql_key:
            in_flight.add(sql_key)

        logger.info("TrackingSqlTool.execute() — SQL: %s", sql)
        result = await super().execute(context, args)
        if sql:
            try:
                cb = _ctx_sql_callback.get()
                if cb is not None:
                    cb(sql)
            except Exception:
                logger.exception("Error en TrackingSqlTool callback")
        return result


class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie("vanna_email") or "guest@example.com"
        group = "admin" if user_email == "admin@example.com" else "user"
        return User(id=user_email, email=user_email, group_memberships=[group])


async def run_agent_stream_text(
    agent: Agent,
    request_context: RequestContext,
    message: str,
    conversation_id: Optional[str],
    retry: bool = False,
) -> AsyncGenerator[str, None]:

    original_question = message

    if _has_destructive_intent(original_question):
        logger.warning("Intención destructiva detectada y bloqueada: %.100s", original_question)
        _ctx_intent.set("CHAT")
        yield _READONLY_REFUSAL
        return

    # Clasifica la intención y enruta a SQL, stats, discovery, conversación o rechazo (predicción)
    if not retry:
        intent = await _classify_intent(original_question)
        # Gráfico de seguimiento ("haz un gráfico con esos valores", "grafica eso"):
        # el clasificador ve un mensaje sin contexto y lo manda a CHAT, pero el
        # agente SQL SÍ tiene el historial de conversación y puede resolverlo.
        if intent == "CHAT" and _is_chart_question(original_question):
            logger.info("Reclasificado CHAT→SQL: petición de gráfico contextual")
            intent = "SQL"
        if intent == "PREDICTION" and _is_historical_sales_question(original_question):
            logger.info("Reclasificado PREDICTION→SQL: pregunta sobre año histórico en el dataset")
            intent = "SQL"
        if intent == "DISCOVERY" and not _is_discovery_question(original_question):
            # El clasificador ve un mensaje sin contexto (ej. "¿y cuáles de esos son de
            # EE.UU.?", un seguimiento con pronombres) y a veces lo confunde con DISCOVERY.
            # Las preguntas de discovery genuinas SIEMPRE contienen palabras clave literales
            # (DISCOVERY_KEYWORDS) — si no las trae, es casi seguro un falso positivo.
            logger.info("Reclasificado DISCOVERY→SQL: no contiene palabras clave de discovery")
            intent = "SQL"
        _ctx_intent.set(intent)
        if intent == "CHAT":
            yield await _get_chat_response(original_question)
            return
        if intent == "PREDICTION":
            logger.info("Pregunta predictiva detectada — fuera de alcance, respondiendo amablemente")
            yield _PREDICTION_REFUSAL
            return
        if intent == "STATS":
            logger.info("Pregunta de análisis estadístico detectada — ejecutando STATS")
            narrative, table_html = await _run_stats(original_question)
            if table_html:
                yield table_html
            if narrative:
                yield narrative
            return
        if intent == "DISCOVERY":
            logger.info("Discovery detectado — consultando tablas y registros de la BD")
            table_html, summary = await _run_discovery(original_question)
            if table_html:
                yield table_html
            if summary:
                yield summary
            return

    # En reintento forzamos SQL fresco evictando la entrada cacheada
    if retry:
        _evict_sql_cache(original_question)
        logger.info("Reintento — cache evictado para '%s'", original_question)

    cache_hit = _search_sql_cache(original_question)
    if cache_hit:
        cached_sql    = cache_hit.get("sql")
        full_response = cache_hit.get("full_response") or ""
        has_temporals = cache_hit.get("has_temporals", False)

        is_chart = _is_chart_question(original_question)

        # El SQL cacheado puede haberse guardado ANTES de que existieran las
        # reglas de validación actuales (fan-out de JOINs, fuente de ventas
        # incompleta, tablas de benchmark, columnas *Key expuestas) — sin este
        # chequeo, una entrada vieja podría servir cifras incorrectas
        # indefinidamente (ej. el fan-out de DimGeography) sin que nada lo
        # detecte, ya sea reejecutando el SQL o, peor, devolviendo directamente
        # el full_response ya cacheado sin volver a tocar la BD.
        if cached_sql:
            cache_problems = _sql_validation_problems(cached_sql, original_question)
            if cache_problems:
                logger.warning(
                    "Cache HIT descartado — SQL con problemas conocidos: %s",
                    "; ".join(cache_problems),
                )
                _evict_sql_cache(original_question)
                cached_sql = None
                full_response = ""

        # Match exacto: partir full_response en tabla + texto para que el
        # frontend pueda animar cada sección por separado.
        # Si es pregunta de gráfico y la respuesta cacheada no tiene tabla
        # (fue respuesta de 1 fila), ignorar caché y regenerar con RAG.
        if full_response and not has_temporals:
            if is_chart:
                # Gráficos siempre re-ejecutan SQL: el payload ECharts no se guarda
                # en full_response y sin él el frontend no renderiza el gráfico
                logger.info("Cache HIT: pregunta de gráfico — re-ejecutando SQL para regenerar gráfico")
            else:
                logger.info("Cache HIT con full_response")
                if '<table' in full_response.lower() and '</table>' in full_response.lower():
                    table_end = full_response.lower().rfind('</table>') + len('</table>')
                    yield full_response[:table_end].strip()      # → tipo 'table' → fade-in
                    text_part = full_response[table_end:].strip()
                    if text_part:
                        yield text_part                          # → tipo 'text' → typewriter
                else:
                    yield full_response                          # respuesta solo-texto
                return

        if cached_sql and state.SQL_RUNNER:
            try:
                tool_args = RunSqlToolArgs(sql=cached_sql)
                df = await state.SQL_RUNNER.run_sql(tool_args, None)
                logger.info("Cache HIT (SQL re-ejecutado) — %d filas", len(df))

                if not df.empty:
                    if is_chart and len(df) < 2:
                        logger.info("Cache HIT ignorado: pregunta de gráfico con 1 fila — regenerando con RAG")
                    else:
                        cols = df.columns.tolist()
                        rows = df.to_dict("records")

                        if len(cols) > 1 and len(rows) > 1:
                            html = ['<table class="data-table"><thead><tr>']
                            for col in cols:
                                html.append(f'<th>{_safe_str(col)}</th>')
                            html.append('</tr></thead><tbody>')
                            for r in rows[:state.MAX_ROWS_LIMIT]:
                                html.append('<tr>')
                                for c in cols:
                                    html.append(f'<td>{_format_cell(r.get(c, ""), c)}</td>')
                                html.append('</tr>')
                            html.append('</tbody></table>')
                            yield "\n".join(html)

                        analysis = await _generate_analysis(original_question, rows, cols)
                        if analysis:
                            yield analysis

                        if is_chart:
                            chart_payload = _build_chart_payload(original_question, cols, rows)
                            if chart_payload:
                                yield CHART_SENTINEL + json.dumps(chart_payload)
                                logger.info("Gráfico ECharts generado desde cache")
                        return
                else:
                    async for token in _stream_no_results(original_question):
                        yield token
                    return
            except Exception:
                logger.exception("Error re-ejecutando SQL desde cache, continuando con flujo normal")

    message = _inject_schema_rag(message)

    captured_sql: list[str] = []

    def on_sql_executed(sql: str):
        if sql not in captured_sql:
            captured_sql.append(sql)
            logger.info("SQL capturado: %s", sql)

    _ctx_sql_callback.set(on_sql_executed)
    _ctx_sql_question.set(original_question)
    _ctx_sql_inflight.set(set())

    pre_table_buffer: list[str] = []
    tabla_chunks: list[str] = []
    post_table_chunks: list[str] = []
    table_seen = False
    meaningful_post_analysis = False  # True cuando se emitió análisis sustancial (>20 chars)
    response_chunks: list[str] = []

    async def _step_agent(conv_id: Optional[str]):
        """Ejecuta el agente llenando los buffers. Lanza excepción si falla."""
        nonlocal table_seen, meaningful_post_analysis
        agen = agent.send_message(
            request_context=request_context,
            message=message,
            conversation_id=conv_id,
        )
        try:
            while True:
                try:
                    component = await asyncio.wait_for(agen.__anext__(), timeout=AGENT_STEP_TIMEOUT)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning("Agent timeout — paso excedió %.0f s", AGENT_STEP_TIMEOUT)
                    raise TimeoutError("timeout")

                text_output = extract_text_from_component(component)
                if not text_output:
                    continue

                if '<table' in text_output:
                    table_seen = True
                    tabla_chunks.append(text_output)
                elif table_seen:
                    if not _is_vague_analysis(text_output):
                        post_table_chunks.append(text_output)
                        response_chunks.append(text_output)
                        yield text_output
                        if len(text_output.strip()) > 20:
                            meaningful_post_analysis = True
                else:
                    pre_table_buffer.append(text_output)
        finally:
            await agen.aclose()

    try:
        async for chunk in _step_agent(conversation_id):
            yield chunk
    except TimeoutError:
        yield ERROR_RETRY_SENTINEL + "La consulta tardó demasiado. Puedes intentarlo de nuevo."
        return
    except Exception as exc:
        # Historial corrupto por tool_call sin respuesta → limpiar y reintentar una vez
        if "tool_call" in str(exc).lower() and conversation_id and not retry:
            logger.warning(
                "Historial corrupto (tool_calls sin respuesta) — limpiando conv %s y reintentando",
                conversation_id,
            )
            try:
                await agent.conversation_store.delete_conversation(conversation_id)
            except Exception:
                pass
            try:
                async for chunk in _step_agent(None):
                    yield chunk
            except Exception:
                logger.exception("Error en reintento tras limpiar conversación")
                yield ERROR_RETRY_SENTINEL + "No pude procesar tu pregunta en este momento. Intenta de nuevo."
                return
        else:
            logger.exception("Error en agent.send_message")
            yield ERROR_RETRY_SENTINEL + "No pude procesar tu pregunta en este momento. Intenta de nuevo."
            return

    logger.info(
        "Buffers — pre_table:%d tabla:%d post_table:%d meaningful=%s",
        len(pre_table_buffer), len(tabla_chunks), len(post_table_chunks), meaningful_post_analysis,
    )
    if not post_table_chunks and table_seen:
        logger.warning("Tabla encontrada pero post_table_chunks está vacío — el LLM no generó análisis posterior")

    if table_seen and tabla_chunks:
        merged_table = _merge_multiple_tables(tabla_chunks) if len(tabla_chunks) > 1 else tabla_chunks[0]
        response_chunks.append(merged_table)

        # Si no viene gráfico, la tabla debe aparecer ANTES del análisis
        is_chart = _is_chart_question(original_question)
        if not is_chart:
            yield TABLE_FLUSH_SENTINEL  # señal para app.py: emitir tabla ya
        yield merged_table

        if meaningful_post_analysis:
            # Análisis sustancial ya emitido en el loop; solo registrar para cache
            pass
        elif captured_sql and state.SQL_RUNNER:
            # Sin análisis sustancial (vacío, trivial o vago): streaming fallback
            if post_table_chunks:
                logger.warning("Análisis del LLM trivial/vago — reemplazando con fallback streaming")
            best_sql = captured_sql[-1]
            try:
                df = await state.SQL_RUNNER.run_sql(RunSqlToolArgs(sql=best_sql), None)
                if not df.empty:
                    cols = df.columns.tolist()
                    rows = df.to_dict("records")
                    analysis_text = await _generate_analysis(original_question, rows, cols)
                    if analysis_text:
                        yield analysis_text
                        response_chunks.append(analysis_text)
                        logger.info("Análisis de respaldo generado correctamente")
            except Exception:
                logger.exception("Error generando análisis de respaldo")
    elif captured_sql and state.SQL_RUNNER:
        # El stream produjo output vacío pero el SQL fue ejecutado — fallback completo
        logger.warning("Stream vacío con SQL capturado — activando fallback de renderizado")
        df = None
        last_sql_error: str = ""
        for sql_attempt in reversed(captured_sql):
            try:
                tool_args = RunSqlToolArgs(sql=sql_attempt)
                df = await state.SQL_RUNNER.run_sql(tool_args, None)
                logger.info("Fallback SQL exitoso (intento con SQL más reciente)")
                break
            except Exception as e:
                last_sql_error = str(e)
                logger.warning("Fallback SQL falló, probando anterior: %s", e)

        # Si todos los intentos fallaron con error SQL, pedir corrección al LLM
        if df is None and last_sql_error and captured_sql:
            logger.warning("SQL error — solicitando corrección LLM: %s", last_sql_error[:200])
            corrected = await _get_corrected_sql(
                original_question,
                captured_sql[-1],
                f"Error SQL de SQL Server: {_extract_sql_error_detail(last_sql_error)}",
            )
            if corrected:
                try:
                    df = await state.SQL_RUNNER.run_sql(RunSqlToolArgs(sql=corrected), None)
                    captured_sql[-1] = corrected
                    _evict_sql_cache(original_question)
                    logger.info("Corrección LLM exitosa tras error SQL (%d filas)", len(df))
                except Exception as e:
                    logger.warning("SQL corregido también falló: %s", e)

        if df is None:
            yield ERROR_RETRY_SENTINEL + "No pude obtener los datos para esa consulta. Intenta de nuevo o reformula la pregunta."
            return
        try:
            if df.empty:
                logger.warning("Fallback devolvió 0 filas — intentando corrección LLM")
                corrected_sql = await _get_corrected_sql(
                    original_question,
                    captured_sql[-1],
                    "La consulta devolvió 0 resultados — revisa los JOINs y filtros",
                )
                if corrected_sql:
                    try:
                        df = await state.SQL_RUNNER.run_sql(RunSqlToolArgs(sql=corrected_sql), None)
                        if not df.empty:
                            captured_sql[-1] = corrected_sql
                            _evict_sql_cache(original_question)
                            logger.info("Retry exitoso tras corrección LLM (0 filas → %d filas)", len(df))
                    except Exception as e:
                        logger.warning("Retry SQL también falló: %s", e)

            if not df.empty:
                cols = df.columns.tolist()
                rows = df.to_dict("records")

                if len(cols) > 1 and len(rows) > 1:
                    html = ['<table class="data-table"><thead><tr>']
                    for col in cols:
                        html.append(f'<th>{_safe_str(col)}</th>')
                    html.append('</tr></thead><tbody>')
                    for r in rows[:state.MAX_ROWS_LIMIT]:
                        html.append('<tr>')
                        for c in cols:
                            html.append(f'<td>{_format_cell(r.get(c, ""), c)}</td>')
                        html.append('</tr>')
                    html.append('</tbody></table>')
                    table_html = "\n".join(html)
                    response_chunks.append(table_html)
                    if not _is_chart_question(original_question):
                        yield TABLE_FLUSH_SENTINEL
                    yield table_html

                analysis_text = await _generate_analysis(original_question, rows, cols)
                if analysis_text:
                    yield analysis_text
                    response_chunks.append(analysis_text)
                    logger.info("Fallback completo emitido correctamente")
            else:
                no_results_text = ""
                async for token in _stream_no_results(original_question):
                    no_results_text += token
                    yield token
                if no_results_text:
                    response_chunks.append(no_results_text)
        except Exception:
            logger.exception("Error en fallback de renderizado")
            yield ERROR_RETRY_SENTINEL + "Tuve un problema al preparar la respuesta. Puedes intentarlo de nuevo."
    else:
        combined_pre = " ".join(pre_table_buffer)
        if pre_table_buffer and _is_vague_analysis(combined_pre) and captured_sql and state.SQL_RUNNER:
            logger.warning("Texto del LLM detectado como vago (alias de columna) — regenerando con datos reales")
            try:
                tool_args = RunSqlToolArgs(sql=captured_sql[-1])
                df = await state.SQL_RUNNER.run_sql(tool_args, None)
                if not df.empty:
                    cols = df.columns.tolist()
                    rows = df.to_dict("records")
                    analysis_text = await _generate_analysis(original_question, rows, cols)
                    if analysis_text:
                        yield analysis_text
                        response_chunks.append(analysis_text)
                        pre_table_buffer[:] = [analysis_text]
                        logger.info("Análisis de reemplazo generado correctamente")
            except Exception:
                logger.exception("Error regenerando análisis desde texto vago")
                for chunk in pre_table_buffer:
                    response_chunks.append(chunk)
                    yield chunk
        else:
            for chunk in pre_table_buffer:
                response_chunks.append(chunk)
                yield chunk

    # Guard: si el agente completó sin generar ningún contenido útil, evitar burbuja vacía.
    # También limpiar el historial: output vacío casi siempre indica historial corrupto
    # (tool_call sin tool_response) que Vanna absorbe internamente sin relanzar la excepción.
    if not response_chunks:
        logger.warning(
            "Agente sin output — posible historial corrupto, limpiando conv %s y generando fallback",
            conversation_id,
        )
        if conversation_id:
            try:
                await agent.conversation_store.delete_conversation(conversation_id)
                logger.info("Conversación %s limpiada tras output vacío", conversation_id)
            except Exception:
                pass
        intent = _ctx_intent.get() or "SQL"
        if intent == "CHAT":
            fallback = await _get_chat_response(original_question)
            yield fallback
        else:
            yield ERROR_RETRY_SENTINEL + "No pude procesar tu consulta en este momento. Por favor, intenta de nuevo."
        return

    if _is_chart_question(original_question) and captured_sql and state.SQL_RUNNER:
        try:
            tool_args = RunSqlToolArgs(sql=captured_sql[-1])
            df = await state.SQL_RUNNER.run_sql(tool_args, None)
            if not df.empty and len(df.columns) >= 2 and len(df) >= 1:
                chart_payload = _build_chart_payload(
                    original_question,
                    df.columns.tolist(),
                    df.to_dict("records"),
                )
                if chart_payload:
                    yield CHART_SENTINEL + json.dumps(chart_payload)
                    logger.info("Gráfico ECharts generado y enviado")
        except Exception:
            logger.exception("Error generando gráfico")

    # Solo cachear si el SQL produjo datos reales (tabla HTML presente).
    # Verificamos '<table' en lugar de comparar el mensaje de "sin resultados", porque
    # el mensaje varía según el idioma y el flujo, pero la tabla HTML es inequívoca.
    if captured_sql:
        combined_response = "\n".join(response_chunks)
        if "<table" in combined_response:
            _store_sql_cache(
                original_question,
                captured_sql[-1],
                full_response=combined_response,
            )
            logger.info("SQL guardado en cache para: '%s'", original_question)
        else:
            logger.info("No se guarda en cache: SQL sin resultados o respuesta vacía")
    else:
        logger.info("No se capturó SQL, no se guarda en cache")


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
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    state.SQL_RUNNER = SqlServerRunner(odbc_str)

    llm = OpenAILlmService(model=model, api_key=api_key)

    tools = ToolRegistry()
    tracking_tool = TrackingSqlTool(sql_runner=state.SQL_RUNNER)
    tools.register_local_tool(tracking_tool, access_groups=["admin", "user"])
    logger.info("TrackingSqlTool registrado")

    agent_memory = ChromaAgentMemory(
        persist_directory=path_memory,
        collection_name=os.getenv("CHROMA_COLLECTION") or "vanna_tool_memory",
    )

    state.SCHEMA_STORE = SchemaVectorStore(
        persist_dir=path_schema,
        collection_name=os.getenv("SCHEMA_COLLECTION") or "tiara_schema",
        embedding_mode="default",
    )

    try:
        state.SCHEMA_META = analyze_schema()
        logger.info(
            "Schema analizado: %d tablas | union_facts=%s | date=%s",
            len(state.SCHEMA_META.tables),
            state.SCHEMA_META.union_fact_tables,
            state.SCHEMA_META.date_table,
        )

        state._PK_GRAIN_TABLE.clear()
        for tname, tmeta in state.SCHEMA_META.tables.items():
            if len(tmeta.pks) == 1:
                state._PK_GRAIN_TABLE[tmeta.pks[0].lower()] = tname.lower()
        logger.info("PK grain table cargada: %d columnas clave únicas", len(state._PK_GRAIN_TABLE))
    except Exception:
        logger.exception("Error analizando schema — el agente usará fallbacks hardcodeados")
        state.SCHEMA_META = None

    # Re-ingest automático si el esquema cambió o el store está vacío
    try:
        from .ingest_schema import check_and_reingest_if_changed
        check_and_reingest_if_changed(
            target_store=state.SCHEMA_STORE, verbose=True, schema_meta=state.SCHEMA_META,
        )
    except Exception:
        logger.exception("Error en check_and_reingest — el agente continuará sin RAG actualizado")

    # Cargar FKs en memoria para validación dinámica de JOINs — viene directo de
    # state.SCHEMA_META (ya analizado arriba), sin tocar la BD ni el LLM. Los docs
    # de join path/chain ya quedaron indexados por check_and_reingest_if_changed
    # arriba cuando el esquema cambió; no hace falta reconstruirlos en cada arranque.
    try:
        state._FK_COL_SET.clear()
        for tname, tmeta in state.SCHEMA_META.tables.items():
            for col, _, ref_table, ref_col in tmeta.fk_refs:
                state._FK_COL_SET.add((tname.lower(), col.lower()))
                state._FK_COL_SET.add((ref_table.lower(), ref_col.lower()))
        logger.info("FK col set cargado: %d pares (tabla, columna) únicos", len(state._FK_COL_SET))
    except Exception:
        logger.exception("Error cargando FK col set en memoria")

    return Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=SimpleUserResolver(),
        agent_memory=agent_memory,
        # temperature=0: la generación de SQL debe ser lo más determinista posible —
        # un mismo pregunta no debería producir SQL distinto (y por tanto resultados
        # distintos) entre un intento y otro. max_tool_iterations más bajo que el
        # default (10): si el agente no resuelve la consulta en pocos intentos, es
        # mejor que falle rápido con un mensaje claro que agotar reintentos costosos.
        config=AgentConfig(temperature=0, max_tool_iterations=6),
    )
