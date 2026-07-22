from __future__ import annotations

import logging
import os

import httpx
from dotenv import load_dotenv

from backend.schema_store import SchemaVectorStore
from .schema_analyzer import (
    analyze_schema, SchemaMetadata, TableMeta,
    _NUMERIC_TYPES, _DATE_TYPES, _TEXT_TYPES, _EXCLUDED_SUFFIXES,
)

load_dotenv()
logger = logging.getLogger(__name__)

HASH_FILE = os.path.join(os.path.dirname(__file__), ".schema_hash")


def load_stored_hash() -> str:
    """Lee el hash guardado en disco. Devuelve '' si no existe."""
    try:
        with open(HASH_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def save_hash(h: str) -> None:
    """Persiste el hash del esquema en disco."""
    with open(HASH_FILE, "w") as f:
        f.write(h)


def check_and_reingest_if_changed(
    target_store=None,
    verbose: bool = True,
    schema_meta: SchemaMetadata | None = None,
) -> bool:
    """
    Calcula el hash actual del esquema de la BD y lo compara con el guardado.
    Reglas:
    - Si el store está vacío → ingest obligatorio.
    - Si no hay hash guardado pero el store tiene contenido → solo guardar el hash
      (primera vez usando este sistema, sin re-ingest innecesario).
    - Si el hash cambió → re-ingest.
    - Si el hash es igual → sin cambios.
    Retorna True si se realizó re-ingest.
    """
    try:
        if schema_meta is None:
            schema_meta = analyze_schema()
        current_hash = schema_meta.schema_hash
    except Exception:
        logger.exception("No se pudo calcular hash del esquema — se omite verificación")
        return False

    stored_hash = load_stored_hash()
    store_empty = (target_store is not None and target_store.count() == 0)

    if not stored_hash and not store_empty:
        logger.info(
            "Sin hash previo, store no vacío (%d docs) — guardando hash=%s sin re-ingest",
            target_store.count() if target_store else -1,
            current_hash,
        )
        save_hash(current_hash)
        return False

    if current_hash == stored_hash and not store_empty:
        if verbose:
            logger.info("Schema sin cambios (hash=%s) — sin re-ingest", current_hash)
        return False

    reason = "store vacío" if store_empty else f"hash cambió: {stored_hash!r} → {current_hash!r}"
    logger.info("Re-ingest del esquema: %s", reason)
    if verbose:
        print(f"\nEsquema cambió ({reason}) — ejecutando re-ingest...\n")

    ingest(target_store=target_store, schema_meta=schema_meta)
    save_hash(current_hash)
    return True


def _generate_table_concepts(tname: str, tmeta: TableMeta) -> str:
    """Genera sinónimos y conceptos de negocio (español + inglés) para una tabla,
    derivados de su nombre y columnas reales — sin asumir ningún esquema específico,
    funciona para cualquier base de datos conectada. Enriquece la búsqueda semántica
    del RAG. Retorna "" si no hay API key o la llamada falla (degradación silenciosa)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    col_names = ", ".join(c[0] for c in tmeta.columns[:25])
    prompt = (
        f"Tabla de base de datos: {tname} (tipo: {tmeta.table_type})\n"
        f"Columnas: {col_names}\n\n"
        "Genera palabras y frases en ESPAÑOL e inglés que un usuario de negocio usaría "
        "para preguntar sobre estos datos en lenguaje natural: sinónimos del nombre de "
        "la tabla, términos de negocio relacionados, preguntas típicas. "
        "Responde ÚNICAMENTE con las palabras/frases separadas por espacios, sin "
        "numeración ni explicación, máximo 60 palabras."
    )
    try:
        with httpx.Client(timeout=15) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "max_tokens": 150,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            return response.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("Error generando conceptos semánticos para %s", tname)
        return ""


def _classify_columns(columns):
    metrics, dimensions, temporals = [], [], []
    for name, typ, *_ in columns:
        t = typ.lower()
        n = name.lower()
        if "key" in n or n.endswith("id"):
            continue
        elif t in _NUMERIC_TYPES:
            metrics.append(name)
        elif t in _DATE_TYPES:
            temporals.append(name)
        elif t in _TEXT_TYPES:
            dimensions.append(name)
    return metrics, dimensions, temporals


def build_schema_doc(tname: str, tmeta: TableMeta, concepts: str = "") -> str:
    column_lines = []
    for name, typ, length, nullable in tmeta.columns:
        null_txt = "NULL" if nullable else "NOT NULL"
        column_lines.append(f"{name} ({typ}) {null_txt}")
    cols = "\n".join(column_lines)

    pk_text = ", ".join(tmeta.pks) if tmeta.pks else "None"

    rel_lines = [
        f"{tmeta.schema}.{tname}.{col} -> {ref_schema}.{ref_table}.{ref_col}"
        for col, ref_schema, ref_table, ref_col in tmeta.fk_refs
    ]
    rels = "\n".join(rel_lines) if rel_lines else "None"

    metrics, dimensions, temporals = _classify_columns(tmeta.columns)

    metric_hint = ""
    if tmeta.table_type == "Fact" and metrics:
        metric_hint = (
            f"\nMétricas agregables (SUM / AVG / COUNT / STDEV):\n"
            f"{', '.join(metrics)}\n"
            f"\nAtributos para GROUP BY / filtros:\n"
            f"{', '.join(dimensions) if dimensions else 'ver columnas *Key para JOIN'}\n"
            f"\nColumnas temporales (filtrar / agrupar por fecha):\n"
            f"{', '.join(temporals) if temporals else 'usar la columna *Key de fecha → JOIN con la tabla de tiempo'}\n"
        )
    elif tmeta.table_type == "Dimension" and (dimensions or temporals):
        metric_hint = (
            f"\nAtributos textuales (GROUP BY / filtros / etiquetas):\n"
            f"{', '.join(dimensions)}\n"
        )
        if temporals:
            metric_hint += f"\nColumnas de fecha:\n{', '.join(temporals)}\n"
    elif tmeta.table_type == "View":
        parts = []
        if metrics:
            parts.append(f"Métricas agregables (SUM / AVG / COUNT):\n{', '.join(metrics)}\n")
        if dimensions:
            parts.append(f"Atributos (GROUP BY / filtros):\n{', '.join(dimensions)}\n")
        if temporals:
            parts.append(f"Columnas de fecha:\n{', '.join(temporals)}\n")
        metric_hint = (
            "\nEsta es una VISTA (VIEW): ya combina datos de varias tablas base. "
            "Prefiérela sobre reconstruir el JOIN manualmente si cubre la pregunta.\n"
            + "\n".join(parts)
        )

    doc = f"""
Tabla: {tmeta.schema}.{tname}

Tipo de tabla: {tmeta.table_type}

Conceptos relacionados: {concepts}

Columnas:
{cols}
{metric_hint}
Primary Key:
{pk_text}

Relaciones (esta tabla referencia a):
{rels}

Uso:
- Tablas Fact contienen métricas (ventas, cantidades, montos): usar SUM, AVG, COUNT, STDEV, RANK.
- Tablas Dim contienen atributos descriptivos: usar para GROUP BY, etiquetas y filtros.
- Para estadísticas por grupo: JOIN Fact + Dim → GROUP BY columna Dim → AGG(métrica Fact).
- Para tendencias temporales: JOIN con la tabla de tiempo → GROUP BY año / trimestre.
- Para rankings: ORDER BY AGG DESC con TOP N, o usar RANK() OVER (ORDER BY AGG DESC).
- Para crecimiento YoY: subconsulta o LAG() OVER (ORDER BY año).
""".strip()

    return doc


def build_join_path_docs(schema_meta: SchemaMetadata, concepts_cache: dict[str, str]) -> list[dict]:
    """Genera documentos de join paths directos y cadenas de 2 saltos a partir
    del SchemaMetadata ya analizado (sin volver a consultar la BD ni al LLM —
    concepts_cache ya trae los conceptos de cada tabla precalculados una sola vez)."""
    docs = []

    def _is_excluded(tname: str) -> bool:
        return any(tname.endswith(s) for s in _EXCLUDED_SUFFIXES)

    fk_index: dict = {}
    all_fks: list[tuple] = []
    for tname, tmeta in schema_meta.tables.items():
        if _is_excluded(tname):
            continue
        for col, ref_schema, ref_table, ref_col in tmeta.fk_refs:
            if _is_excluded(ref_table):
                continue
            fk_index.setdefault(tname, []).append((col, ref_schema, ref_table, ref_col))
            all_fks.append((tmeta.schema, tname, col, ref_schema, ref_table, ref_col))

    def concepts(table_name: str) -> str:
        return concepts_cache.get(table_name, "")

    for p_schema, p_table, p_col, r_schema, r_table, r_col in all_fks:
        doc_id = f"rel:{p_table}.{p_col}:{r_table}.{r_col}"
        doc = (
            f"JOIN: {p_schema}.{p_table}.{p_col} → {r_schema}.{r_table}.{r_col}\n"
            f"SQL: JOIN {r_schema}.{r_table} ON {p_table}.{p_col} = {r_table}.{r_col}\n"
            f"Contexto {p_table}: {concepts(p_table)[:200]}\n"
            f"Contexto {r_table}: {concepts(r_table)[:200]}"
        )
        docs.append({
            "id": doc_id,
            "doc": doc,
            "meta": {"type": "join_path", "from_table": p_table, "to_table": r_table},
        })

    for p_schema, p_table, p_col, r_schema, r_table, r_col in all_fks:
        for r2_col, r2_schema, r2_table, r2_rcol in fk_index.get(r_table, []):
            if r2_table == p_table:
                continue
            doc_id = f"chain:{p_table}.{p_col}:{r_table}.{r2_col}:{r2_table}"
            doc = (
                f"JOIN CHAIN: {p_table} → {r_table} → {r2_table}\n"
                f"Para acceder a {r2_table} desde {p_table}:\n"
                f"  JOIN {r_schema}.{r_table} ON {p_table}.{p_col} = {r_table}.{r_col}\n"
                f"  JOIN {r2_schema}.{r2_table} ON {r_table}.{r2_col} = {r2_table}.{r2_rcol}\n"
                f"Contexto {p_table}: {concepts(p_table)[:150]}\n"
                f"Contexto {r_table}: {concepts(r_table)[:150]}\n"
                f"Contexto {r2_table}: {concepts(r2_table)[:150]}"
            )
            docs.append({
                "id": doc_id,
                "doc": doc,
                "meta": {
                    "type": "join_chain",
                    "from_table": p_table,
                    "via_table": r_table,
                    "to_table": r2_table,
                },
            })

    return docs


def ingest_join_paths(target_store, schema_meta: SchemaMetadata, concepts_cache: dict[str, str]) -> int:
    """Indexa relaciones FK como documentos independientes en el schema store."""
    docs = build_join_path_docs(schema_meta, concepts_cache)

    for d in docs:
        target_store.upsert(
            ids=[d["id"]],
            documents=[d["doc"]],
            metadatas=[d["meta"]],
        )

    logger.info("Join paths indexados: %d", len(docs))
    print(f"Join paths indexados: {len(docs)}")
    return len(docs)


def ingest(target_store=None, schema_meta: SchemaMetadata | None = None):
    if target_store is None:
        target_store = SchemaVectorStore(
            persist_dir="backend/vanna_chromadb/schema_store",
            collection_name=os.getenv("SCHEMA_COLLECTION") or "tiara_schema",
            embedding_mode="default",
        )

    if schema_meta is None:
        schema_meta = analyze_schema()

    # Las tablas de benchmark/demo (sufijos en _EXCLUDED_SUFFIXES, ej. XL_CCI,
    # XL_PageCompressed) tienen datos sintéticos inflados y NUNCA deben usarse
    # para responder preguntas de negocio (ver _validate_no_excluded_tables en
    # sql_pipeline.py) — indexarlas en el RAG solo le resta espacio relevante a
    # las tablas reales en el top-K de la búsqueda semántica, sin ningún
    # beneficio (ya están bloqueadas si el LLM las usa).
    indexable_tables = {
        tname: tmeta
        for tname, tmeta in schema_meta.tables.items()
        if not any(tname.endswith(s) for s in _EXCLUDED_SUFFIXES)
    }
    skipped = len(schema_meta.tables) - len(indexable_tables)
    logger.info("Tablas encontradas: %d (%d excluidas del RAG por ser benchmark/demo)", len(schema_meta.tables), skipped)
    print(f"\nTablas encontradas: {len(schema_meta.tables)} ({skipped} excluidas del RAG por ser benchmark/demo)\n")

    # Borrar entradas YA indexadas de tablas excluidas en ingests previos —
    # upsert no elimina documentos, solo agrega/actualiza, así que sin esto
    # los docs viejos de estas tablas seguirían compitiendo por espacio en el
    # top-K de la búsqueda semántica indefinidamente.
    excluded_names = {tname for tname in schema_meta.tables if tname not in indexable_tables}
    if excluded_names:
        try:
            existing = target_store.col.get(include=["metadatas"])
            stale_ids = [
                id_ for id_, meta in zip(existing.get("ids") or [], existing.get("metadatas") or [])
                if (meta or {}).get("table") in excluded_names
                or (meta or {}).get("from_table") in excluded_names
                or (meta or {}).get("to_table") in excluded_names
                or (meta or {}).get("via_table") in excluded_names
            ]
            if stale_ids:
                target_store.col.delete(ids=stale_ids)
                logger.info("Entradas de tablas excluidas borradas del RAG: %d", len(stale_ids))
                print(f"Entradas de tablas excluidas borradas del RAG: {len(stale_ids)}\n")
        except Exception:
            logger.exception("Error limpiando entradas de tablas excluidas del RAG")

    # Genera los conceptos semánticos de cada tabla UNA sola vez (llamada al LLM)
    # y los reutiliza tanto en su propio doc de esquema como en cualquier doc de
    # join path/chain que la mencione — evita repetir la misma llamada N veces.
    concepts_cache = {
        tname: _generate_table_concepts(tname, tmeta)
        for tname, tmeta in indexable_tables.items()
    }

    count = 0
    for tname, tmeta in indexable_tables.items():
        doc = build_schema_doc(tname, tmeta, concepts_cache.get(tname, ""))

        target_store.upsert(
            ids=[f"{tmeta.schema}.{tname}"],
            documents=[doc],
            metadatas=[{"schema": tmeta.schema, "table": tname}],
        )

        logger.info("Indexed %s.%s", tmeta.schema, tname)
        print(f"Indexed {tmeta.schema}.{tname}")
        count += 1

    logger.info("Total tablas indexadas: %d", count)
    print(f"\nTotal tablas indexadas: {count}")

    print("\nIndexando join paths y cadenas FK...\n")
    ingest_join_paths(target_store, schema_meta, concepts_cache)


if __name__ == "__main__":
    print("\nIngestando esquema → Chroma\n")
    ingest()
    print("\nSchema ingest terminado\n")
