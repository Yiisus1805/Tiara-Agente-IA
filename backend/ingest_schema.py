import os
import hashlib
from urllib.parse import quote_plus
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from backend.schema_store import SchemaVectorStore


def stable_id(s: str) -> str:
    """
    Genera un ID estable basado en hash.
    Esto evita duplicados cuando re-ingestamos el schema.
    """
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def build_semantic_doc(table_key: str, columns: list) -> str:
    """
    Construye un documento semántico optimizado para RAG.
    Esto mejora la generación de SQL por el LLM.
    """
    lines = []
    lines.append(f"Tabla: {table_key}")
    lines.append("")
    lines.append("Columnas:")

    for _, col, dtype in columns:
        lines.append(f"- {col} ({dtype})")

    return "\n".join(lines)


def main() -> None:
    # ================================
    # Cargar .env desde la raíz
    # ================================
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    odbc_str = os.getenv("SQLSERVER_ODBC")

    if not odbc_str:
        raise RuntimeError("❌ SQLSERVER_ODBC no encontrada en el archivo .env")

    # ================================
    # Ruta de persistencia Chroma
    # ================================
    project_root = Path(__file__).resolve().parents[1]

    persist_dir = project_root / "backend" / "vanna_chromadb" / "schema_store"
    persist_dir.mkdir(parents=True, exist_ok=True)

    collection_name = os.getenv("SCHEMA_COLLECTION") or "tiara_schema"

    print(f"📦 Usando colección Chroma: {collection_name}")
    print(f"📂 Directorio persistente: {persist_dir}")

    store = SchemaVectorStore(
        persist_dir=str(persist_dir),
        collection_name=collection_name,
        embedding_mode="default",
    )

    # ================================
    # Conexión SQL Server
    # ================================
    print("🔌 Conectando a SQL Server...")

    engine = create_engine(
        "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_str),
        pool_pre_ping=True,
        future=True,
    )

    # ================================
    # Query para obtener schema
    # ================================
    sql = """
    SELECT 
        TABLE_SCHEMA,
        TABLE_NAME,
        COLUMN_NAME,
        DATA_TYPE,
        ORDINAL_POSITION
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA NOT IN ('information_schema', 'sys')
    ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """

    print("📊 Extrayendo metadatos del esquema...")

    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()

    if not rows:
        print("⚠️ No se encontraron tablas en la base de datos.")
        return

    tables = {}

    for schema, table, col, dtype, pos in rows:
        key = f"{schema}.{table}"
        tables.setdefault(key, []).append((int(pos), str(col), str(dtype)))

    print(f"📚 Tablas encontradas: {len(tables)}")

    ids = []
    docs = []
    metas = []

    for table_key, cols in tables.items():

        cols_sorted = sorted(cols, key=lambda x: x[0])

        doc = build_semantic_doc(table_key, cols_sorted)

        doc_id = stable_id("schema:" + table_key)

        ids.append(doc_id)
        docs.append(doc)

        metas.append({
            "type": "schema",
            "table": table_key
        })

    if ids:

        print("💾 Insertando documentos en ChromaDB...")

        store.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas
        )

        print("")
        print("✅ Ingesta de schema completada")
        print(f"📊 Tablas procesadas: {len(tables)}")
        print(f"📄 Documentos insertados: {len(ids)}")
        print(f"📦 Total documentos en colección: {store.count()}")

    else:
        print("⚠️ No se generaron documentos para ingestar.")


if __name__ == "__main__":
    main()