import os
import hashlib
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from schema_store import SchemaVectorStore


def stable_id(s: str) -> str:
    """ID estable para upsert (si re-ingestas, actualiza en vez de duplicar)."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main() -> None:
    # Carga backend/.env aunque ejecutes desde raíz
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

    odbc_str = os.getenv("SQLSERVER_ODBC")
    if not odbc_str:
        raise RuntimeError("SQLSERVER_ODBC no encontrada en backend/.env")

    # --- RUTA CORREGIDA ---
    # Apuntamos a la subcarpeta 'schema_store' para no chocar con la memoria del agente
    project_root = os.path.dirname(os.path.dirname(__file__))  # TIARA_PROJECT/
    persist_dir = os.path.join(project_root, "backend", "vanna_chromadb", "schema_store")
    os.makedirs(persist_dir, exist_ok=True)

    # Colección para schema
    collection_name = os.getenv("SCHEMA_COLLECTION") or "tiara_schema"

    store = SchemaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_mode="default", 
    )

    # Conexión SQL Server
    engine = create_engine(
        "mssql+pyodbc:///?odbc_connect=" + quote_plus(odbc_str),
        pool_pre_ping=True,
        future=True,
    )

    # Query para extraer metadatos de las tablas
    sql = """
    SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, ORDINAL_POSITION
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA NOT IN ('information_schema', 'sys')
    ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
    """

    print(f"Conectando a la base de datos para extraer esquema...")
    with engine.connect() as conn:
        rows = conn.execute(text(sql)).fetchall()

    # Agrupar por tabla
    tables = {}
    for schema, table, col, dtype, pos in rows:
        key = f"{schema}.{table}"
        tables.setdefault(key, []).append((int(pos), str(col), str(dtype)))

    ids = []
    docs = []
    metas = []

    for table_key, cols in tables.items():
        # Ordenar columnas por su posición original
        cols_sorted = sorted(cols, key=lambda x: x[0])
        cols_txt = ", ".join([f"{c} ({t})" for _, c, t in cols_sorted])

        # Formato de documento para el RAG
        doc = f"Tabla {table_key}. Columnas: {cols_txt}."
        doc_id = stable_id("schema:" + table_key)

        ids.append(doc_id)
        docs.append(doc)
        metas.append({"type": "schema", "table": table_key})

    # Guardar en ChromaDB
    if ids:
        store.upsert(ids=ids, documents=docs, metadatas=metas)
        print(f"[OK] Ingesta schema completada.")
        print(f"     Tablas procesadas: {len(tables)}")
        print(f"     Directorio: {persist_dir}")
        print(f"     Documentos totales en colección: {store.count()}")
    else:
        print("[!] No se encontraron tablas para ingestar. Revisa tu conexión u ODBC.")


if __name__ == "__main__":
    main()