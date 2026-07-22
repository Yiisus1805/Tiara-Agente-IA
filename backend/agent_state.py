"""
Estado compartido del agente — variables globales mutables que `build_agent()`
(en agent_logic.py) asigna al arrancar, y que sql_pipeline.py / agent_features.py
necesitan leer. Vive en su propio módulo para evitar imports circulares entre
agent_logic.py y los módulos que éste importa.
"""
from __future__ import annotations

from .schema_store import SchemaVectorStore
from .database import SqlServerRunner
from .schema_analyzer import SchemaMetadata

SCHEMA_STORE: SchemaVectorStore | None = None
SQL_RUNNER: SqlServerRunner | None = None
SQL_CACHE = None
SCHEMA_META: SchemaMetadata | None = None

# FK col set — cargado al arrancar desde la BD real, sin hardcoding
# Contiene (table_name_lower, col_name_lower) para todos los lados FK y PK
_FK_COL_SET: set[tuple[str, str]] = set()

# Para cada columna *Key con PK de una sola columna, la tabla cuya PK es esa
# columna (su tabla "grano"). Ej. "salesterritorykey" -> "dimsalesterritory".
# Permite detectar JOINs riesgosos contra tablas donde esa columna es solo un
# atributo denormalizado (puede repetirse) en vez de la clave única de grano.
_PK_GRAIN_TABLE: dict[str, str] = {}

RAG_K_FETCH = 15
RAG_K_FINAL = 8
SQL_CACHE_THRESHOLD = 0.98
MAX_ROWS_LIMIT = 500
MAX_RESPONSE_CACHE_LEN = 100_000
