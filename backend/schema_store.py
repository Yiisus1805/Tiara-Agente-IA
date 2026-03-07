from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class SchemaVectorStore:
    """
    Vector store persistente (ChromaDB) para conocimiento técnico de esquema/DDL.

    - Guarda documentos como:
        "Tabla dbo.Ventas. Columnas: cliente_id (int), fecha_venta (date), total (decimal)..."

    - Recupera documentos relevantes por similitud semántica (embeddings locales).
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "tiara_schema",
        embedding_mode: str = "default",
    ):
        # Aseguramos que el directorio exista antes de inicializar el cliente
        os.makedirs(persist_dir, exist_ok=True)

        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # Inicialización del cliente persistente
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )

        # Usamos la función de embedding por defecto (all-MiniLM-L6-v2)
        if embedding_mode != "default":
            raise ValueError("Para esquema local, usa embedding_mode='default'.")

        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # Obtener o crear colección.
        try:
            # Intentamos obtener la colección existente
            self.col = self.client.get_collection(
                name=collection_name, 
                embedding_function=self.embedding_function
            )
        except Exception:
            # Si no existe, la creamos
            self.col = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "TIARA schema/DDL vector memory"},
            )

    def upsert(self, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Inserta o actualiza documentos en la base vectorial."""
        self.col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self,
        query_text: str,
        k: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Busca los fragmentos de esquema más relevantes.
        Devuelve lista de hits:
            [{"id": ..., "doc": ..., "meta": ..., "distance": ...}, ...]
        """
        query_text = (query_text or "").strip()
        if not query_text:
            return []

        # CORRECCIÓN: Eliminamos "ids" de include porque se devuelven siempre por defecto
        # en las versiones nuevas de ChromaDB.
        res = self.col.query(
            query_texts=[query_text],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Extraemos las listas de la respuesta (Chroma devuelve listas de listas)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0] # Los IDs se recuperan así

        out: List[Dict[str, Any]] = []
        # El zip ahora tiene todos los elementos necesarios
        for id_, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append({"id": id_, "doc": doc, "meta": meta, "distance": dist})
        
        return out

    def count(self) -> int:
        """Devuelve el número de documentos cargados en la colección."""
        try:
            return int(self.col.count())
        except Exception:
            return -1

