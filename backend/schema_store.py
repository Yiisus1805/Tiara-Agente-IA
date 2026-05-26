from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class SchemaVectorStore:
    """
    Vector store para almacenar conocimiento del esquema SQL
    optimizado para RAG SQL agents.
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "tiara_schema",
        embedding_mode: str = "default",
    ):

        os.makedirs(persist_dir, exist_ok=True)

        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        if embedding_mode != "default":
            raise ValueError(
                "Para esquema local usa embedding_mode='default'"
            )

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )

        # Si la colección existente no tiene hnsw:space=cosine, la recreamos
        try:
            existing = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
            existing_space = (existing.metadata or {}).get("hnsw:space", "l2")
            if existing_space != "cosine":
                self.client.delete_collection(collection_name)
                raise Exception("Recrear con cosine")
            self.col = existing

        except ValueError:
            # Conflicto de embedding function — borrar y recrear
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
            self.col = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "TIARA SQL schema vector store",
                    "hnsw:space": "cosine",
                },
            )

        except Exception:
            try:
                self.col = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={
                        "description": "TIARA SQL schema vector store",
                        "hnsw:space": "cosine",
                    },
                )
            except Exception:
                # Si ya existe, obtenerla sin validar embedding
                self.col = self.client.get_collection(name=collection_name)

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:

        if not ids:
            return

        self.col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )


    def query(
        self,
        query_text: str,
        k: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        query_text = (query_text or "").strip()

        if not query_text:
            return []

        # Chroma no acepta n_results mayor al número de documentos
        total = self.count()
        if total <= 0:
            return []
        n_results = min(k, total)

        res = self.col.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        results: List[Dict[str, Any]] = []

        for id_, doc, meta, dist in zip(ids, docs, metas, dists):

            # Distancia cosine en Chroma va de 0 a 2:
            #   0   = vectores idénticos  → similitud 1.0
            #   1   = ortogonales         → similitud 0.5
            #   2   = opuestos            → similitud 0.0
            similarity = 1.0 - (dist / 2.0) if dist is not None else 0.0

            results.append(
                {
                    "id": id_,
                    "doc": doc,
                    "meta": meta,
                    "distance": dist,
                    "score": similarity,
                }
            )

        return results


    def smart_query(
        self,
        query_text: str,
        k_fetch: int = 12,
        k_final: int = 6,
    ) -> List[Dict[str, Any]]:

        hits = self.query(query_text, k=k_fetch)

        if not hits:
            return []

        # Ordenar por similitud descendente
        hits.sort(key=lambda x: x["score"], reverse=True)

        seen_tables = set()
        filtered: List[Dict[str, Any]] = []

        for h in hits:

            meta = h.get("meta") or {}
            table = meta.get("table")

            if table and table in seen_tables:
                continue

            if table:
                seen_tables.add(table)

            filtered.append(h)

            if len(filtered) >= k_final:
                break

        return filtered


    def count(self) -> int:

        try:
            return int(self.col.count())
        except Exception:
            return -1

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            self.col = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": "TIARA SQL schema vector store",
                    "hnsw:space": "cosine",
                },
            )
        except Exception as e:
            raise RuntimeError(f"Error reseteando colección: {e}") from e


    def preview(self, n: int = 5):

        try:

            res = self.col.get(limit=n)

            docs = res.get("documents", [])

            for i, d in enumerate(docs):
                print(f"\n--- DOC {i+1} ---\n")
                print(d[:500])

        except Exception:
            print("No se pudo mostrar preview")