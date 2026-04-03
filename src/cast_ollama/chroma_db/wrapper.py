from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
except Exception as exc:  # pragma: no cover - optional dependency safety
    chromadb = None
    Settings = None
    CHROMA_IMPORT_ERROR = exc
else:
    CHROMA_IMPORT_ERROR = None

from cast_ollama.config import Config

logger = logging.getLogger(__name__)


class ChromaDBWrapper:
    _instance = None

    def __new__(cls):
        configured_path = Path(
            getattr(Config, "CHROMA_PERSIST_DIR", None) or getattr(Config, "CHROMA_PATH", ".cast_ollama/chroma")
        ).expanduser()
        if cls._instance is None or cls._instance._path != configured_path:
            cls._instance = super(ChromaDBWrapper, cls).__new__(cls)
            cls._instance._path = configured_path
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if chromadb is None or Settings is None:
            raise RuntimeError(f"chromadb is unavailable: {CHROMA_IMPORT_ERROR}")
        self._path.mkdir(parents=True, exist_ok=True)
        settings = Settings(allow_reset=True, is_persistent=True, persist_directory=str(self._path))
        self._client = chromadb.PersistentClient(path=str(self._path), settings=settings)
        self._collection = self._client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Persistent ChromaDB initialized at %s", self._path)

    def get_collection(self):
        return self._collection

    def reset(self):
        try:
            self._client.delete_collection(Config.CHROMA_COLLECTION)
        except Exception:
            logger.debug("Chroma collection %s did not exist during reset", Config.CHROMA_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )


# Standalone functions matching oracle_db.operations signature

def insert_chunk(
    chunk_id: str,
    file_path: str,
    chunk_content: str,
    metadata: Dict,
    embedding: List[float],
    chunking_method: str,
):
    db = ChromaDBWrapper()
    collection = db.get_collection()

    meta_copy = metadata.copy()
    meta_copy["dependencies"] = json.dumps(meta_copy.get("dependencies", []))
    meta_copy["chunking_method"] = chunking_method
    meta_copy["file_path"] = file_path
    meta_copy = {key: value for key, value in meta_copy.items() if value is not None}

    if not chunk_id:
        chunk_id = str(uuid.uuid4())

    collection.add(
        documents=[chunk_content],
        metadatas=[meta_copy],
        ids=[chunk_id],
        embeddings=[embedding],
    )
    logger.debug("Inserted chunk %s to ChromaDB", chunk_id)


def bulk_insert_chunks(chunks: List[Dict]):
    db = ChromaDBWrapper()
    collection = db.get_collection()

    documents = []
    metadatas = []
    ids = []
    embeddings = []

    for chunk in chunks:
        meta = chunk["metadata"].copy()
        meta["dependencies"] = json.dumps(meta.get("dependencies", []))
        meta["chunking_method"] = chunk["chunking_method"]
        meta["file_path"] = chunk["file_path"]
        meta["chunk_type"] = meta.get("chunk_type")
        meta["function_name"] = meta.get("function_name")
        meta["parent_class"] = meta.get("parent_class")
        meta["complexity"] = meta.get("complexity")
        meta = {key: value for key, value in meta.items() if value is not None}

        documents.append(chunk["chunk_content"])
        metadatas.append(meta)
        ids.append(chunk["chunk_id"])
        embeddings.append(
            chunk["embedding"].tolist() if hasattr(chunk["embedding"], "tolist") else chunk["embedding"]
        )

    collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
    logger.info("Bulk inserted %s chunks to ChromaDB", len(chunks))


def search_by_vector(
    query_embedding: Any,
    limit: int = 150,
    chunking_method: Optional[str] = None,
) -> List[Dict]:
    db = ChromaDBWrapper()
    collection = db.get_collection()

    where_filter = {"chunking_method": chunking_method} if chunking_method else None
    q_emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=limit,
        where=where_filter,
    )

    if not results["ids"]:
        return []

    transformed_results = []
    for index, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][index]
        deps = meta.get("dependencies", "[]")
        try:
            deps_list = json.loads(deps)
        except Exception:
            deps_list = []

        transformed_results.append(
            {
                "chunk_id": chunk_id,
                "file_path": meta.get("file_path"),
                "chunk_content": results["documents"][0][index],
                "chunk_type": meta.get("chunk_type"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "function_name": meta.get("function_name"),
                "class_name": meta.get("class_name"),
                "symbol_name": meta.get("symbol_name"),
                "symbol_path": meta.get("symbol_path"),
                "parent_class": meta.get("parent_class"),
                "docstring": meta.get("docstring"),
                "purpose": meta.get("purpose"),
                "dependencies": deps_list,
                "complexity": meta.get("complexity"),
                "chunking_method": meta.get("chunking_method"),
                "metadata_json": meta,
                "distance": results["distances"][0][index],
            }
        )

    return transformed_results


def get_chunk_by_id(chunk_id: str) -> Optional[Dict]:
    db = ChromaDBWrapper()
    collection = db.get_collection()

    result = collection.get(ids=[chunk_id], include=["metadatas", "documents"])
    if not result["ids"]:
        return None

    meta = result["metadatas"][0]
    deps = meta.get("dependencies", "[]")
    try:
        deps_list = json.loads(deps)
    except Exception:
        deps_list = []

    return {
        "chunk_id": result["ids"][0],
        "file_path": meta.get("file_path"),
        "chunk_content": result["documents"][0],
        "chunk_type": meta.get("chunk_type"),
        "start_line": meta.get("start_line"),
        "end_line": meta.get("end_line"),
        "function_name": meta.get("function_name"),
        "class_name": meta.get("class_name"),
        "symbol_name": meta.get("symbol_name"),
        "symbol_path": meta.get("symbol_path"),
        "parent_class": meta.get("parent_class"),
        "docstring": meta.get("docstring"),
        "purpose": meta.get("purpose"),
        "dependencies": deps_list,
        "complexity": meta.get("complexity"),
        "chunking_method": meta.get("chunking_method"),
        "metadata_json": meta,
    }


def delete_all_chunks():
    db = ChromaDBWrapper()
    db.reset()
    logger.info("All chunks deleted from ChromaDB.")
