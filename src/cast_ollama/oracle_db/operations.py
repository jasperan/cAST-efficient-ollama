from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np

from cast_ollama.config import Config
from cast_ollama.oracle_db.connection import DBConnection, ORACLE_IMPORT_ERROR, oracledb

try:
    from cast_ollama.chroma_db import wrapper as chroma_wrapper

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chroma_wrapper = None

logger = logging.getLogger(__name__)
USE_ORACLE = True


def _use_fallback() -> bool:
    global USE_ORACLE

    if Config.LOCAL_ONLY or str(Config.STORAGE_BACKEND).lower() == "chroma":
        USE_ORACLE = False
        return True

    if oracledb is None:
        USE_ORACLE = False
        return True

    if not USE_ORACLE:
        return True

    try:
        db = DBConnection()
        conn = db.get_connection()
        conn.close()
        return False
    except Exception as exc:
        logger.warning("Oracle DB connection failed (%s). Switching to persistent ChromaDB fallback.", exc)
        USE_ORACLE = False
        return True


def get_storage_backend() -> str:
    return "chroma-persistent" if _use_fallback() else "oracle"


def _require_fallback() -> None:
    if not CHROMA_AVAILABLE or chroma_wrapper is None:
        raise RuntimeError(
            "Oracle is unavailable and ChromaDB fallback could not be imported. "
            f"Oracle import error: {ORACLE_IMPORT_ERROR}"
        )


def insert_chunk(
    chunk_id: str,
    file_path: str,
    chunk_content: str,
    metadata: Dict,
    embedding: np.ndarray,
    chunking_method: str,
):
    if _use_fallback():
        _require_fallback()
        return chroma_wrapper.insert_chunk(
            chunk_id,
            file_path,
            chunk_content,
            metadata,
            embedding.tolist(),
            chunking_method,
        )

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        metadata_json = json.dumps(metadata)
        dependencies = json.dumps(metadata.get("dependencies", []))

        insert_sql = """
        INSERT INTO code_chunks (
            chunk_id, file_path, chunk_content, chunk_type, start_line, end_line,
            function_name, parent_class, docstring, purpose, dependencies, complexity,
            embedding, chunking_method, metadata_json
        ) VALUES (
            :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15
        )
        """
        cursor.execute(
            insert_sql,
            (
                chunk_id,
                file_path,
                chunk_content,
                metadata.get("chunk_type"),
                metadata.get("start_line"),
                metadata.get("end_line"),
                metadata.get("function_name"),
                metadata.get("parent_class"),
                metadata.get("docstring"),
                metadata.get("purpose"),
                dependencies,
                metadata.get("complexity"),
                embedding.tolist(),
                chunking_method,
                metadata_json,
            ),
        )

        conn.commit()
        logger.info("Inserted chunk %s", chunk_id)
    except oracledb.Error as exc:
        logger.error("Error inserting chunk %s: %s", chunk_id, exc)
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def bulk_insert_chunks(chunks: List[Dict]):
    if _use_fallback():
        _require_fallback()
        return chroma_wrapper.bulk_insert_chunks(chunks)

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        data = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            metadata_json = json.dumps(metadata)
            dependencies = json.dumps(metadata.get("dependencies", []))
            embedding_list = (
                chunk["embedding"].tolist()
                if isinstance(chunk["embedding"], np.ndarray)
                else chunk["embedding"]
            )

            data.append(
                (
                    chunk["chunk_id"],
                    chunk["file_path"],
                    chunk["chunk_content"],
                    metadata.get("chunk_type"),
                    metadata.get("start_line"),
                    metadata.get("end_line"),
                    metadata.get("function_name"),
                    metadata.get("parent_class"),
                    metadata.get("docstring"),
                    metadata.get("purpose"),
                    dependencies,
                    metadata.get("complexity"),
                    embedding_list,
                    chunk["chunking_method"],
                    metadata_json,
                )
            )

        insert_sql = """
        INSERT INTO code_chunks (
            chunk_id, file_path, chunk_content, chunk_type, start_line, end_line,
            function_name, parent_class, docstring, purpose, dependencies, complexity,
            embedding, chunking_method, metadata_json
        ) VALUES (
            :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15
        )
        """
        cursor.executemany(insert_sql, data)
        conn.commit()
        logger.info("Bulk inserted %s chunks", len(chunks))
    except oracledb.Error as exc:
        logger.error("Error in bulk insert: %s", exc)
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def search_by_vector(
    query_embedding: np.ndarray,
    limit: int = 150,
    chunking_method: Optional[str] = None,
) -> List[Dict]:
    if _use_fallback():
        _require_fallback()
        return chroma_wrapper.search_by_vector(query_embedding, limit, chunking_method)

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        query_embedding_list = query_embedding.tolist()
        where_clause = "WHERE chunking_method = :3" if chunking_method else ""
        params = [limit, query_embedding_list]
        if chunking_method:
            params.append(chunking_method)

        search_sql = f"""
        SELECT chunk_id, file_path, chunk_content, chunk_type, start_line, end_line,
               function_name, parent_class, docstring, purpose, dependencies, complexity,
               chunking_method, metadata_json,
               VECTOR_DISTANCE(embedding, :2, COSINE) as distance
        FROM code_chunks
        {where_clause}
        ORDER BY distance ASC
        FETCH FIRST :1 ROWS ONLY
        """
        cursor.execute(search_sql, params)

        results = []
        for row in cursor:
            results.append(
                {
                    "chunk_id": row[0],
                    "file_path": row[1],
                    "chunk_content": row[2],
                    "chunk_type": row[3],
                    "start_line": row[4],
                    "end_line": row[5],
                    "function_name": row[6],
                    "parent_class": row[7],
                    "docstring": row[8],
                    "purpose": row[9],
                    "dependencies": json.loads(row[10]),
                    "complexity": row[11],
                    "chunking_method": row[12],
                    "metadata_json": json.loads(row[13]),
                    "distance": row[14],
                }
            )

        return results
    except oracledb.Error as exc:
        logger.error("Error in vector search: %s", exc)
        raise
    finally:
        cursor.close()
        conn.close()


def get_chunk_by_id(chunk_id: str) -> Optional[Dict]:
    if _use_fallback():
        _require_fallback()
        return chroma_wrapper.get_chunk_by_id(chunk_id)

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        select_sql = """
        SELECT chunk_id, file_path, chunk_content, chunk_type, start_line, end_line,
               function_name, parent_class, docstring, purpose, dependencies, complexity,
               chunking_method, metadata_json
        FROM code_chunks
        WHERE chunk_id = :1
        """
        cursor.execute(select_sql, (chunk_id,))

        row = cursor.fetchone()
        if row:
            return {
                "chunk_id": row[0],
                "file_path": row[1],
                "chunk_content": row[2],
                "chunk_type": row[3],
                "start_line": row[4],
                "end_line": row[5],
                "function_name": row[6],
                "parent_class": row[7],
                "docstring": row[8],
                "purpose": row[9],
                "dependencies": json.loads(row[10]),
                "complexity": row[11],
                "chunking_method": row[12],
                "metadata_json": json.loads(row[13]),
            }
        return None
    except oracledb.Error as exc:
        logger.error("Error getting chunk %s: %s", chunk_id, exc)
        raise
    finally:
        cursor.close()
        conn.close()


def delete_all_chunks():
    if _use_fallback():
        _require_fallback()
        return chroma_wrapper.delete_all_chunks()

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM code_chunks")
        conn.commit()
        logger.info("All chunks deleted.")
    except oracledb.Error as exc:
        logger.error("Error deleting chunks: %s", exc)
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
