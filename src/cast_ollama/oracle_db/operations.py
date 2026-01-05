import oracledb
from cast_ollama.oracle_db.connection import DBConnection
import json
import logging
from typing import List, Dict, Optional
import numpy as np
import sys

# Attempt generic fallback import if needed
try:
    from cast_ollama.chroma_db import wrapper as chroma_wrapper
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global flag to track backend availability
USE_ORACLE = True

def check_backend():
    global USE_ORACLE
    try:
        # Simple check if we can import oracledb and if connection works?
        # For now, just rely on ImportError context or connection failure
        # In this environment, oracledb package might exist but connection fails, or package missing.
        # If 'oracledb' module is mocked or fails import, we switch.
        pass
    except:
        pass

# Determine backend preference at module load time or first call
try:
    # Try to initialize a pool or check connection? 
    # Or lazily handle it. Lazily is safer.
    pass
except:
    USE_ORACLE = False

def _use_fallback():
    global USE_ORACLE
    if not USE_ORACLE:
        return True
    
    # Try getting a connection to verify
    try:
        db = DBConnection()
        conn = db.get_connection()
        conn.close()
        return False
    except Exception as e:
        logger.warning(f"Oracle DB connection failed ({e}). Switching to ChromaDB fallback.")
        USE_ORACLE = False
        return True

def insert_chunk(chunk_id: str, file_path: str, chunk_content: str, metadata: Dict, embedding: np.ndarray, chunking_method: str):
    if _use_fallback():
        if CHROMA_AVAILABLE:
            return chroma_wrapper.insert_chunk(chunk_id, file_path, chunk_content, metadata, embedding.tolist(), chunking_method)
        else:
            raise RuntimeError("Oracle unavailable and ChromaDB fallback not found.")

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        # Prepare metadata
        metadata_json = json.dumps(metadata)
        dependencies = json.dumps(metadata.get('dependencies', []))

        insert_sql = """
        INSERT INTO code_chunks (
            chunk_id, file_path, chunk_content, chunk_type, start_line, end_line,
            function_name, parent_class, docstring, purpose, dependencies, complexity,
            embedding, chunking_method, metadata_json
        ) VALUES (
            :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15
        )
        """
        cursor.execute(insert_sql, (
            chunk_id, file_path, chunk_content, metadata.get('chunk_type'),
            metadata.get('start_line'), metadata.get('end_line'),
            metadata.get('function_name'), metadata.get('parent_class'),
            metadata.get('docstring'), metadata.get('purpose'), dependencies,
            metadata.get('complexity'), embedding.tolist(), chunking_method, metadata_json
        ))

        conn.commit()
        logger.info(f"Inserted chunk {chunk_id}")
    except oracledb.Error as e:
        logger.error(f"Error inserting chunk {chunk_id}: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def bulk_insert_chunks(chunks: List[Dict]):
    if _use_fallback():
        if CHROMA_AVAILABLE:
            # Chroma expects embedding list, Oracle ops usually got ndarray. Conversion handled in wrapper.
            return chroma_wrapper.bulk_insert_chunks(chunks)
        else:
            raise RuntimeError("Oracle unavailable and ChromaDB fallback not found.")

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        data = []
        for chunk in chunks:
            metadata = chunk['metadata']
            metadata_json = json.dumps(metadata)
            dependencies = json.dumps(metadata.get('dependencies', []))
            embedding_list = chunk['embedding'].tolist() if isinstance(chunk['embedding'], np.ndarray) else chunk['embedding']

            data.append((
                chunk['chunk_id'], chunk['file_path'], chunk['chunk_content'], metadata.get('chunk_type'),
                metadata.get('start_line'), metadata.get('end_line'),
                metadata.get('function_name'), metadata.get('parent_class'),
                metadata.get('docstring'), metadata.get('purpose'), dependencies,
                metadata.get('complexity'), embedding_list, chunk['chunking_method'], metadata_json
            ))

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
        logger.info(f"Bulk inserted {len(chunks)} chunks")
    except oracledb.Error as e:
        logger.error(f"Error in bulk insert: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def search_by_vector(query_embedding: np.ndarray, limit: int = 150, chunking_method: Optional[str] = None) -> List[Dict]:
    if _use_fallback():
        if CHROMA_AVAILABLE:
            return chroma_wrapper.search_by_vector(query_embedding, limit, chunking_method)
        else:
            raise RuntimeError("Oracle unavailable and ChromaDB fallback not found.")

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
            result = {
                'chunk_id': row[0],
                'file_path': row[1],
                'chunk_content': row[2],
                'chunk_type': row[3],
                'start_line': row[4],
                'end_line': row[5],
                'function_name': row[6],
                'parent_class': row[7],
                'docstring': row[8],
                'purpose': row[9],
                'dependencies': json.loads(row[10]),
                'complexity': row[11],
                'chunking_method': row[12],
                'metadata_json': json.loads(row[13]),
                'distance': row[14]
            }
            results.append(result)

        return results
    except oracledb.Error as e:
        logger.error(f"Error in vector search: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_chunk_by_id(chunk_id: str) -> Optional[Dict]:
    if _use_fallback():
        if CHROMA_AVAILABLE:
            return chroma_wrapper.get_chunk_by_id(chunk_id)
        else:
            raise RuntimeError("Oracle unavailable and ChromaDB fallback not found.")

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
                'chunk_id': row[0],
                'file_path': row[1],
                'chunk_content': row[2],
                'chunk_type': row[3],
                'start_line': row[4],
                'end_line': row[5],
                'function_name': row[6],
                'parent_class': row[7],
                'docstring': row[8],
                'purpose': row[9],
                'dependencies': json.loads(row[10]),
                'complexity': row[11],
                'chunking_method': row[12],
                'metadata_json': json.loads(row[13])
            }
        return None
    except oracledb.Error as e:
        logger.error(f"Error getting chunk {chunk_id}: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def delete_all_chunks():
    if _use_fallback():
        if CHROMA_AVAILABLE:
            return chroma_wrapper.delete_all_chunks()
        else:
            raise RuntimeError("Oracle unavailable and ChromaDB fallback not found.")

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM code_chunks")
        conn.commit()
        logger.info("All chunks deleted.")
    except oracledb.Error as e:
        logger.error(f"Error deleting chunks: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
