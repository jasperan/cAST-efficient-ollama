from __future__ import annotations

import logging

from cast_ollama.config import Config
from cast_ollama.oracle_db.connection import DBConnection, oracledb

logger = logging.getLogger(__name__)


def create_tables():
    if oracledb is None:
        raise RuntimeError("oracledb is not installed; setup requires the Oracle backend.")

    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        create_table_sql = f"""
        CREATE TABLE code_chunks (
            chunk_id VARCHAR2(255) PRIMARY KEY,
            file_path VARCHAR2(512),
            chunk_content CLOB,
            chunk_type VARCHAR2(50),
            start_line NUMBER,
            end_line NUMBER,
            function_name VARCHAR2(255),
            parent_class VARCHAR2(255),
            docstring VARCHAR2(2000),
            purpose VARCHAR2(2000),
            dependencies CLOB,
            complexity VARCHAR2(20),
            embedding VECTOR({Config.EMBEDDING_DIMENSION}, FLOAT32),
            chunking_method VARCHAR2(20),
            created_at TIMESTAMP DEFAULT SYSDATE,
            metadata_json CLOB
        )
        """
        cursor.execute(create_table_sql)

        create_index_sql = """
        CREATE VECTOR INDEX code_chunks_embedding_idx ON code_chunks(embedding)
        """
        cursor.execute(create_index_sql)

        conn.commit()
        logger.info("Table and index created successfully.")
    except oracledb.Error as exc:
        logger.error("Error creating tables: %s", exc)
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
