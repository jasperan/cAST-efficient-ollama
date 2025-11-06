from oracle_db.connection import DBConnection
import logging

logger = logging.getLogger(__name__)

def create_tables():
    db = DBConnection()
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        # Create table
        create_table_sql = """
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
            dependencies CLOB,  -- JSON array
            complexity VARCHAR2(20),
            embedding VECTOR(384, FLOAT32),
            chunking_method VARCHAR2(20),  -- 'random' or 'ast'
            created_at TIMESTAMP DEFAULT SYSDATE,
            metadata_json CLOB  -- Full metadata as JSON
        )
        """
        cursor.execute(create_table_sql)

        # Create vector index
        create_index_sql = """
        CREATE VECTOR INDEX code_chunks_embedding_idx ON code_chunks(embedding)
        """
        cursor.execute(create_index_sql)

        conn.commit()
        logger.info("Table and index created successfully.")
    except oracledb.Error as e:
        logger.error(f"Error creating tables: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
