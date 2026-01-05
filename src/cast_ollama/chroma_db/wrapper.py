import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Optional, Any
import json
import uuid

logger = logging.getLogger(__name__)

class ChromaDBWrapper:
    _instance = None
    _client = None
    _collection = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaDBWrapper, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        try:
            # Using ephemeral client for demo/fallback purposes
            # For persistence, one would use PersistentClient
            # self._client = chromadb.PersistentClient(path="./chroma_db") 
            self._client = chromadb.Client(Settings(allow_reset=True))
            self._collection = self._client.get_or_create_collection(name="code_chunks")
            logger.info("ChromaDB initialized.")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def get_collection(self):
        return self._collection
    
    def reset(self):
        self._client.reset()
        self._collection = self._client.get_or_create_collection(name="code_chunks")


# Standalone functions matching oracle_db.operations signature

def insert_chunk(chunk_id: str, file_path: str, chunk_content: str, metadata: Dict, embedding: List[float], chunking_method: str):
    db = ChromaDBWrapper()
    collection = db.get_collection()
    
    # Flatten metadata for Chroma (vectors)
    # Chroma handles basic types, but nested JSON needs stringifying
    meta_copy = metadata.copy()
    meta_copy['dependencies'] = json.dumps(meta_copy.get('dependencies', []))
    meta_copy['chunking_method'] = chunking_method
    meta_copy['file_path'] = file_path
    
    # Remove None values as ChromaDB does not support them
    meta_copy = {k: v for k, v in meta_copy.items() if v is not None}
    
    # Ensure ID
    if not chunk_id:
        chunk_id = str(uuid.uuid4())
        
    collection.add(
        documents=[chunk_content],
        metadatas=[meta_copy],
        ids=[chunk_id],
        embeddings=[embedding]
    )
    logger.debug(f"Inserted chunk {chunk_id} to ChromaDB")

def bulk_insert_chunks(chunks: List[Dict]):
    db = ChromaDBWrapper()
    collection = db.get_collection()
    
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    for chunk in chunks:
        doc = chunk['chunk_content']
        emb = chunk['embedding'].tolist() if hasattr(chunk['embedding'], 'tolist') else chunk['embedding']
        cid = chunk['chunk_id']
        
        meta = chunk['metadata'].copy()
        meta['dependencies'] = json.dumps(meta.get('dependencies', []))
        meta['chunking_method'] = chunk['chunking_method']
        meta['file_path'] = chunk['file_path']
        
        # Add extra fields to metadata that oracle has as columns
        meta['chunk_type'] = meta.get('chunk_type')
        meta['function_name'] = meta.get('function_name')
        meta['parent_class'] = meta.get('parent_class')
        meta['complexity'] = meta.get('complexity')
        
        # Store full metadata json logic similar to Oracle
        # metadata_json = json.dumps(chunk['metadata']) # Avoid dup if not needed, but Oracle uses it
        
        # Remove None values
        meta = {k: v for k, v in meta.items() if v is not None}

        documents.append(doc)
        metadatas.append(meta)
        ids.append(cid)
        embeddings.append(emb)
        
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    logger.info(f"Bulk inserted {len(chunks)} chunks to ChromaDB")

def search_by_vector(query_embedding: Any, limit: int = 150, chunking_method: Optional[str] = None) -> List[Dict]:
    db = ChromaDBWrapper()
    collection = db.get_collection()
    
    # Chroma filter
    where_filter = {}
    if chunking_method:
        where_filter = {"chunking_method": chunking_method}
    
    # Convert numpy to list if needed
    q_emb = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
    
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=limit,
        where=where_filter if where_filter else None
    )
    
    # Transform results to match Oracle output format
    transformed_results = []
    
    if not results['ids']:
        return []
        
    # Chroma returns lists of lists
    count = len(results['ids'][0])
    for i in range(count):
        meta = results['metadatas'][0][i]
        
        # Reconstruct dependencies list
        deps = meta.get('dependencies', '[]')
        try:
            deps_list = json.loads(deps)
        except:
            deps_list = []
            
        res = {
            'chunk_id': results['ids'][0][i],
            'file_path': meta.get('file_path'),
            'chunk_content': results['documents'][0][i],
            'chunk_type': meta.get('chunk_type'),
            'start_line': meta.get('start_line'),
            'end_line': meta.get('end_line'),
            'function_name': meta.get('function_name'),
            'parent_class': meta.get('parent_class'),
            'docstring': meta.get('docstring'),
            'purpose': meta.get('purpose'),
            'dependencies': deps_list,
            'complexity': meta.get('complexity'),
            'chunking_method': meta.get('chunking_method'),
            'metadata_json': meta, # Fallback, might not match exact structure but sufficient for demo
            'distance': results['distances'][0][i] # Cosine distance
        }
        transformed_results.append(res)
        
    return transformed_results

def get_chunk_by_id(chunk_id: str) -> Optional[Dict]:
    db = ChromaDBWrapper()
    collection = db.get_collection()
    
    result = collection.get(ids=[chunk_id], include=["metadatas", "documents"])
    
    if not result['ids']:
        return None
        
    meta = result['metadatas'][0]
    
    # Reconstruct
    deps = meta.get('dependencies', '[]')
    try:
        deps_list = json.loads(deps)
    except:
        deps_list = []
        
    return {
        'chunk_id': result['ids'][0],
        'file_path': meta.get('file_path'),
        'chunk_content': result['documents'][0],
        'chunk_type': meta.get('chunk_type'),
        'start_line': meta.get('start_line'),
        'end_line': meta.get('end_line'),
        'function_name': meta.get('function_name'),
        'parent_class': meta.get('parent_class'),
        'docstring': meta.get('docstring'),
        'purpose': meta.get('purpose'),
        'dependencies': deps_list,
        'complexity': meta.get('complexity'),
        'chunking_method': meta.get('chunking_method'),
        'metadata_json': meta
    }

def delete_all_chunks():
    db = ChromaDBWrapper()
    db.reset() # This is the easiest way to clear in-memory/allow_reset mode
    logger.info("All chunks deleted from ChromaDB.")
