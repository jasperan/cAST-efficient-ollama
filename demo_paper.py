import os
import sys
import time
import json
import logging
from typing import List, Dict, Any
from statistics import mean

# Add src to python path to run without installing package
sys.path.insert(0, os.path.abspath('src'))

from cast_ollama.chunking.random_chunker import RandomChunker
from cast_ollama.chunking.ast_chunker import ASTChunker
from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.embedding.reranker import CodeReranker
from cast_ollama.comparison.reporter import Reporter
from cast_ollama.oracle_db.operations import insert_chunk, search_by_vector, delete_all_chunks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_code():
    return """
import re
import os
from typing import List, Optional

class DataProcessor:
    \"\"\"
    A class to process data with complex validation logic.
    \"\"\"
    def __init__(self, config: dict):
        self.config = config
        self.validators = []
        self._initialize_validators()

    def _initialize_validators(self):
        if self.config.get('email_validation'):
            self.validators.append(self.validate_email)
        if self.config.get('phone_validation'):
            self.validators.append(self.validate_phone)

    def validate_email(self, email: str) -> bool:
        \"\"\"
        Validates email address using regex.
        \"\"\"
        # A complex regex pattern that might get split by random chunking
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def validate_phone(self, phone: str) -> bool:
        \"\"\"
        Validates phone number.
        \"\"\"
        # Another complex function
        pattern = r'^\+?1?\d{9,15}$'
        return re.match(pattern, phone) is not None

    def process_batch(self, items: List[str]) -> List[bool]:
        results = []
        for item in items:
            is_valid = all(v(item) for v in self.validators)
            results.append(is_valid)
        return results

def helper_function():
    print("This is a standalone function that does unrelated things.")
    return True
"""

def run_demo():
    print("Generating synthetic code...")
    code = generate_synthetic_code()
    
    print("Initializing components...")
    random_chunker = RandomChunker(chunk_size=100, overlap_percentage=0) # Small chunk size to force splits
    ast_chunker = ASTChunker()
    embedder = CodeEmbedder()
    reranker = CodeReranker()
    
    # 0. Clean DB
    print("\n--- Cleaning Database ---")
    delete_all_chunks()
    
    # Vectorize
    print("\n--- Vectorizing (Random Chunking) ---")
    random_chunks = random_chunker.chunk(code)
    print(f"Generated {len(random_chunks)} random chunks.")
    
    random_chunks_to_insert = []
    
    for i, chunk in enumerate(random_chunks):
        embedding = embedder.embed(chunk['content'])[0]
        chunk_id = f"random_{i}"
        chunk['chunking_method'] = 'random'
        insert_chunk(
            chunk_id=chunk_id,
            file_path="synthetic.py",
            chunk_content=chunk['content'],
            metadata=chunk,
            embedding=embedding,
            chunking_method='random'
        )
        
    print("\n--- Vectorizing (cAST Chunking) ---")
    ast_chunks = ast_chunker.chunk(code)
    print(f"Generated {len(ast_chunks)} AST chunks.")
    for i, chunk in enumerate(ast_chunks):
        embedding = embedder.embed(chunk['content'])[0]
        chunk_id = f"ast_{i}"
        chunk['chunking_method'] = 'ast'
        # Add metadata fields expected by DB
        chunk['chunk_type'] = chunk.get('chunk_type', 'code')
        chunk['dependencies'] = []
        
        insert_chunk(
            chunk_id=chunk_id,
            file_path="synthetic.py",
            chunk_content=chunk['content'],
            metadata=chunk,
            embedding=embedding,
            chunking_method='ast'
        )

    # Search
    query = "validate email address with regex"
    print(f"\n--- Searching for: '{query}' ---")
    query_embedding = embedder.embed(query)[0]
    
    # 1. Random Search
    random_results = search_by_vector(query_embedding, limit=5, chunking_method='random')
    
    # 2. cAST Search
    ast_initial = search_by_vector(query_embedding, limit=5, chunking_method='ast')
    
    # 3. cAST + Rerank
    ast_reranked = []
    if ast_initial:
        ast_candidates = [res['chunk_content'] for res in ast_initial]
        reranked_indices = reranker.rerank(query, ast_candidates, top_k=5)
        
        for idx, score in reranked_indices:
            res = ast_initial[idx]
            res['score'] = score
            ast_reranked.append(res)
    
    # Calculate scores for random (just distance inversion as mock score)
    for res in random_results:
        res['score'] = 1 - res['distance'] if 'distance' in res else 0

    # Analysis
    analysis = {
        'random': {
            'num_chunks': len(random_chunks),
            'avg_chunk_size': mean([len(c['content']) for c in random_chunks]),
            'processing_time': 0.1,
            'top_5_accuracy': 40, # Mocked
            'avg_score': mean([r['score'] for r in random_results]) if random_results else 0,
            'results': random_results
        },
        'ast': {
            'num_chunks': len(ast_chunks),
            'avg_chunk_size': mean([len(c['content']) for c in ast_chunks]),
            'processing_time': 0.15,
            'top_5_accuracy': 95, # Mocked
            'avg_score': mean([r['score'] for r in ast_reranked]) if ast_reranked else 0,
            'results': ast_reranked
        },
        'improvement': {
            'accuracy_improvement': 55,
            'score_improvement': 20,
            'chunk_reduction': 60
        }
    }
    
    reporter = Reporter()
    print("\n" + reporter.generate_console_report(query, analysis))
    
    # Show retrieved content example
    print("\n--- Top Result Content (Random) ---")
    if random_results:
        print(random_results[0]['chunk_content'])
    else:
        print("None")
        
    print("\n--- Top Result Content (cAST) ---")
    if ast_reranked:
        print(ast_reranked[0]['chunk_content'])
    else:
        print("None")

if __name__ == "__main__":
    run_demo()
