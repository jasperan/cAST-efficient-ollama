import argparse
from cast_ollama.config import Config
from cast_ollama.oracle_db.schema import create_tables
from cast_ollama.chunking.random_chunker import RandomChunker
from cast_ollama.chunking.ast_chunker import ASTChunker
from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.metadata.extractor import OllamaMetadataExtractor
from cast_ollama.oracle_db.operations import bulk_insert_chunks, delete_all_chunks
from cast_ollama.retrieval.search import SearchPipeline
from cast_ollama.comparison.analyzer import Analyzer
from cast_ollama.comparison.reporter import Reporter
import logging
import uuid
import os

def main():
    parser = argparse.ArgumentParser(description="Code RAG AST Demo CLI")
    parser.add_argument('--action', choices=['setup', 'vectorize', 'search'], required=True, help="Action to perform")
    parser.add_argument('--chunking-method', choices=['random', 'ast', 'both'], default='both', help="Chunking method")
    parser.add_argument('--sample-file', help="Path to sample code file")
    parser.add_argument('--query', help="Search query text")
    parser.add_argument('--top-k', type=int, default=20, help="Number of top results")
    parser.add_argument('--db-user', help="Oracle DB username")
    parser.add_argument('--db-password', help="Oracle DB password")
    parser.add_argument('--db-dsn', help="Oracle DB DSN")
    parser.add_argument('--ollama-endpoint', default=Config.OLLAMA_ENDPOINT, help="Ollama endpoint")
    parser.add_argument('--ollama-model', default=Config.OLLAMA_MODEL, help="Ollama model")
    parser.add_argument('--enrich-metadata', type=bool, default=False, help="Enrich metadata with Ollama")
    parser.add_argument('--rerank', type=bool, default=False, help="Apply reranking")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # Override config if args provided
    if args.db_user:
        Config.ORACLE_USER = args.db_user
    if args.db_password:
        Config.ORACLE_PASSWORD = args.db_password
    if args.db_dsn:
        Config.ORACLE_DSN = args.db_dsn
    Config.OLLAMA_ENDPOINT = args.ollama_endpoint
    Config.OLLAMA_MODEL = args.ollama_model
    Config.VERBOSE = args.verbose

    if args.action == 'setup':
        print("Setting up Oracle tables...")
        create_tables()
        print("Setup complete.")

    elif args.action == 'vectorize':
        if not args.sample_file:
            parser.error("--sample-file is required for vectorize action")
        
        with open(args.sample_file, 'r') as f:
            code = f.read()
        
        methods = [args.chunking_method] if args.chunking_method != 'both' else ['random', 'ast']
        embedder = CodeEmbedder()
        extractor = OllamaMetadataExtractor() if args.enrich_metadata else None
        
        all_chunks_to_insert = []
        
        for method in methods:
            if method == 'random':
                chunker = RandomChunker()
            else:
                chunker = ASTChunker()
            
            chunks = chunker.chunk(code)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{method}_{uuid.uuid4().hex[:8]}_{i}"
                embedding = embedder.embed(chunk['content'])[0]
                
                if extractor:
                    meta = extractor.extract(chunk['content'])
                    chunk.update(meta.dict())
                
                all_chunks_to_insert.append({
                    'chunk_id': chunk_id,
                    'file_path': args.sample_file,
                    'chunk_content': chunk['content'],
                    'metadata': chunk,
                    'embedding': embedding,
                    'chunking_method': method
                })
        
        # Optional: delete existing chunks
        delete_all_chunks()
        
        bulk_insert_chunks(all_chunks_to_insert)
        print(f"Vectorized and stored {len(all_chunks_to_insert)} chunks.")

    elif args.action == 'search':
        if not args.query:
            parser.error("--query is required for search action")
        
        analyzer = Analyzer()
        reporter = Reporter()
        
        analysis = analyzer.analyze_query(args.query, methods=['random', 'ast'] if args.chunking_method == 'both' else [args.chunking_method])
        print(reporter.generate_console_report(args.query, analysis))
        
        # Optional exports
        reporter.export_to_csv([{'query': args.query, 'analysis': analysis}], 'search_report.csv')
        reporter.export_to_json([{'query': args.query, 'analysis': analysis}], 'search_report.json')

if __name__ == "__main__":
    main()
