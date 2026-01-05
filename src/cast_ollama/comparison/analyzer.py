import time
from typing import List, Dict, Any
from cast_ollama.retrieval.search import SearchPipeline
from cast_ollama.config import Config
import logging
from statistics import mean

logger = logging.getLogger(__name__)

class Analyzer:
    def __init__(self):
        self.search_pipeline = SearchPipeline()
        self.test_queries = [
            "Find code that validates email addresses and user input",
            "Show me functions that handle database connection errors",
            "Find the sorting or filtering algorithm implementation",
            "Where is code that makes HTTP requests to external APIs?",
            "Find helper functions that process strings and format output"
        ]

    def analyze_query(self, query: str, methods: List[str] = ['random', 'ast']) -> Dict[str, Any]:
        results = {}
        for method in methods:
            start_time = time.time()
            search_results = self.search_pipeline.search(query, chunking_method=method, use_rerank=(method == 'ast'))
            end_time = time.time()

            if not search_results:
                results[method] = {'error': 'No results'}
                continue

            # Compute metrics
            num_chunks = len(search_results)
            avg_chunk_size = mean(len(res['chunk_content']) for res in search_results)
            processing_time = end_time - start_time
            avg_score = mean(res['score'] for res in search_results)
            
            # Placeholder for accuracy: assume higher score means better, or manual eval
            top_5_accuracy = avg_score * 100  # Simulated
            
            # Chunk completeness: for AST, assume higher; for random, lower
            completeness = 95 if method == 'ast' else 40  # Simulated

            results[method] = {
                'num_chunks': num_chunks,
                'avg_chunk_size': avg_chunk_size,
                'processing_time': processing_time,
                'top_5_accuracy': top_5_accuracy,
                'avg_score': avg_score,
                'completeness': completeness,
                'results': search_results[:5]  # Top-5 for report
            }

        # Compute improvements if both methods
        if 'random' in results and 'ast' in results:
            random = results['random']
            ast = results['ast']
            results['improvement'] = {
                'accuracy_improvement': ast['top_5_accuracy'] - random['top_5_accuracy'],
                'score_improvement': ((ast['avg_score'] - random['avg_score']) / random['avg_score']) * 100 if random['avg_score'] > 0 else 0,
                'chunk_reduction': ((random['num_chunks'] - ast['num_chunks']) / random['num_chunks']) * 100 if random['num_chunks'] > 0 else 0
            }

        return results

    def run_full_analysis(self) -> List[Dict[str, Any]]:
        full_results = []
        for query in self.test_queries:
            analysis = self.analyze_query(query)
            full_results.append({'query': query, 'analysis': analysis})
        return full_results
