from typing import List, Dict, Optional
from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.embedding.reranker import CodeReranker
from cast_ollama.oracle_db.operations import search_by_vector, get_chunk_by_id
from cast_ollama.config import Config
import logging

logger = logging.getLogger(__name__)

class SearchPipeline:
    def __init__(self):
        self.embedder = CodeEmbedder()
        self.reranker = CodeReranker()

    def _expand_context(self, result: Dict, all_results: List[Dict]) -> Dict:
        # Simple context expansion: if has parent_class, try to find the parent class chunk
        if 'parent_class' in result and result['parent_class']:
            parent_name = result['parent_class']
            # Search for class definition with that name
            # For demo, we can assume chunk_id or search
            # But to implement, perhaps add a small search or check in all_results
            for res in all_results:
                if res['chunk_type'] == 'class_definition' and res.get('function_name') == parent_name:
                    result['parent_content'] = res['chunk_content']
                    break
        return result

    def search(self, query: str, top_k_initial: int = Config.TOP_K_RETRIEVAL, top_k_final: int = Config.TOP_K_RERANK, chunking_method: Optional[str] = None, use_rerank: bool = True) -> List[Dict]:
        # Embed query
        query_embedding = self.embedder.embed(query)[0]

        # Initial retrieval
        initial_results = search_by_vector(query_embedding, top_k_initial, chunking_method)

        if not initial_results:
            logger.warning("No results from initial retrieval.")
            return []

        # Remove duplicates based on chunk_id
        unique_results_dict = {res['chunk_id']: res for res in initial_results}
        unique_results = list(unique_results_dict.values())

        candidates = [res['chunk_content'] for res in unique_results]

        if use_rerank:
            # Rerank
            rerank_results = self.reranker.rerank(query, candidates, top_k_final)

            # Map back to results with scores
            final_results = []
            for idx, score in rerank_results:
                res = unique_results[idx]
                res['score'] = score
                final_results.append(res)
        else:
            final_results = unique_results[:top_k_final]
            for res in final_results:
                res['score'] = 1 - res['distance']  # Approximate score from distance

        # Post-processing: remove duplicates (already done), expand context
        expanded_results = []
        for res in final_results:
            expanded = self._expand_context(res, initial_results)  # Use initial for broader context
            expanded_results.append(expanded)

        # Format with metadata (already included)
        return expanded_results
