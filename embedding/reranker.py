from typing import List, Tuple
from FlagEmbedding import FlagReranker

class CodeReranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        self.reranker = FlagReranker(model_name, use_fp16=True)

    def rerank(self, query: str, candidates: List[str], top_k: int) -> List[Tuple[int, float]]:
        pairs = [[query, cand] for cand in candidates]
        scores = self.reranker.compute_score(pairs, normalize=True)  # Scores in [0,1]
        
        # Get top_k indices and scores
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(idx, scores[idx]) for idx in sorted_indices]
        
        return results
