from __future__ import annotations

import logging
from typing import Dict, List, Optional

from cast_ollama.config import Config
from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.embedding.reranker import CodeReranker
from cast_ollama.oracle_db.operations import search_by_vector

logger = logging.getLogger(__name__)


class SearchPipeline:
    def __init__(
        self,
        *,
        embedding_backend: str | None = None,
        reranker_backend: str | None = None,
        embedder: CodeEmbedder | None = None,
        reranker: CodeReranker | None = None,
    ):
        self.embedder = embedder or CodeEmbedder(backend=embedding_backend)
        self.reranker = reranker or CodeReranker(backend=reranker_backend)

    def _expand_context(self, result: Dict, all_results: List[Dict]) -> Dict:
        parent_name = result.get("parent_class")
        if not parent_name:
            return result

        for candidate in all_results:
            if candidate.get("chunk_type") == "class_definition" and (
                candidate.get("function_name") == parent_name
                or candidate.get("parent_class") == parent_name
                or candidate.get("symbol_name") == parent_name
                or candidate.get("class_name") == parent_name
            ):
                result["parent_content"] = candidate["chunk_content"]
                result["parent_chunk_id"] = candidate.get("chunk_id")
                break
        return result

    def search(
        self,
        query: str,
        top_k_initial: int = Config.TOP_K_RETRIEVAL,
        top_k_final: int = Config.TOP_K_RERANK,
        chunking_method: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[Dict]:
        query_embedding = self.embedder.embed(query)[0]
        initial_results = search_by_vector(query_embedding, top_k_initial, chunking_method)
        if not initial_results:
            logger.warning("No results from initial retrieval.")
            return []

        unique_results = list({result["chunk_id"]: dict(result) for result in initial_results}.values())
        candidates = [result["chunk_content"] for result in unique_results]

        if use_rerank:
            rerank_results = self.reranker.rerank(query, candidates, min(top_k_final, len(candidates)))
            final_results = []
            for index, score in rerank_results:
                result = dict(unique_results[index])
                result["score"] = score
                result["embedding_provider"] = getattr(self.embedder, "active_provider", None)
                result["reranker_provider"] = getattr(self.reranker, "active_provider", None)
                final_results.append(result)
        else:
            final_results = [dict(result) for result in unique_results[:top_k_final]]
            for result in final_results:
                result["score"] = max(0.0, 1 - float(result.get("distance", 1.0)))
                result["embedding_provider"] = getattr(self.embedder, "active_provider", None)
                result["reranker_provider"] = None

        return [self._expand_context(result, initial_results) for result in final_results]
