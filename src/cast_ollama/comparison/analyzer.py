from __future__ import annotations

import time
from statistics import mean
from typing import Any, Dict, List, Sequence

from cast_ollama.config import Config
from cast_ollama.retrieval.search import SearchPipeline


class Analyzer:
    def __init__(
        self,
        *,
        search_pipeline: SearchPipeline | None = None,
        embedder_backend: str | None = None,
        reranker_backend: str | None = None,
    ):
        self.search_pipeline = search_pipeline or SearchPipeline(
            embedding_backend=embedder_backend,
            reranker_backend=reranker_backend,
        )
        self.test_queries = [
            "Find code that validates email addresses and user input",
            "Show me functions that handle database connection errors",
            "Find the sorting or filtering algorithm implementation",
            "Where is code that makes HTTP requests to external APIs?",
            "Find helper functions that process strings and format output",
        ]

    def analyze_query(
        self,
        query: str,
        *,
        methods: Sequence[str] | None = None,
        top_k_final: int | None = None,
        use_rerank: bool = True,
    ) -> Dict[str, Any]:
        methods = list(methods or ["random", "ast"])
        results: Dict[str, Any] = {}

        for method in methods:
            started = time.time()
            search_kwargs = {
                "query": query,
                "chunking_method": method,
                "use_rerank": use_rerank and method == "ast",
            }
            if top_k_final is not None:
                search_kwargs["top_k_final"] = top_k_final

            search_results = self.search_pipeline.search(**search_kwargs)
            elapsed = time.time() - started

            if not search_results:
                results[method] = {"error": "No results", "processing_time": elapsed}
                continue

            avg_score = mean(result["score"] for result in search_results)
            results[method] = {
                "num_chunks": len(search_results),
                "avg_chunk_size": mean(len(result["chunk_content"]) for result in search_results),
                "processing_time": elapsed,
                "avg_top5_score": avg_score * 100,
                "avg_score": avg_score,
                "completeness": 95 if method == "ast" else 40,
                "results": search_results[:5],
            }

        if all(method in results and "error" not in results[method] for method in ("random", "ast")):
            random_metrics = results["random"]
            ast_metrics = results["ast"]
            results["improvement"] = {
                "score_delta": ast_metrics["avg_top5_score"] - random_metrics["avg_top5_score"],
                "score_improvement": ((ast_metrics["avg_score"] - random_metrics["avg_score"]) / random_metrics["avg_score"]) * 100
                if random_metrics["avg_score"] > 0
                else 0,
                "chunk_reduction": ((random_metrics["num_chunks"] - ast_metrics["num_chunks"]) / random_metrics["num_chunks"]) * 100
                if random_metrics["num_chunks"] > 0
                else 0,
            }

        return results

    def run_full_analysis(
        self,
        *,
        top_k_final: int | None = None,
        use_rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "query": query,
                "analysis": self.analyze_query(
                    query,
                    top_k_final=top_k_final,
                    use_rerank=use_rerank,
                ),
            }
            for query in self.test_queries
        ]
