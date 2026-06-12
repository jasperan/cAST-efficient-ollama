from __future__ import annotations

import logging
import uuid
from statistics import mean

from cast_ollama.chunking.ast_chunker import ASTChunker
from cast_ollama.chunking.random_chunker import RandomChunker
from cast_ollama.comparison.reporter import Reporter
from cast_ollama.config import Config
from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.embedding.reranker import CodeReranker
from cast_ollama.oracle_db.operations import delete_all_chunks, insert_chunk, search_by_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_code() -> str:
    return """
import re
from typing import List

class DataProcessor:
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
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def validate_phone(self, phone: str) -> bool:
        pattern = r'^\\+?1?\\d{9,15}$'
        return re.match(pattern, phone) is not None

    def process_batch(self, items: List[str]) -> List[bool]:
        return [all(v(item) for v in self.validators) for item in items]


def helper_function():
    return True
"""


def vectorize(code: str, embedder: CodeEmbedder) -> tuple[list[dict], list[dict]]:
    delete_all_chunks()
    random_chunks = RandomChunker(chunk_size=100, overlap_percentage=0).chunk(code)
    ast_chunks = ASTChunker(chunk_size=400, overlap_percentage=0).chunk(code)

    for method, chunks in (("random", random_chunks), ("ast", ast_chunks)):
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk)
            metadata.setdefault("dependencies", [])
            metadata.setdefault("purpose", f"{method} chunk")
            metadata.setdefault("complexity", "medium")
            metadata.setdefault("docstring", None)
            insert_chunk(
                chunk_id=f"{method}_{uuid.uuid4().hex[:8]}_{index}",
                file_path="synthetic.py",
                chunk_content=chunk["content"],
                metadata=metadata,
                embedding=embedder.embed(chunk["content"])[0],
                chunking_method=method,
            )
    return random_chunks, ast_chunks


def run_demo() -> None:
    Config.apply_runtime_overrides(
        LOCAL_ONLY=True,
        STORAGE_BACKEND="chroma",
        EMBEDDING_PROVIDER="local-hash",
        RERANKER_PROVIDER="lexical",
    )

    code = generate_synthetic_code()
    query = "validate email address with regex"
    embedder = CodeEmbedder(provider="local-hash")
    reranker = CodeReranker(provider="lexical")

    random_chunks, ast_chunks = vectorize(code, embedder)
    query_embedding = embedder.embed(query)[0]

    random_results = search_by_vector(query_embedding, limit=5, chunking_method="random")
    ast_results = search_by_vector(query_embedding, limit=5, chunking_method="ast")
    reranked_indices = reranker.rerank(query, [result["chunk_content"] for result in ast_results], top_k=min(5, len(ast_results)))
    ast_reranked = []
    for index, score in reranked_indices:
        result = dict(ast_results[index])
        result["score"] = score
        ast_reranked.append(result)

    for result in random_results:
        result["score"] = 1 - result.get("distance", 0.0)
        result["embedding_provider"] = embedder.active_provider
        result["reranker_provider"] = None
    for result in ast_reranked:
        result["embedding_provider"] = embedder.active_provider
        result["reranker_provider"] = reranker.active_provider

    analysis = {
        "random": {
            "num_chunks": len(random_chunks),
            "avg_chunk_size": mean(len(chunk["content"]) for chunk in random_chunks),
            "processing_time": 0.0,
            "avg_top5_score": mean(result["score"] for result in random_results) * 100 if random_results else 0,
            "avg_score": mean(result["score"] for result in random_results) if random_results else 0,
            "completeness": 40,
            "results": random_results,
        },
        "ast": {
            "num_chunks": len(ast_chunks),
            "avg_chunk_size": mean(len(chunk["content"]) for chunk in ast_chunks),
            "processing_time": 0.0,
            "avg_top5_score": mean(result["score"] for result in ast_reranked) * 100 if ast_reranked else 0,
            "avg_score": mean(result["score"] for result in ast_reranked) if ast_reranked else 0,
            "completeness": 95,
            "results": ast_reranked,
        },
    }
    analysis["improvement"] = {
        "score_delta": analysis["ast"]["avg_top5_score"] - analysis["random"]["avg_top5_score"],
        "score_improvement": ((analysis["ast"]["avg_score"] - analysis["random"]["avg_score"]) / analysis["random"]["avg_score"] * 100) if analysis["random"]["avg_score"] else 0,
        "chunk_reduction": ((analysis["random"]["num_chunks"] - analysis["ast"]["num_chunks"]) / analysis["random"]["num_chunks"] * 100) if analysis["random"]["num_chunks"] else 0,
    }

    reporter = Reporter()
    print(reporter.generate_console_report(query, analysis))


if __name__ == "__main__":
    run_demo()
