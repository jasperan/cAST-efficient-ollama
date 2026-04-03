from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Iterable, List

from cast_ollama.chunking.ast_chunker import ASTChunker
from cast_ollama.chunking.random_chunker import RandomChunker
from cast_ollama.comparison.analyzer import Analyzer
from cast_ollama.comparison.reporter import Reporter
from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.embedding.reranker import CodeReranker
from cast_ollama.metadata.extractor import OllamaMetadataExtractor
from cast_ollama.oracle_db.operations import bulk_insert_chunks, delete_all_chunks
from cast_ollama.retrieval.search import SearchPipeline

DEFAULT_QUERY = "validate email address with regex"


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
        return [all(validator(item) for validator in self.validators) for item in items]


def helper_function():
    return True
""".strip()


def _build_chunker(method: str):
    if method == "random":
        return RandomChunker(chunk_size=100, overlap_percentage=10)
    return ASTChunker()


def vectorize_code(
    code: str,
    *,
    file_path: str,
    methods: Iterable[str],
    embedder: CodeEmbedder,
    enrich_metadata: bool = False,
) -> Dict[str, int]:
    extractor = OllamaMetadataExtractor() if enrich_metadata else None
    chunks_to_insert: List[Dict] = []
    counts: Dict[str, int] = {}

    for method in methods:
        chunker = _build_chunker(method)
        chunks = chunker.chunk(code)
        counts[method] = len(chunks)

        for index, chunk in enumerate(chunks):
            metadata = dict(chunk)
            if extractor:
                metadata.update(extractor.extract(chunk["content"]).model_dump())

            chunks_to_insert.append(
                {
                    "chunk_id": f"{method}_{uuid.uuid4().hex[:8]}_{index}",
                    "file_path": file_path,
                    "chunk_content": chunk["content"],
                    "metadata": metadata,
                    "embedding": embedder.embed(chunk["content"])[0],
                    "chunking_method": method,
                }
            )

    delete_all_chunks()
    bulk_insert_chunks(chunks_to_insert)
    return counts


def run_demo(
    *,
    sample_file: str | None = None,
    query: str = DEFAULT_QUERY,
    embedding_backend: str = "hash",
    reranker_backend: str = "lexical",
    enrich_metadata: bool = False,
    report_prefix: str = "demo_report",
    output_dir: str | None = None,
    top_k_final: int | None = None,
    use_rerank: bool | None = None,
) -> Dict[str, object]:
    code = Path(sample_file).read_text(encoding="utf-8") if sample_file else generate_synthetic_code()
    code_origin = sample_file or "synthetic"

    embedder = CodeEmbedder(provider=embedding_backend)
    reranker = CodeReranker(provider=reranker_backend)
    counts = vectorize_code(
        code,
        file_path=code_origin,
        methods=("random", "ast"),
        embedder=embedder,
        enrich_metadata=enrich_metadata,
    )

    analyzer = Analyzer(search_pipeline=SearchPipeline(embedder=embedder, reranker=reranker))
    analysis = analyzer.analyze_query(
        query,
        methods=["random", "ast"],
        top_k_final=top_k_final,
        use_rerank=True if use_rerank is None else use_rerank,
    )
    reporter = Reporter()
    console_report = reporter.generate_console_report(query, analysis)
    payload = [{"query": query, "analysis": analysis}]
    if output_dir:
        report_base = Path(output_dir) / "demo_report"
    else:
        report_base = Path(report_prefix)
    report_base.parent.mkdir(parents=True, exist_ok=True)
    reporter.export_to_csv(payload, f"{report_base}.csv")
    reporter.export_to_json(payload, f"{report_base}.json")

    return {
        "query": query,
        "code_origin": code_origin,
        "chunk_counts": counts,
        "embedder_backend": embedder.active_backend,
        "reranker_backend": reranker.active_backend,
        "analysis": analysis,
        "console_report": console_report,
    }
