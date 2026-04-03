from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

from cast_ollama.comparison.analyzer import Analyzer
from cast_ollama.comparison.reporter import Reporter
from cast_ollama.config import Config
from cast_ollama.demo import DEFAULT_QUERY, run_demo, vectorize_code
from cast_ollama.diagnostics import collect_runtime_status, format_runtime_status
from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.embedding.reranker import CodeReranker
from cast_ollama.oracle_db.schema import create_tables
from cast_ollama.retrieval.search import SearchPipeline

logger = logging.getLogger(__name__)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Code RAG AST demo CLI")
    parser.add_argument("--action", choices=["setup", "vectorize", "search", "doctor", "demo", "walkthrough"], required=True)
    parser.add_argument("--profile", choices=["auto", "local"], default="auto")
    parser.add_argument("--chunking-method", choices=["random", "ast", "both"], default="both")
    parser.add_argument("--sample-file")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of final results to report")
    parser.add_argument("--report-prefix", default="search_report", help="Report filename stem")
    parser.add_argument("--report-dir", default=Config.REPORT_DIR, help="Directory for report output")
    parser.add_argument("--output", choices=["text", "json"], default="text", help="Doctor output format")
    parser.add_argument("--db-user")
    parser.add_argument("--db-password")
    parser.add_argument("--db-dsn")
    parser.add_argument("--storage-backend", choices=["auto", "chroma", "oracle"], help="Force a storage backend instead of relying on profile/env defaults")
    parser.add_argument("--ollama-endpoint", default=Config.OLLAMA_ENDPOINT)
    parser.add_argument("--ollama-model", default=Config.OLLAMA_MODEL)
    parser.add_argument("--enrich-metadata", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--rerank", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--append", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--embedding-backend", dest="embedding_backend", choices=["auto", "sentence-transformers", "transformers", "hash", "local", "local-hash", "ollama"], default=Config.EMBEDDING_BACKEND)
    parser.add_argument("--embedding-provider", dest="embedding_backend", choices=["auto", "sentence-transformers", "transformers", "hash", "local", "local-hash", "ollama"], help=argparse.SUPPRESS)
    parser.add_argument("--reranker-backend", dest="reranker_backend", choices=["auto", "flag", "flagembedding", "flag-embedding", "lexical", "local"], default=Config.RERANKER_BACKEND)
    parser.add_argument("--reranker-provider", dest="reranker_backend", choices=["auto", "flag", "flagembedding", "flag-embedding", "lexical", "local"], help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    return parser


def resolve_sample_file(sample_file: str | None) -> Path:
    return Path(sample_file) if sample_file else Path("examples/sample.py")


def apply_overrides(args) -> tuple[str, str, bool]:
    Config.reload()
    if args.db_user:
        Config.ORACLE_USER = args.db_user
    if args.db_password:
        Config.ORACLE_PASSWORD = args.db_password
    if args.db_dsn:
        Config.ORACLE_DSN = args.db_dsn

    Config.OLLAMA_ENDPOINT = args.ollama_endpoint
    Config.OLLAMA_MODEL = args.ollama_model
    Config.VERBOSE = args.verbose
    Config.REPORT_DIR = args.report_dir

    embedding_backend = args.embedding_backend
    reranker_backend = args.reranker_backend
    enrich_metadata = args.enrich_metadata
    storage_backend = args.storage_backend or Config.STORAGE_BACKEND

    if args.profile == "local":
        Config.LOCAL_ONLY = True
        storage_backend = "chroma"
        if embedding_backend == "auto":
            embedding_backend = "hash"
        if reranker_backend == "auto":
            reranker_backend = "lexical"
        if not args.enrich_metadata:
            enrich_metadata = False
    else:
        Config.LOCAL_ONLY = str(storage_backend).lower() == "chroma"

    Config.STORAGE_BACKEND = storage_backend
    Config.EMBEDDING_BACKEND = embedding_backend
    Config.EMBEDDING_PROVIDER = embedding_backend
    Config.RERANKER_BACKEND = reranker_backend
    Config.RERANKER_PROVIDER = reranker_backend
    return embedding_backend, reranker_backend, enrich_metadata


def ensure_report_prefix(args, *, default_stem: str) -> str:
    stem = args.report_prefix
    if stem == "search_report":
        stem = default_stem
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    return str(report_dir / stem)


def run_vectorize(args, *, embedding_backend: str, enrich_metadata: bool) -> int:
    if not args.sample_file:
        raise SystemExit("--sample-file is required for vectorize action")

    code = Path(args.sample_file).read_text(encoding="utf-8")
    methods = [args.chunking_method] if args.chunking_method != "both" else ["random", "ast"]
    counts = vectorize_code(
        code,
        file_path=args.sample_file,
        methods=methods,
        embedder=CodeEmbedder(backend=embedding_backend),
        enrich_metadata=enrich_metadata,
    )
    print(f"Vectorized and stored {sum(counts.values())} chunks from {args.sample_file}.")
    return 0


def run_search(args, *, embedding_backend: str, reranker_backend: str) -> int:
    if not args.query:
        raise SystemExit("--query is required for search action")

    analyzer = Analyzer(
        search_pipeline=SearchPipeline(
            embedder=CodeEmbedder(backend=embedding_backend),
            reranker=CodeReranker(backend=reranker_backend),
        )
    )
    reporter = Reporter()
    methods = [args.chunking_method] if args.chunking_method != "both" else ["random", "ast"]
    analysis = analyzer.analyze_query(
        args.query,
        methods=methods,
        top_k_final=args.top_k,
        use_rerank=args.rerank,
    )
    print(reporter.generate_console_report(args.query, analysis))
    report_prefix = ensure_report_prefix(args, default_stem="search_report")
    payload = [{"query": args.query, "analysis": analysis}]
    reporter.export_to_csv(payload, f"{report_prefix}.csv")
    reporter.export_to_json(payload, f"{report_prefix}.json")
    return 0


def run_doctor(args, *, embedding_backend: str, reranker_backend: str) -> int:
    status = collect_runtime_status(
        embedding_backend=embedding_backend,
        reranker_backend=reranker_backend,
        endpoint=args.ollama_endpoint,
        sample_file=str(resolve_sample_file(args.sample_file)),
    )

    if args.output == "json":
        payload = {
            "config": dict(status["config"]),
            "modules": status["modules"],
            "ollama_status": status["ollama_status"],
        }
        payload["config"]["report_dir"] = args.report_dir
        print(json.dumps(payload, indent=2))
    else:
        config = status["config"]
        print(f"Vector backend: {config['storage_backend_resolved']}")
        print(f"Embedding backend: {config['embedding_backend_resolved']}")
        print(f"Reranker backend: {config['reranker_backend_resolved']}")
        print(format_runtime_status(status))
    return 0


def run_demo_action(args, *, embedding_backend: str, reranker_backend: str, enrich_metadata: bool) -> int:
    result = run_demo(
        sample_file=args.sample_file,
        query=args.query or DEFAULT_QUERY,
        embedding_backend=embedding_backend,
        reranker_backend=reranker_backend,
        enrich_metadata=enrich_metadata,
        report_prefix=ensure_report_prefix(args, default_stem="demo_report"),
        top_k_final=args.top_k,
        use_rerank=args.rerank,
    )
    print("Demo complete.")
    print(f"- embedder backend: {result['embedder_backend']}")
    print(f"- reranker backend: {result['reranker_backend']}")
    print(result["console_report"])
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    embedding_backend, reranker_backend, enrich_metadata = apply_overrides(args)

    if args.action == "setup":
        print("Setting up Oracle tables...")
        create_tables()
        print("Setup complete.")
        return 0

    if args.action == "doctor":
        return run_doctor(args, embedding_backend=embedding_backend, reranker_backend=reranker_backend)

    if args.action == "vectorize":
        return run_vectorize(args, embedding_backend=embedding_backend, enrich_metadata=enrich_metadata)

    if args.action in {"demo", "walkthrough"}:
        return run_demo_action(
            args,
            embedding_backend=embedding_backend,
            reranker_backend=reranker_backend,
            enrich_metadata=enrich_metadata,
        )

    if args.action == "search":
        return run_search(args, embedding_backend=embedding_backend, reranker_backend=reranker_backend)

    parser.error(f"Unsupported action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
