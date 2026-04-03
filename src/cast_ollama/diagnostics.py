from __future__ import annotations

import importlib
from typing import Dict, Tuple

import requests

from cast_ollama.config import Config
from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.embedding.reranker import CodeReranker
from cast_ollama.oracle_db.operations import get_storage_backend


def _module_status(module_name: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "available"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def _ollama_status(endpoint: str) -> str:
    try:
        response = requests.get(f"{endpoint.rstrip('/')}/api/tags", timeout=1.5)
        if response.ok:
            return "reachable"
        return f"HTTP {response.status_code}"
    except Exception as exc:  # pragma: no cover
        return f"unreachable ({exc})"


def collect_runtime_status(
    *,
    embedding_backend: str | None = None,
    reranker_backend: str | None = None,
    endpoint: str | None = None,
    sample_file: str | None = None,
) -> Dict[str, object]:
    embedder = CodeEmbedder(provider=embedding_backend or Config.EMBEDDING_PROVIDER)
    reranker = CodeReranker(provider=reranker_backend or Config.RERANKER_PROVIDER)
    embedder.embed("def noop(): pass")
    reranker.rerank("noop", ["def noop(): pass"], 1)

    modules = {
        name: {"ok": ok, "detail": detail}
        for name, (ok, detail) in {
            "tree_sitter": _module_status("tree_sitter_python"),
            "chromadb": _module_status("chromadb"),
            "oracledb": _module_status("oracledb"),
            "ollama": _module_status("ollama"),
            "sentence_transformers": _module_status("sentence_transformers"),
            "FlagEmbedding": _module_status("FlagEmbedding"),
        }.items()
    }

    resolved_embedding = embedder.active_provider or embedder.active_backend or embedder.backend_name or "unresolved"
    resolved_reranker = reranker.active_provider or reranker.active_backend or "unresolved"
    resolved_endpoint = endpoint or Config.OLLAMA_ENDPOINT
    resolved_storage = get_storage_backend()

    return {
        "config": {
            "config_path": Config.CONFIG_PATH or "defaults",
            "storage_backend_requested": Config.STORAGE_BACKEND,
            "storage_backend_resolved": resolved_storage,
            "embedding_backend_requested": embedding_backend or Config.EMBEDDING_PROVIDER,
            "embedding_backend_resolved": resolved_embedding,
            "reranker_backend_requested": reranker_backend or Config.RERANKER_PROVIDER,
            "reranker_backend_resolved": resolved_reranker,
            "ollama_endpoint": resolved_endpoint,
            "chroma_persist_dir": Config.CHROMA_PERSIST_DIR,
            "output_dir": Config.OUTPUT_DIR,
            "local_only": Config.LOCAL_ONLY,
            "sample_file": sample_file,
        },
        "ollama_status": _ollama_status(resolved_endpoint),
        "modules": modules,
    }


def format_runtime_status(status: Dict[str, object]) -> str:
    config = status["config"]
    lines = [
        "runtime diagnostics",
        "===================",
        f"vector backend:   {config['storage_backend_resolved']} (requested: {config['storage_backend_requested']})",
        f"embedder backend: {config['embedding_backend_resolved']} (requested: {config['embedding_backend_requested']})",
        f"reranker backend: {config['reranker_backend_resolved']} (requested: {config['reranker_backend_requested']})",
        f"ollama endpoint:  {config['ollama_endpoint']} [{status['ollama_status']}]",
        f"chroma dir:       {config['chroma_persist_dir']}",
        f"output dir:       {config['output_dir']}",
        "",
        "module checks:",
    ]
    for name, payload in status["modules"].items():
        state = "OK" if payload["ok"] else "WARN"
        lines.append(f"- {name}: {state} ({payload['detail']})")
    return "\n".join(lines)
