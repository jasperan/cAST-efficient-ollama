from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from cast_ollama.config import Config

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import FlagReranker
except Exception:  # pragma: no cover - optional dependency safety
    FlagReranker = None


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


@dataclass
class RerankerBackendInfo:
    provider: str
    model_name: str


class _BaseRerankerBackend:
    info: RerankerBackendInfo

    def rerank(self, query: str, candidates: Sequence[str], top_k: int) -> List[Tuple[int, float]]:
        raise NotImplementedError


class _FlagEmbeddingBackend(_BaseRerankerBackend):
    def __init__(self, model_name: str):
        if FlagReranker is None:
            raise RuntimeError("FlagEmbedding is unavailable")
        self.reranker = FlagReranker(model_name, use_fp16=False)
        self.info = RerankerBackendInfo(provider="flagembedding", model_name=model_name)

    def rerank(self, query: str, candidates: Sequence[str], top_k: int) -> List[Tuple[int, float]]:
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.reranker.compute_score(pairs, normalize=True)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [(index, float(score)) for index, score in ranked[:top_k]]


class _LexicalRerankerBackend(_BaseRerankerBackend):
    def __init__(self):
        self.info = RerankerBackendInfo(provider="lexical", model_name="token-overlap")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _score(self, query: str, candidate: str) -> float:
        query_tokens = self._tokenize(query)
        candidate_tokens = self._tokenize(candidate)
        if not query_tokens or not candidate_tokens:
            return 0.0

        query_set = set(query_tokens)
        candidate_set = set(candidate_tokens)
        overlap = len(query_set & candidate_set) / len(query_set)

        sequence_bonus = 0.0
        candidate_lower = candidate.lower()
        for token in query_tokens:
            if token in candidate_lower:
                sequence_bonus += 1.0 / len(query_tokens)

        density = min(len(query_set & candidate_set) / max(len(candidate_set), 1), 1.0)
        score = (0.6 * overlap) + (0.3 * sequence_bonus) + (0.1 * density)
        return max(0.0, min(score, 1.0))

    def rerank(self, query: str, candidates: Sequence[str], top_k: int) -> List[Tuple[int, float]]:
        ranked = sorted(
            ((index, self._score(query, candidate)) for index, candidate in enumerate(candidates)),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(index, float(score)) for index, score in ranked[:top_k]]


class CodeReranker:
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.model_name = model_name or Config.RERANKER_MODEL
        self.provider = (backend or provider or Config.RERANKER_PROVIDER).lower()
        self.requested_backend = self.provider
        self._backend: Optional[_BaseRerankerBackend] = None
        self.backend_info: Optional[RerankerBackendInfo] = None

    def _build_backend(self) -> _BaseRerankerBackend:
        requested = self.provider
        if requested in {"lexical", "local", "fallback"}:
            return _LexicalRerankerBackend()

        if requested in {"flag", "flagembedding", "flag-embedding", "bge"}:
            return _FlagEmbeddingBackend(self.model_name)

        if requested != "auto":
            raise ValueError(f"Unsupported reranker provider: {requested}")

        try:
            return _FlagEmbeddingBackend(self.model_name)
        except Exception as exc:
            logger.warning("Falling back to lexical reranking because FlagEmbedding is unavailable: %s", exc)
            return _LexicalRerankerBackend()

    def _ensure_backend(self) -> _BaseRerankerBackend:
        if self._backend is None:
            self._backend = self._build_backend()
            self.backend_info = self._backend.info
        return self._backend

    def rerank(self, query: str, candidates: Sequence[str], top_k: int) -> List[Tuple[int, float]]:
        backend = self._ensure_backend()
        try:
            return backend.rerank(query, candidates, top_k)
        except Exception as exc:
            if backend.info.provider == "lexical":
                raise
            logger.warning("Reranker backend failed during scoring; retrying with lexical backend: %s", exc)
            self._backend = _LexicalRerankerBackend()
            self.backend_info = self._backend.info
            return self._backend.rerank(query, candidates, top_k)

    @property
    def active_provider(self) -> Optional[str]:
        backend = self._ensure_backend()
        return backend.info.provider

    @property
    def active_backend(self) -> Optional[str]:
        return self.active_provider
