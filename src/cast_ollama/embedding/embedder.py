from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Sequence, Union

import numpy as np

from cast_ollama.config import Config

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|\S")


class BackendUnavailableError(RuntimeError):
    """Raised when a runtime backend cannot be used."""


class HashingEmbeddingBackend:
    name = "hash"

    def __init__(self, dimension: int = Config.EMBEDDING_DIMENSION) -> None:
        self.dimension = int(dimension)

    def _tokenize(self, text: str) -> List[str]:
        tokens = TOKEN_RE.findall(text.lower())
        return tokens or [text.lower() or ""]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = -1.0 if digest[4] % 2 else 1.0
                embeddings[row, index] += sign
            norm = float(np.linalg.norm(embeddings[row]))
            if norm > 0:
                embeddings[row] /= norm
        return embeddings


class OllamaEmbeddingBackend:
    name = "ollama"

    def __init__(self, model_name: str = Config.OLLAMA_EMBED_MODEL) -> None:
        try:
            from ollama import embed as ollama_embed
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailableError("ollama package is not installed") from exc
        self._embed_fn = ollama_embed
        self.model_name = model_name
        self.dimension = Config.EMBEDDING_DIMENSION

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        try:
            response = self._embed_fn(model=self.model_name, input=list(texts))
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailableError(
                f"Ollama embedding backend unavailable for model '{self.model_name}'"
            ) from exc
        vectors = np.asarray(response["embeddings"], dtype=np.float32)
        if vectors.ndim != 2:
            raise BackendUnavailableError("Ollama embedding response was not two-dimensional")
        self.dimension = int(vectors.shape[1])
        return vectors


class SentenceTransformerEmbeddingBackend:
    name = "sentence-transformers"

    def __init__(self, model_name: str = Config.SENTENCE_TRANSFORMER_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise BackendUnavailableError("sentence-transformers is not installed") from exc
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover
            raise BackendUnavailableError(
                f"SentenceTransformer backend unavailable for model '{model_name}'"
            ) from exc
        dimension = self.model.get_sentence_embedding_dimension()
        self.dimension = int(dimension or Config.EMBEDDING_DIMENSION)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


# Test-friendly aliases
_HashingEmbeddingBackend = HashingEmbeddingBackend
_SentenceTransformerBackend = SentenceTransformerEmbeddingBackend
_OllamaEmbeddingBackend = OllamaEmbeddingBackend


class CodeEmbedder:
    def __init__(
        self,
        backend: str = Config.EMBEDDING_BACKEND,
        dimension: int = Config.EMBEDDING_DIMENSION,
        vector_size: int | None = None,
        provider: str | None = None,
    ) -> None:
        self.requested_backend = provider or backend
        self.model_name = Config.SENTENCE_TRANSFORMER_MODEL
        self.dimension = int(vector_size or dimension)
        self.backend_name: str | None = None
        self.active_backend: str | None = None
        self.active_provider: str | None = None
        self._backend_order = self._resolve_backend_order(self.requested_backend)
        self._backend = None

    def _normalize_backend(self, backend: str) -> str:
        alias_map = {
            "auto": "auto",
            "local": "hash",
            "local-hash": "hash",
            "hash": "hash",
            "ollama": "ollama",
            "sentence-transformer": "sentence-transformers",
            "sentence_transformer": "sentence-transformers",
            "transformer": "sentence-transformers",
            "transformers": "sentence-transformers",
            "sentence-transformers": "sentence-transformers",
        }
        normalized = alias_map.get(backend.lower(), backend.lower())
        return normalized

    def _resolve_backend_order(self, backend: str) -> List[str]:
        normalized = self._normalize_backend(backend)
        if normalized == "auto":
            return ["hash", "sentence-transformers", "ollama"]
        return [normalized]

    def _instantiate_backend(self, backend_name: str):
        if backend_name == "ollama":
            return _OllamaEmbeddingBackend()
        if backend_name == "sentence-transformers":
            return _SentenceTransformerBackend()
        if backend_name == "hash":
            return _HashingEmbeddingBackend(dimension=self.dimension)
        raise BackendUnavailableError(f"Unknown embedding backend '{backend_name}'")

    def _ensure_backend(self, backend_name: str):
        if self._backend is None or self.backend_name != backend_name:
            self._backend = self._instantiate_backend(backend_name)
            self.backend_name = backend_name
        return self._backend

    def embed(self, text: Union[str, Sequence[str]]) -> np.ndarray:
        texts = [text] if isinstance(text, str) else list(text)
        last_error: Exception | None = None
        requested_is_auto = self._normalize_backend(self.requested_backend) == "auto"

        for backend_name in self._backend_order:
            try:
                backend = self._ensure_backend(backend_name)
                vectors = backend.embed(texts)
                self.dimension = int(vectors.shape[1])
                self.backend_name = backend_name
                self.active_backend = backend_name
                self.active_provider = "local-hash" if backend_name == "hash" else backend_name
                return vectors
            except Exception as exc:
                last_error = exc if isinstance(exc, BackendUnavailableError) else BackendUnavailableError(str(exc))
                logger.warning("Embedding backend '%s' unavailable: %s", backend_name, last_error)
                self._backend = None
                self.backend_name = None
                self.active_backend = None
                self.active_provider = None
                if not requested_is_auto:
                    break

        raise RuntimeError("Unable to embed text with any configured backend") from last_error
