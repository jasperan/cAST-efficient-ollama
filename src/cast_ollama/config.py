from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class AppConfig:
    DEFAULTS = {
        "oracle": {"user": "scott", "password": "password", "dsn": "localhost:1521/freepdb1"},
        "ollama": {
            "endpoint": "http://localhost:11434",
            "model": "llama2",
            "embed_model": "nomic-embed-text",
        },
        "embedding": {"model": "microsoft/codebert-base", "backend": "hash", "dimension": 768},
        "reranker": {"model": "BAAI/bge-reranker-v2-m3", "backend": "lexical"},
        "chroma": {"persist_dir": ".cast_ollama/chroma", "collection": "code_chunks"},
        "reporting": {"report_dir": "."},
        "app": {"verbose": False, "chunk_size": 2000, "overlap_percentage": 10, "top_k_retrieval": 150, "top_k_rerank": 20},
    }

    def __init__(self) -> None:
        self._load_config()

    def _load_config(self) -> None:
        config_data = self._load_yaml_config()
        self.CONFIG_PATH = "config.yaml" if Path("config.yaml").exists() else None
        self.CONFIG_PATH = str(Path("config.yaml")) if Path("config.yaml").exists() else None

        self.ORACLE_USER = self._resolve(config_data, "oracle", "user", env="ORACLE_USER")
        self.ORACLE_PASSWORD = self._resolve(config_data, "oracle", "password", env="ORACLE_PASSWORD")
        self.ORACLE_DSN = self._resolve(config_data, "oracle", "dsn", env="ORACLE_DSN")

        self.OLLAMA_ENDPOINT = self._resolve(config_data, "ollama", "endpoint", env="OLLAMA_ENDPOINT")
        self.OLLAMA_MODEL = self._resolve(config_data, "ollama", "model", env="OLLAMA_MODEL")
        self.OLLAMA_EMBED_MODEL = self._resolve(config_data, "ollama", "embed_model", env="OLLAMA_EMBED_MODEL")

        self.EMBEDDING_MODEL = self._resolve(config_data, "embedding", "model", env="EMBEDDING_MODEL")
        self.SENTENCE_TRANSFORMER_MODEL = self.EMBEDDING_MODEL
        self.EMBEDDING_BACKEND = self._resolve(config_data, "embedding", "backend", env="EMBEDDING_BACKEND")
        self.EMBEDDING_DIM = int(self._resolve(config_data, "embedding", "dimension", env="EMBEDDING_DIM"))
        self.EMBEDDING_DIMENSION = self.EMBEDDING_DIM

        self.RERANKER_MODEL = self._resolve(config_data, "reranker", "model", env="RERANKER_MODEL")
        self.RERANKER_BACKEND = self._resolve(config_data, "reranker", "backend", env="RERANKER_BACKEND")

        self.CHROMA_PERSIST_DIR = self._resolve(config_data, "chroma", "persist_dir", env="CHROMA_PERSIST_DIR")
        self.CHROMA_PATH = os.getenv("CHROMA_PATH", self.CHROMA_PERSIST_DIR)
        self.CHROMA_PERSIST_DIR = self.CHROMA_PATH
        self.CHROMA_COLLECTION = self._resolve(config_data, "chroma", "collection", env="CHROMA_COLLECTION")
        self.REPORT_DIR = self._resolve(config_data, "reporting", "report_dir", env="REPORT_DIR")
        self.EXPORT_DIR = os.getenv("EXPORT_DIR", self.REPORT_DIR)
        self.OUTPUT_DIR = self.EXPORT_DIR
        self.OUTPUT_DIR = self.REPORT_DIR
        self.OUTPUT_DIR = self.EXPORT_DIR
        self.OUTPUT_DIR = self.EXPORT_DIR

        self.VERBOSE = self._as_bool(self._resolve(config_data, "app", "verbose", env="VERBOSE"))
        self.CHUNK_SIZE = int(self._resolve(config_data, "app", "chunk_size", env="CHUNK_SIZE"))
        self.OVERLAP_PERCENTAGE = int(self._resolve(config_data, "app", "overlap_percentage", env="OVERLAP_PERCENTAGE"))
        self.TOP_K_RETRIEVAL = int(self._resolve(config_data, "app", "top_k_retrieval", env="TOP_K_RETRIEVAL"))
        self.TOP_K_RERANK = int(self._resolve(config_data, "app", "top_k_rerank", env="TOP_K_RERANK"))
        self.STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", os.getenv("BACKEND", "auto"))
        self.LOCAL_ONLY = self._as_bool(os.getenv("LOCAL_ONLY", False))
        self.EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", self.EMBEDDING_BACKEND)
        self.RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", self.RERANKER_BACKEND)

    def _load_yaml_config(self) -> dict:
        config_file = Path("config.yaml")
        if not config_file.exists():
            return {}
        with config_file.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _resolve(self, config_data: dict, section: str, key: str, *, env: str) -> Any:
        if env in os.environ:
            return os.environ[env]
        return config_data.get(section, {}).get(key, self.DEFAULTS[section][key])

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def reload(self) -> None:
        self._load_config()

    def apply_runtime_overrides(self, **overrides: Any) -> None:
        for key, value in overrides.items():
            if value is None:
                continue
            setattr(self, key, value)

        if "EMBEDDING_PROVIDER" in overrides and overrides["EMBEDDING_PROVIDER"] is not None:
            self.EMBEDDING_BACKEND = overrides["EMBEDDING_PROVIDER"]
            self.EMBEDDING_PROVIDER = overrides["EMBEDDING_PROVIDER"]
        if "EMBEDDING_BACKEND" in overrides and overrides["EMBEDDING_BACKEND"] is not None:
            self.EMBEDDING_BACKEND = overrides["EMBEDDING_BACKEND"]
            self.EMBEDDING_PROVIDER = overrides["EMBEDDING_BACKEND"]
        if "RERANKER_PROVIDER" in overrides and overrides["RERANKER_PROVIDER"] is not None:
            self.RERANKER_BACKEND = overrides["RERANKER_PROVIDER"]
            self.RERANKER_PROVIDER = overrides["RERANKER_PROVIDER"]
        if "RERANKER_BACKEND" in overrides and overrides["RERANKER_BACKEND"] is not None:
            self.RERANKER_BACKEND = overrides["RERANKER_BACKEND"]
            self.RERANKER_PROVIDER = overrides["RERANKER_BACKEND"]
        if "EXPORT_DIR" in overrides and overrides["EXPORT_DIR"] is not None:
            self.REPORT_DIR = overrides["EXPORT_DIR"]
            self.EXPORT_DIR = overrides["EXPORT_DIR"]
            self.OUTPUT_DIR = overrides["EXPORT_DIR"]
            self.EXPORT_DIR = overrides["EXPORT_DIR"]
            self.OUTPUT_DIR = overrides["EXPORT_DIR"]
        if "OUTPUT_DIR" in overrides and overrides["OUTPUT_DIR"] is not None:
            self.REPORT_DIR = overrides["OUTPUT_DIR"]
            self.EXPORT_DIR = overrides["OUTPUT_DIR"]
            self.OUTPUT_DIR = overrides["OUTPUT_DIR"]
        if "CHROMA_PATH" in overrides and overrides["CHROMA_PATH"] is not None:
            self.CHROMA_PATH = overrides["CHROMA_PATH"]
            self.CHROMA_PERSIST_DIR = overrides["CHROMA_PATH"]
        if "CHROMA_PERSIST_DIR" in overrides and overrides["CHROMA_PERSIST_DIR"] is not None:
            self.CHROMA_PERSIST_DIR = overrides["CHROMA_PERSIST_DIR"]
            self.CHROMA_PATH = overrides["CHROMA_PERSIST_DIR"]
        if self.LOCAL_ONLY and str(self.STORAGE_BACKEND).lower() == "auto":
            self.STORAGE_BACKEND = "chroma"


Config = AppConfig()
