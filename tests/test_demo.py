import tempfile
import unittest
from pathlib import Path

from cast_ollama.chroma_db.wrapper import ChromaDBWrapper
from cast_ollama.config import Config
from cast_ollama.demo import run_demo


class TestDemo(unittest.TestCase):
    def test_demo_generates_reports(self):
        previous_backend = Config.EMBEDDING_BACKEND
        previous_reranker = Config.RERANKER_BACKEND
        previous_chroma_dir = Config.CHROMA_PERSIST_DIR

        with tempfile.TemporaryDirectory() as tmp_dir:
            Config.EMBEDDING_BACKEND = "hash"
            Config.RERANKER_BACKEND = "lexical"
            Config.CHROMA_PERSIST_DIR = str(Path(tmp_dir) / "chroma")
            ChromaDBWrapper._instance = None
            ChromaDBWrapper._client = None
            ChromaDBWrapper._collection = None

            result = run_demo(sample_file="examples/sample.py", output_dir=tmp_dir, use_rerank=True)
            self.assertIn("random", result["analysis"])
            self.assertIn("ast", result["analysis"])
            self.assertTrue((Path(tmp_dir) / "demo_report.csv").exists())
            self.assertTrue((Path(tmp_dir) / "demo_report.json").exists())

        Config.EMBEDDING_BACKEND = previous_backend
        Config.RERANKER_BACKEND = previous_reranker
        Config.CHROMA_PERSIST_DIR = previous_chroma_dir
        ChromaDBWrapper._instance = None
        ChromaDBWrapper._client = None
        ChromaDBWrapper._collection = None


if __name__ == "__main__":
    unittest.main()
