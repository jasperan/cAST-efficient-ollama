import unittest
from unittest.mock import patch

import numpy as np

from cast_ollama.embedding.embedder import CodeEmbedder
from cast_ollama.embedding.reranker import CodeReranker


class TestEmbedding(unittest.TestCase):
    def test_local_hash_embedder_is_deterministic(self):
        embedder = CodeEmbedder(provider='local-hash', dimension=64)
        embedding_a = embedder.embed("def hello(): return 'world'")
        embedding_b = embedder.embed("def hello(): return 'world'")
        embedding_c = embedder.embed("def goodbye(): return 'moon'")

        self.assertEqual(embedding_a.shape, (1, 64))
        self.assertTrue(np.allclose(embedding_a, embedding_b))
        self.assertFalse(np.allclose(embedding_a, embedding_c))
        self.assertAlmostEqual(float(np.linalg.norm(embedding_a[0])), 1.0, places=5)

    def test_auto_embedder_falls_back_to_local_hash(self):
        with patch('cast_ollama.embedding.embedder._SentenceTransformerBackend', side_effect=RuntimeError('boom')):
            embedder = CodeEmbedder(provider='auto', dimension=32)
            embedding = embedder.embed('def fallback(): pass')

        self.assertEqual(embedding.shape, (1, 32))
        self.assertEqual(embedder.active_provider, 'local-hash')

    def test_lexical_reranker_prefers_relevant_candidate(self):
        reranker = CodeReranker(provider='lexical')
        query = 'validate email address'
        candidates = [
            'def validate_email(email): return email.endswith("@example.com")',
            'print("hello world")',
        ]
        results = reranker.rerank(query, candidates, top_k=2)
        self.assertEqual(results[0][0], 0)
        self.assertGreater(results[0][1], results[1][1])

    def test_auto_reranker_falls_back_to_lexical(self):
        with patch('cast_ollama.embedding.reranker._FlagEmbeddingBackend', side_effect=RuntimeError('boom')):
            reranker = CodeReranker(provider='auto')
            results = reranker.rerank('validate email', ['def validate_email(email): pass'], top_k=1)

        self.assertEqual(results[0][0], 0)
        self.assertEqual(reranker.active_provider, 'lexical')


if __name__ == '__main__':
    unittest.main()
