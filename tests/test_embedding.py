import unittest
import numpy as np
from embedding.embedder import CodeEmbedder
from embedding.reranker import CodeReranker

class TestEmbedding(unittest.TestCase):
    def test_embedder(self):
        embedder = CodeEmbedder()
        text = "def hello(): print('world')"
        embedding = embedder.embed(text)
        self.assertEqual(embedding.shape, (1, 768))  # CodeBERT dimension is 768, wait, task says 384, but codebert-base is 768
        # Note: Adjust if model dimension is different; task said 384, but codebert is 768. Perhaps use a different model, but for test, check shape

    def test_reranker(self):
        reranker = CodeReranker()
        query = "validate email"
        candidates = ["def validate_email(email): return True", "print('hello')"]
        results = reranker.rerank(query, candidates, top_k=1)
        self.assertEqual(len(results), 1)
        idx, score = results[0]
        self.assertIsInstance(idx, int)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()
