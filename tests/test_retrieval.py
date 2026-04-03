import unittest
from unittest.mock import patch

import numpy as np

from cast_ollama.retrieval.search import SearchPipeline


class DummyEmbedder:
    active_provider = 'local-hash'

    def embed(self, text):
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)


class DummyReranker:
    active_provider = 'lexical'

    def rerank(self, query, candidates, top_k):
        return [(0, 0.95), (1, 0.25)][:top_k]


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.pipeline = SearchPipeline(embedder=DummyEmbedder(), reranker=DummyReranker())

    @patch('cast_ollama.retrieval.search.search_by_vector')
    def test_search_adds_scores_and_provider_metadata(self, mock_search):
        mock_search.return_value = [
            {
                'chunk_id': 'method-1',
                'chunk_content': 'def validate_email(self, email): return True',
                'distance': 0.1,
                'chunk_type': 'function_definition',
                'parent_class': 'Validator',
                'symbol_name': 'validate_email',
            },
            {
                'chunk_id': 'class-1',
                'chunk_content': 'class Validator:\n    pass',
                'distance': 0.2,
                'chunk_type': 'class_definition',
                'class_name': 'Validator',
                'symbol_name': 'Validator',
            },
        ]

        results = self.pipeline.search('validate email', top_k_initial=2, top_k_final=2, use_rerank=True)
        self.assertEqual(len(results), 2)
        self.assertIn('score', results[0])
        self.assertEqual(results[0]['embedding_provider'], 'local-hash')
        self.assertEqual(results[0]['reranker_provider'], 'lexical')
        self.assertIn('parent_content', results[0])
        self.assertEqual(results[0]['parent_chunk_id'], 'class-1')

    @patch('cast_ollama.retrieval.search.search_by_vector')
    def test_search_without_rerank_uses_distance(self, mock_search):
        mock_search.return_value = [
            {'chunk_id': '1', 'chunk_content': 'code1', 'distance': 0.1, 'chunk_type': 'function_definition'},
            {'chunk_id': '2', 'chunk_content': 'code2', 'distance': 0.2, 'chunk_type': 'class_definition'},
        ]

        results = self.pipeline.search('test query', top_k_initial=2, top_k_final=2, use_rerank=False)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]['score'], 0.9)
        self.assertIsNone(results[0]['reranker_provider'])


if __name__ == '__main__':
    unittest.main()
