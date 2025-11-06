import unittest
from unittest.mock import patch
from retrieval.search import SearchPipeline
import numpy as np

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.pipeline = SearchPipeline()

    @patch('retrieval.search.search_by_vector')
    def test_search(self, mock_search):
        mock_results = [
            {'chunk_id': '1', 'chunk_content': 'code1', 'distance': 0.1, 'chunk_type': 'function'},
            {'chunk_id': '2', 'chunk_content': 'code2', 'distance': 0.2, 'chunk_type': 'class'}
        ]
        mock_search.return_value = mock_results

        results = self.pipeline.search("test query", top_k_initial=2, top_k_final=2, use_rerank=False)
        self.assertEqual(len(results), 2)
        self.assertIn('score', results[0])

    @patch('retrieval.search.search_by_vector')
    def test_rerank(self, mock_search):
        mock_results = [
            {'chunk_id': '1', 'chunk_content': 'relevant code', 'distance': 0.1},
            {'chunk_id': '2', 'chunk_content': 'less relevant', 'distance': 0.2}
        ]
        mock_search.return_value = mock_results

        results = self.pipeline.search("test query", top_k_initial=2, top_k_final=1, use_rerank=True)
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0]['score'], 0)

if __name__ == '__main__':
    unittest.main()
