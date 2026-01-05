import unittest
from unittest.mock import patch
from cast_ollama.metadata.extractor import OllamaMetadataExtractor, CodeMetadata

class TestOllamaMetadata(unittest.TestCase):
    def setUp(self):
        self.extractor = OllamaMetadataExtractor()

    @patch('cast_ollama.metadata.extractor.chat')
    def test_extract(self, mock_chat):
        mock_response = {
            'message': {
                'content': '{"purpose": "Test purpose", "input_params": "param1", "return_type": "bool", "docstring": "Test doc", "dependencies": ["re"], "complexity": "low"}'
            }
        }
        mock_chat.return_value = mock_response

        code = "def test(): pass"
        metadata = self.extractor.extract(code)
        self.assertIsInstance(metadata, CodeMetadata)
        self.assertEqual(metadata.purpose, "Test purpose")
        self.assertEqual(metadata.complexity, "low")

    def test_cache(self):
        code = "def test(): pass"
        # First call
        with patch('cast_ollama.metadata.extractor.chat') as mock_chat:
            mock_chat.return_value = {
                'message': {'content': '{"purpose": "Cached", "input_params": "", "return_type": "", "docstring": null, "dependencies": [], "complexity": "low"}'}
            }
            metadata1 = self.extractor.extract(code)

        # Second call, should use cache
        with patch('cast_ollama.metadata.extractor.chat') as mock_chat:
            mock_chat.side_effect = Exception("Should not be called")
            metadata2 = self.extractor.extract(code)

        self.assertEqual(metadata1.purpose, metadata2.purpose)

    @patch('cast_ollama.metadata.extractor.chat', side_effect=Exception("Ollama error"))
    def test_fallback(self, mock_chat):
        code = "def test(): pass"
        metadata = self.extractor.extract(code)
        self.assertEqual(metadata.purpose, "Unable to extract - check Ollama connection")
        self.assertEqual(metadata.complexity, "medium")

if __name__ == '__main__':
    unittest.main()
