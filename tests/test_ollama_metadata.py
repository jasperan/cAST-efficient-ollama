import unittest
from unittest.mock import patch

from cast_ollama.metadata.extractor import CodeMetadata, OllamaMetadataExtractor


class TestOllamaMetadata(unittest.TestCase):
    def setUp(self):
        self.extractor = OllamaMetadataExtractor()

    @patch("cast_ollama.metadata.extractor.chat")
    def test_extract(self, mock_chat):
        mock_chat.return_value = {
            "message": {
                "content": '{"purpose": "Test purpose", "input_params": "param1", "return_type": "bool", "docstring": "Test doc", "dependencies": ["re"], "complexity": "low"}'
            }
        }

        metadata = self.extractor.extract("def test(): pass")
        self.assertIsInstance(metadata, CodeMetadata)
        self.assertEqual(metadata.purpose, "Test purpose")
        self.assertEqual(metadata.complexity, "low")

    def test_cache(self):
        code = "def test(): pass"
        with patch("cast_ollama.metadata.extractor.chat") as mock_chat:
            mock_chat.return_value = {
                "message": {
                    "content": '{"purpose": "Cached", "input_params": "", "return_type": "", "docstring": null, "dependencies": [], "complexity": "low"}'
                }
            }
            metadata1 = self.extractor.extract(code)

        with patch("cast_ollama.metadata.extractor.chat") as mock_chat:
            mock_chat.side_effect = Exception("Should not be called")
            metadata2 = self.extractor.extract(code)

        self.assertEqual(metadata1.purpose, metadata2.purpose)

    @patch("cast_ollama.metadata.extractor.chat", side_effect=Exception("Ollama error"))
    def test_fallback(self, _mock_chat):
        metadata = self.extractor.extract("def test(): pass")
        self.assertEqual(metadata.purpose, "Unable to extract - check Ollama connection")
        self.assertEqual(metadata.complexity, "medium")


if __name__ == "__main__":
    unittest.main()
