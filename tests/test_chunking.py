import unittest
from cast_ollama.chunking.random_chunker import RandomChunker
from cast_ollama.chunking.ast_chunker import ASTChunker

class TestChunking(unittest.TestCase):
    def setUp(self):
        self.sample_code = """
def hello_world():
    print("Hello, world!")

class TestClass:
    def method(self):
        return 42
"""

    def test_random_chunker(self):
        chunker = RandomChunker(chunk_size=50, overlap_percentage=10)
        chunks = chunker.chunk(self.sample_code)
        self.assertGreater(len(chunks), 0)
        self.assertIn('content', chunks[0])
        self.assertIn('start_line', chunks[0])
        self.assertIn('end_line', chunks[0])

    def test_ast_chunker(self):
        chunker = ASTChunker(chunk_size=1000, overlap_percentage=10)
        chunks = chunker.chunk(self.sample_code)
        self.assertGreater(len(chunks), 0)
        self.assertIn('content', chunks[0])
        self.assertIn('chunk_type', chunks[0])
        self.assertIn('start_line', chunks[0])
        self.assertIn('end_line', chunks[0])

if __name__ == '__main__':
    unittest.main()
