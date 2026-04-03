import unittest

from cast_ollama.chunking.ast_chunker import ASTChunker
from cast_ollama.chunking.random_chunker import RandomChunker


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.sample_code = '''
def hello_world():
    print("Hello, world!")

class TestClass:
    def method(self):
        return 42
'''

    def test_random_chunker(self):
        chunker = RandomChunker(chunk_size=50, overlap_percentage=10)
        chunks = chunker.chunk(self.sample_code)
        self.assertGreater(len(chunks), 0)
        self.assertIn('content', chunks[0])
        self.assertIn('start_line', chunks[0])
        self.assertIn('end_line', chunks[0])

    def test_ast_chunker_prefers_semantic_symbols(self):
        chunker = ASTChunker(chunk_size=1000, overlap_percentage=0)
        chunks = chunker.chunk(self.sample_code)
        chunk_map = {(chunk['chunk_type'], chunk.get('symbol_name')): chunk for chunk in chunks}

        self.assertIn(('function_definition', 'hello_world'), chunk_map)
        self.assertIn(('class_definition', 'TestClass'), chunk_map)
        self.assertIn(('function_definition', 'method'), chunk_map)
        self.assertEqual(chunk_map[('function_definition', 'method')]['parent_class'], 'TestClass')


if __name__ == '__main__':
    unittest.main()
