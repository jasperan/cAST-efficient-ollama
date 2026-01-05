import tree_sitter
import tree_sitter_python
from tree_sitter import Language, Parser
from typing import List, Dict, Optional
from cast_ollama.config import Config
import re

class ASTChunker:
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, overlap_percentage: int = Config.OVERLAP_PERCENTAGE):
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        # Modern tree-sitter initialization
        self.language = Language(tree_sitter_python.language())
        self.parser = Parser(self.language)

    def _count_non_whitespace(self, text: str) -> int:
        return len(re.sub(r'\s+', '', text))

    def _extract_node_text(self, code: str, node: tree_sitter.Node) -> str:
        return code[node.start_byte:node.end_byte]

    def _recursive_chunk(self, code: str, node: tree_sitter.Node, parent_type: Optional[str] = None) -> List[Dict]:
        chunks = []
        node_text = self._extract_node_text(code, node)
        node_size = self._count_non_whitespace(node_text)

        chunk_type = node.type
        function_name = None
        parent_class = parent_type if parent_type else None

        if chunk_type == 'function_definition':
            for child in node.children:
                if child.type == 'identifier':
                    function_name = self._extract_node_text(code, child)
                    break
        elif chunk_type == 'class_definition':
            for child in node.children:
                if child.type == 'identifier':
                    parent_class = self._extract_node_text(code, child)
                    break

        if node_size <= self.chunk_size:
            # If fits, add as chunk
            chunks.append({
                'content': node_text,
                'chunk_type': chunk_type,
                'function_name': function_name,
                'parent_class': parent_class,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
            })
        else:
            # Recurse into children
            small_chunks = []
            for child in node.children:
                small_chunks.extend(self._recursive_chunk(code, child, parent_class))

            # Merge small sibling nodes greedily
            merged = []
            current_merged = ''
            current_start = None
            current_end = None
            for sc in small_chunks:
                sc_text = sc['content']
                if self._count_non_whitespace(current_merged + sc_text) <= self.chunk_size:
                    current_merged += sc_text
                    if current_start is None:
                        current_start = sc['start_line']
                    current_end = sc['end_line']
                else:
                    if current_merged:
                        merged.append({
                            'content': current_merged,
                            'chunk_type': 'merged',
                            'start_line': current_start,
                            'end_line': current_end,
                        })
                    current_merged = sc_text
                    current_start = sc['start_line']
                    current_end = sc['end_line']
            if current_merged:
                merged.append({
                    'content': current_merged,
                    'chunk_type': 'merged',
                    'start_line': current_start,
                    'end_line': current_end,
                })
            chunks.extend(merged)

        # Add overlap to chunks
        for i in range(len(chunks) - 1):
            overlap_lines = int((chunks[i]['end_line'] - chunks[i]['start_line'] + 1) * self.overlap_percentage / 100)
            overlap_text = '\n'.join(code.splitlines()[chunks[i]['end_line'] - overlap_lines : chunks[i]['end_line']])
            chunks[i+1]['content'] = overlap_text + chunks[i+1]['content']
            chunks[i+1]['start_line'] = chunks[i]['end_line'] - overlap_lines + 1

        return chunks

    def chunk(self, code: str) -> List[Dict]:
        tree = self.parser.parse(bytes(code, 'utf8'))
        return self._recursive_chunk(code, tree.root_node)
