from typing import List, Dict
from cast_ollama.config import Config

class RandomChunker:
    def __init__(self, chunk_size: int = 500, overlap_percentage: int = 10):
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage

    def chunk(self, code: str) -> List[Dict]:
        chunks = []
        overlap_size = int(self.chunk_size * self.overlap_percentage / 100)
        start = 0
        code_length = len(code)

        lines = code.splitlines(True)  # Preserve line endings
        current_line = 1

        while start < code_length:
            end = min(start + self.chunk_size, code_length)
            chunk_content = code[start:end]

            # Calculate start and end lines
            chunk_lines = chunk_content.splitlines(True)
            start_line = current_line
            end_line = current_line + len(chunk_lines) - 1

            chunks.append({
                'content': chunk_content,
                'start_line': start_line,
                'end_line': end_line,
                'char_start': start,
                'char_end': end
            })

            start += self.chunk_size - overlap_size
            current_line = end_line - int(len(chunk_lines) * (self.overlap_percentage / 100)) + 1  # Approximate line overlap

        return chunks
