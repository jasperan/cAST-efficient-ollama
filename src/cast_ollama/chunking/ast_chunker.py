from __future__ import annotations

import ast
import logging
import re
from typing import Dict, List, Optional

from cast_ollama.config import Config

try:
    import tree_sitter
    import tree_sitter_python
    from tree_sitter import Language, Parser
except Exception:  # pragma: no cover - optional dependency
    tree_sitter = None
    tree_sitter_python = None
    Language = None
    Parser = None

logger = logging.getLogger(__name__)


class ASTChunker:
    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, overlap_percentage: int = Config.OVERLAP_PERCENTAGE):
        self.chunk_size = chunk_size
        self.overlap_percentage = overlap_percentage
        self.backend = "python-ast"
        self.language = None
        self.parser = None

        if Language is not None and Parser is not None and tree_sitter_python is not None:
            try:
                self.language = Language(tree_sitter_python.language())
                self.parser = Parser()
                try:
                    self.parser.language = self.language
                except AttributeError:
                    self.parser = Parser(self.language)
                self.backend = "tree-sitter"
            except Exception as exc:  # pragma: no cover - depends on installed tree-sitter version
                logger.warning("tree-sitter backend unavailable (%s); using python ast fallback.", exc)
                self.language = None
                self.parser = None

    def _count_non_whitespace(self, text: str) -> int:
        return len(re.sub(r"\s+", "", text))

    def _extract_node_text(self, code: str, node: tree_sitter.Node) -> str:  # type: ignore[name-defined]
        return code[node.start_byte : node.end_byte]

    def _apply_overlap(self, code: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return chunks

        lines = code.splitlines()
        for index in range(1, len(chunks)):
            previous = chunks[index - 1]
            current = chunks[index]
            span = previous["end_line"] - previous["start_line"] + 1
            overlap_lines = int(span * self.overlap_percentage / 100)
            if overlap_lines <= 0:
                continue
            start = max(previous["end_line"] - overlap_lines, 0)
            overlap_text = "\n".join(lines[start : previous["end_line"]])
            if overlap_text:
                current["content"] = f"{overlap_text}\n{current['content']}"
                current["start_line"] = min(current["start_line"], previous["end_line"] - overlap_lines + 1)
        return chunks

    def _recursive_chunk(self, code: str, node: tree_sitter.Node, parent_type: Optional[str] = None) -> List[Dict]:  # type: ignore[name-defined]
        chunks: List[Dict] = []
        node_text = self._extract_node_text(code, node)
        node_size = self._count_non_whitespace(node_text)

        chunk_type = node.type
        function_name = None
        class_name = None
        parent_class = parent_type if parent_type else None

        if chunk_type == "module":
            for child in node.children:
                chunks.extend(self._recursive_chunk(code, child, parent_class))
            if chunks:
                return chunks

        if chunk_type == "function_definition":
            for child in node.children:
                if child.type == "identifier":
                    function_name = self._extract_node_text(code, child)
                    break
        elif chunk_type == "class_definition":
            for child in node.children:
                if child.type == "identifier":
                    class_name = self._extract_node_text(code, child)
                    break
            parent_class = None

        if chunk_type not in {"function_definition", "class_definition"}:
            for child in node.children:
                chunks.extend(self._recursive_chunk(code, child, parent_class))
        elif node_size <= self.chunk_size:
            chunks.append(
                {
                    "content": node_text,
                    "chunk_type": chunk_type,
                    "function_name": function_name,
                    "class_name": class_name,
                    "symbol_name": function_name or class_name,
                    "parent_class": parent_class,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                }
            )
            if chunk_type == "class_definition":
                for child in node.children:
                    chunks.extend(self._recursive_chunk(code, child, class_name))
        else:
            next_parent = class_name or parent_class
            for child in node.children:
                chunks.extend(self._recursive_chunk(code, child, next_parent))

        return chunks

    def _stdlib_chunks(self, code: str) -> List[Dict]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [
                {
                    "content": code,
                    "chunk_type": "module",
                    "function_name": None,
                    "symbol_name": None,
                    "parent_class": None,
                    "start_line": 1,
                    "end_line": len(code.splitlines()) or 1,
                }
            ]

        lines = code.splitlines()
        chunks: List[Dict] = []

        def visit(nodes: List[ast.AST], parent_class: Optional[str] = None) -> None:
            for node in nodes:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue

                start_line = getattr(node, "lineno", 1)
                end_line = getattr(node, "end_lineno", start_line)
                snippet = "\n".join(lines[start_line - 1 : end_line])
                chunk = {
                    "content": snippet,
                    "chunk_type": "class_definition" if isinstance(node, ast.ClassDef) else "function_definition",
                    "function_name": None if isinstance(node, ast.ClassDef) else node.name,
                    "class_name": node.name if isinstance(node, ast.ClassDef) else None,
                    "symbol_name": node.name,
                    "parent_class": None if isinstance(node, ast.ClassDef) else parent_class,
                    "start_line": start_line,
                    "end_line": end_line,
                }
                if self._count_non_whitespace(snippet) <= self.chunk_size or not isinstance(node, ast.ClassDef):
                    chunks.append(chunk)

                if isinstance(node, ast.ClassDef):
                    visit(node.body, node.name)

        visit(tree.body)
        if not chunks:
            chunks.append(
                {
                    "content": code,
                    "chunk_type": "module",
                    "function_name": None,
                    "symbol_name": None,
                    "parent_class": None,
                    "start_line": 1,
                    "end_line": len(lines) or 1,
                }
            )
        return self._apply_overlap(code, chunks)

    def chunk(self, code: str) -> List[Dict]:
        if self.parser is None:
            return self._stdlib_chunks(code)

        tree = self.parser.parse(bytes(code, "utf8"))
        chunks = self._recursive_chunk(code, tree.root_node)
        if not chunks:
            return self._stdlib_chunks(code)
        return self._apply_overlap(code, chunks)
