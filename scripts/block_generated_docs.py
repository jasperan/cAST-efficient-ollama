#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

BLOCK_PATTERNS = [
    re.compile(r"(^|/)\.omx/"),
    re.compile(r"(^|/)IMPLEMENTATION(?:-[^/]+)?\.md$", re.IGNORECASE),
    re.compile(r"(^|/)AGENT_HARNESS_ANNEX\.md$", re.IGNORECASE),
    re.compile(r"(^|/)SKILLS_EXECUTION_SUMMARY\.md$", re.IGNORECASE),
    re.compile(r"(^|/).+_FEATURES\.md$", re.IGNORECASE),
    re.compile(r"(^|/).+_ENHANCEMENTS\.md$", re.IGNORECASE),
    re.compile(r"(^|/).+_REPORT\.md$", re.IGNORECASE),
]

ALLOWLIST = {
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "project_summary_ollama.md",
    "cursor_prompt_ollama.md",
}


def main(argv: list[str]) -> int:
    blocked: list[str] = []
    for raw_path in argv[1:]:
        normalized = raw_path.replace("\\", "/")
        if Path(normalized).name in ALLOWLIST:
            continue
        if any(pattern.search(normalized) for pattern in BLOCK_PATTERNS):
            blocked.append(raw_path)

    if not blocked:
        return 0

    print("Refusing to commit generated/agent markdown artifacts:", file=sys.stderr)
    for path in blocked:
        print(f" - {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
