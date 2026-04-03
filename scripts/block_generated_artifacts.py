#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

PATTERNS = (
    ".omx/*",
    "docs/plans/*",
    "IMPLEMENTATION.md",
    "IMPLEMENTATION-*.md",
    "*_IMPLEMENTATION.md",
    "task_plan.md",
    "findings.md",
    "progress.md",
    "TESTING_REPORT.md",
    "AGENT_HARNESS_ANNEX.md",
    "SKILLS_EXECUTION_SUMMARY.md",
    "CODE_REVIEW.md",
    "SECURITY_REVIEW.md",
)


def is_blocked(path: str) -> bool:
    normalized = Path(path).as_posix()
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in PATTERNS)


def main(argv: list[str]) -> int:
    blocked = [path for path in argv if is_blocked(path)]
    if not blocked:
        return 0

    print("Refusing to commit generated agent artifacts:")
    for path in blocked:
        print(f"  - {path}")
    print("Move the file outside the repo or add it to an ignored location before committing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
