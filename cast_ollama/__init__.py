from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC_PACKAGE = _ROOT / "src" / "cast_ollama"

__path__ = [str(_SRC_PACKAGE)]
