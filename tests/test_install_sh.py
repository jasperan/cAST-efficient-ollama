"""Regression: install.sh must not execute content from an unverified pre-existing directory."""
import os
import stat
import subprocess
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parents[1] / "install.sh"


def _run(tmp_path: Path, origin: str | None):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    git = stubs / "git"
    git.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo git version 2.43.0; exit 0; fi\n'
        'if [ "$1" = "-C" ] && [ "$3" = "remote" ]; then echo "' + (origin or "") + '"; exit 0; fi\n'
        "exit 128\n"
    )
    git.chmod(git.stat().st_mode | stat.S_IEXEC)
    target = tmp_path / "install"
    (target / ".venv" / "bin").mkdir(parents=True)
    marker = tmp_path / "marker"
    (target / ".venv" / "bin" / "activate").write_text(f"touch {marker}\n")
    if origin is not None:
        (target / ".git").mkdir()
    env = {
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "PROJECT_DIR": str(target),
        "REPO_URL": "https://example.invalid/r.git",
        "HOME": str(tmp_path),
    }
    proc = subprocess.run(["bash", str(INSTALL_SH)], env=env, capture_output=True, text=True, timeout=60)
    return proc, marker


def test_rejects_directory_that_is_not_a_checkout(tmp_path):
    proc, marker = _run(tmp_path, origin=None)
    assert proc.returncode != 0
    assert not marker.exists()


def test_rejects_checkout_with_foreign_origin(tmp_path):
    proc, marker = _run(tmp_path, origin="https://evil.invalid/x.git")
    assert proc.returncode != 0
    assert "Refusing" in proc.stdout
    assert not marker.exists()
