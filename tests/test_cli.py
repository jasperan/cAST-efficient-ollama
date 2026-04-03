import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cast_ollama.cli import main


class TestCLI(unittest.TestCase):
    def test_doctor_command(self):
        stdout = io.StringIO()
        with patch(
            "sys.argv",
            [
                "cast-ollama",
                "--action",
                "doctor",
                "--profile",
                "local",
            ],
        ), redirect_stdout(stdout):
            main()
        output = stdout.getvalue()
        self.assertIn("embedder backend", output.lower())
        self.assertIn("reranker backend", output.lower())

    def test_demo_command_creates_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "cast-ollama",
                    "--action",
                    "demo",
                    "--profile",
                    "local",
                    "--sample-file",
                    "examples/sample.py",
                    "--report-dir",
                    tmpdir,
                ],
            ), redirect_stdout(stdout):
                main()

            output = stdout.getvalue()
            self.assertIn("Demo complete", output)
            self.assertTrue((Path(tmpdir) / "demo_report.csv").exists())
            self.assertTrue((Path(tmpdir) / "demo_report.json").exists())


if __name__ == "__main__":
    unittest.main()
