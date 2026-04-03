import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestCliEndToEnd(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.sample_file = self.repo_root / "examples" / "sample.py"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workdir = Path(self.temp_dir.name)
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(self.repo_root / "src")
        self.env["CHROMA_PATH"] = str(self.workdir / "chroma")
        self.base_cmd = [
            sys.executable,
            "-B",
            str(self.repo_root / "main.py"),
            "--embedding-backend",
            "hash",
            "--reranker-backend",
            "lexical",
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, *extra_args):
        return subprocess.run(
            [*self.base_cmd, *extra_args],
            cwd=self.workdir,
            env=self.env,
            check=True,
            text=True,
            capture_output=True,
        )

    def test_doctor_and_walkthrough(self):
        doctor = self._run("--action", "doctor", "--output", "json")
        payload = json.loads(doctor.stdout)
        self.assertTrue(payload["config"]["embedding_backend_resolved"].endswith("hash"))
        self.assertEqual(payload["config"]["reranker_backend_resolved"], "lexical")

        vectorize = self._run(
            "--action",
            "vectorize",
            "--chunking-method",
            "both",
            "--sample-file",
            str(self.sample_file),
        )
        self.assertIn("Vectorized and stored", vectorize.stdout)

        search = self._run(
            "--action",
            "search",
            "--chunking-method",
            "both",
            "--query",
            "validate email",
        )
        self.assertIn("CHUNKING STRATEGY COMPARISON REPORT", search.stdout)
        self.assertTrue((self.workdir / "search_report.csv").exists())
        self.assertTrue((self.workdir / "search_report.json").exists())


if __name__ == "__main__":
    unittest.main()
