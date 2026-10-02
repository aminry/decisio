"""The driving demo's server writes the request and the full answer of each model decision when asked to (added)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import server  # noqa: E402


class DecisionLogTest(unittest.TestCase):
    def test_logs_only_when_the_path_is_set(self):
        body = {"state": {"car": {"speed": 3}}, "questions": {"motion": {"type": "choice", "criteria": {"drive": None, "stop": None}}}}
        out = {"answers": {"motion": {"choice": "drive", "probabilities": {"drive": 0.8, "stop": 0.2}}}, "meta": {"latency_ms": 12.0}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "decisions.jsonl"
            os.environ.pop("DEMO_DECISION_LOG", None)
            server.log_decision(body, out)
            self.assertFalse(path.exists())
            os.environ["DEMO_DECISION_LOG"] = str(path)
            try:
                server.log_decision(body, out)
            finally:
                del os.environ["DEMO_DECISION_LOG"]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request"]["questions"], body["questions"])
        self.assertEqual(rows[0]["response"]["answers"]["motion"]["probabilities"], {"drive": 0.8, "stop": 0.2})
        self.assertEqual(rows[0]["meta"]["latency_ms"], 12.0)


if __name__ == "__main__":
    unittest.main()
