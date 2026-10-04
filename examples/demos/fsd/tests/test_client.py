"""The System One client: where it posts, what it sends (no key), what it reads back."""

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev.client import JevClient, JevError  # noqa: E402
from jev.config import Settings  # noqa: E402


class Handler(BaseHTTPRequestHandler):
    seen = []
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append((self.path, dict(self.headers), body))
        if Handler.status != 200:
            out = json.dumps({"detail": {"message": "nope"}}).encode()
            self.send_response(Handler.status)
        else:
            qs = body["questions"]
            answers = {
                q: {"type": "choice", "choice": next(iter(v["criteria"])), "probabilities": {k: 1 / len(v["criteria"]) for k in v["criteria"]}}
                for q, v in qs.items()
            }
            out = json.dumps({"model": "stub", "answers": answers, "usage": {"input_tokens": 40, "output_tokens": 1}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-decisio-server-ms", "12.5")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        out = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class ClientTests(unittest.TestCase):
    def setUp(self):
        Handler.seen, Handler.status = [], 200
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.client = JevClient(Settings(environ={"SYSTEMONE_BASE_URL": "http://127.0.0.1:%d" % self.server.server_port}, env_file=Path("/nonexistent")))

    def tearDown(self):
        self.server.shutdown()

    def test_posts_the_request_body_unchanged_with_no_key(self):
        qs = {"motion": {"type": "choice", "instructions": "go?", "criteria": {"drive": "go", "stop": "halt"}}}
        result = self.client.system_one({"speed": 3}, qs)
        path, headers, body = Handler.seen[0]
        self.assertEqual(path, "/v1/systemone")
        self.assertEqual(body, {"state": {"speed": 3}, "questions": qs})  # no model name unless one is set
        self.assertNotIn("Authorization", headers)
        self.assertEqual(result.answers["motion"]["choice"], "drive")
        meta = result.meta()
        self.assertEqual(meta["server_ms"], 12.5)
        self.assertEqual(meta["option_counts"], {"motion": 2})
        self.assertEqual((meta["input_tokens"], meta["cost_usd"]), (40, 0.0))
        self.assertTrue(self.client.configured and self.client.health() == {"ok": True})

    def test_model_is_sent_when_set(self):
        client = JevClient(Settings(environ={"SYSTEMONE_BASE_URL": self.client.base_url, "SYSTEMONE_MODEL": "m"}, env_file=Path("/nonexistent")))
        client.system_one({}, {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}}})
        self.assertEqual(Handler.seen[0][2]["model"], "m")

    def test_errors_are_reported(self):
        Handler.status = 422
        with self.assertRaises(JevError) as ctx:
            self.client.system_one({}, {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None}}})
        self.assertEqual(ctx.exception.status, 422)
        dead = JevClient(Settings(environ={"SYSTEMONE_BASE_URL": "http://127.0.0.1:1"}, env_file=Path("/nonexistent")))
        with self.assertRaises(JevError):
            dead.system_one({}, {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None}}})


if __name__ == "__main__":
    unittest.main()
