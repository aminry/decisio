# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Client for the letters endpoint (`/v1/answer`, decisio.serve.vllm_engine). Standard library only.

    from decisio.serve.client import Client
    c = Client("http://localhost:8000")
    probs, timing = c.answer("Ticket 8842 ...", [
        {"kind": "noul", "instructions": "Is the customer reporting an outage?"},
        {"kind": "choice", "instructions": "Which team?", "options": ["billing", "technical", "sales"]},
    ], adapter=None)

    python -m decisio.serve.client http://localhost:8000          # health, then a demo request
"""
import json
import sys
import urllib.request


class Client:
    def __init__(self, url, timeout=600):
        self.url, self.timeout = url.rstrip("/"), timeout

    def _call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{e.code}: {e.read().decode()}") from None

    def health(self):
        return self._call("/health")

    def answer(self, state, questions, adapter=None):
        """One list of probabilities per question (in option order; yes/no is [yes, no]), and timing."""
        body = {"state": state, "questions": questions}
        if adapter:
            body["adapter"] = adapter
        out = self._call("/v1/answer", body)
        return [a["probs"] for a in out["answers"]], out["timing"]


if __name__ == "__main__":
    c = Client(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000")
    print(json.dumps(c.health(), indent=1))
    state = ("Ticket 8842. Customer writes: our API integration started returning 500 errors on every request "
             "about 20 minutes ago, and we cannot process customer orders until this is fixed.")
    qs = [{"kind": "noul", "instructions": "Is the customer reporting an outage?"},
          {"kind": "choice", "instructions": "Which team should handle this?",
           "options": ["billing", "technical", "sales", "account"]},
          {"kind": "score", "instructions": "How urgent is this?", "options": ["low", "medium", "high", "critical"]}]
    for name in [None] + c.health().get("adapters", []):
        probs, timing = c.answer(state, qs, adapter=name)
        print(f"adapter={name}: server {timing['server_ms']:.1f} ms")
        for q, p in zip(qs, probs):
            print(f"  {q['instructions'][:40]:40s} {[round(x, 3) for x in p]}")
