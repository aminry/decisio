"""The one place the demo talks to a System One server.

Changed from upstream (see ../ATTRIBUTION.md): upstream called TypeSafe's hosted API with a key through the
`typesafe-sdk`. This version posts the same request body to `<base URL>/v1/systemone` on a server you run (a Decisio
server, or any server that speaks that wire format), with no key and no vendor SDK. The request and response JSON are
unchanged. Around the call sit what upstream had: request/response capture for the "Inspect JSON" drawer, and a call
counter (the spend guard stays as a counter and an optional rate limit; there is no price for a local server).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from .config import Settings

SDK_VERSION: Optional[str] = None
# a local server has no per-token price; kept so the HUD's cost column reads 0 instead of disappearing
PRICE_PER_INPUT_TOKEN_USD = 0.0
RETRYABLE_STATUSES = {408, 429, 502, 503, 504}


class JevError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class JevResult:
    """One System One call: typed answers plus everything the UI shows about the call."""

    def __init__(self, request: dict, response: dict, latency_ms: float, backend: str = "", server_ms: Optional[float] = None):
        self.request = request
        self.response = response
        self.latency_ms = latency_ms
        self.backend = backend
        self.server_ms = server_ms
        self.answers: Dict[str, dict] = response.get("answers", {})
        self.model: str = response.get("model", request.get("model", ""))
        self.usage: dict = response.get("usage") or {}

    @property
    def input_tokens(self) -> int:
        return int(self.usage.get("input_tokens") or 0)

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * PRICE_PER_INPUT_TOKEN_USD

    def meta(self) -> dict:
        return {
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "server_ms": self.server_ms,
            "question_count": len(self.request.get("questions", {})),
            "option_counts": {
                qid: len(q.get("criteria") or {}) for qid, q in self.request.get("questions", {}).items()
            },
            "input_tokens": self.input_tokens,
            "output_tokens": int(self.usage.get("output_tokens") or 0),
            "cost_usd": self.cost_usd,
            "backend": self.backend,
        }

    def trace(self) -> dict:
        return {"request": self.request, "response": self.response, "meta": self.meta()}


class SpendGuard:
    """Per-process call counter and an optional local rate limit (0 = none)."""

    def __init__(self, budget_usd: float = 0.0, rpm: int = 0):
        self.budget_usd = budget_usd
        self.rpm = rpm
        self.spent_usd = 0.0
        self.live_calls = 0
        self.input_tokens = 0
        self._times: deque = deque()
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            now = time.monotonic()
            while self._times and self._times[0] < now - 60:
                self._times.popleft()
            if self.rpm > 0 and len(self._times) >= self.rpm:
                raise JevError("Local rate limit: more than %d calls in the last minute." % self.rpm, status=429)
            self._times.append(now)

    def record(self, result: JevResult) -> None:
        with self._lock:
            self.live_calls += 1
            self.input_tokens += result.input_tokens
            self.spent_usd += result.cost_usd

    def snapshot(self) -> dict:
        with self._lock:
            return {"budget_usd": self.budget_usd, "spent_usd": self.spent_usd, "live_calls": self.live_calls,
                    "input_tokens": self.input_tokens, "rpm": self.rpm}


class JevClient:
    """A System One client for `settings.base_url`. Thread-safe; one kept-alive connection per thread."""

    def __init__(self, settings: Optional[Settings] = None, guard: Optional[SpendGuard] = None, timeout: float = 30.0):
        self.settings = settings or Settings()
        self.timeout = timeout
        self.guard = guard or SpendGuard(self.settings.budget_usd, self.settings.rpm)
        self._local = threading.local()

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def base_url(self) -> str:
        return self.settings.base_url

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    @property
    def key_source(self) -> Optional[str]:
        return None

    @property
    def backend(self) -> str:
        return "system one at %s" % self.base_url

    def system_one(self, state: Any, questions: Dict[str, dict]) -> JevResult:
        payload: Dict[str, Any] = {"state": state, "questions": questions}
        if self.model:
            payload["model"] = self.model
        self.guard.check()
        started = time.perf_counter()
        response, headers = self._post("/v1/systemone", payload)
        latency_ms = (time.perf_counter() - started) * 1000
        server_ms = headers.get("x-decisio-server-ms")
        result = JevResult(payload, response, latency_ms, backend=self.backend,
                           server_ms=float(server_ms) if server_ms is not None else None)
        self.guard.record(result)
        return result

    def models(self) -> dict:
        data = urllib.request.urlopen(urllib.request.Request(self.base_url + "/v1/models"), timeout=5).read()
        return json.loads(data)

    def health(self) -> Optional[dict]:
        """What the server says about itself, or None (a short probe; never raises)."""
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(self.base_url + path, timeout=2) as r:
                    return json.loads(r.read() or b"null") or {}
            except (OSError, ValueError):
                continue
        return None

    def _connection(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            parts = urlsplit(self.base_url)
            cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
            conn = cls(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), timeout=self.timeout)
            self._local.conn = conn
        return conn

    def _post(self, path: str, body: dict, max_attempts: int = 3):
        data = json.dumps(body).encode()
        attempt = 0
        while True:
            attempt += 1
            try:
                conn = self._connection()
                conn.request("POST", path, data, {"Content-Type": "application/json"})
                resp = conn.getresponse()
                raw = resp.read()
                headers = {k.lower(): v for k, v in resp.getheaders()}
                status = resp.status
            except (OSError, http.client.HTTPException) as err:
                self._local.conn = None
                if attempt < max_attempts:
                    time.sleep(0.2 * attempt)
                    continue
                raise JevError("Could not reach the System One server at %s: %s" % (self.base_url, err)) from err
            if status == 200:
                return json.loads(raw.decode()), headers
            try:
                parsed = json.loads(raw.decode(errors="replace"))
            except ValueError:
                parsed = raw.decode(errors="replace")
            if status in RETRYABLE_STATUSES and attempt < max_attempts:
                time.sleep(0.2 * attempt)
                continue
            raise JevError(_error_message(status, parsed), status=status, body=parsed)


def _error_message(status: int, body: Any) -> str:
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and detail.get("message"):
        text = detail["message"]
    elif detail:
        text = json.dumps(detail)[:400]
    else:
        text = str(body)[:400]
    if status == 422:
        return "The server could not validate the request (422): %s" % text
    return "System One server error %s: %s" % (status, text)


# --- small helpers for building questions ----------------------------------------------


def noul(instructions: Any, yes: Any = None, no: Any = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if yes is not None or no is not None:
        q["criteria"] = {"true": yes, "false": no}
    return q


def choice(instructions: Any, criteria: Dict[str, Any]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: Any, levels: list) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def chosen(answer: dict) -> tuple:
    """(value, probability of that value) for a Choice answer."""
    value = answer["choice"]
    return value, float(answer.get("probabilities", {}).get(value, answer.get("confidence", 0.0)))


def ranked(answer: dict, limit: int = 3) -> list:
    probs = answer.get("probabilities", {})
    return [
        {"value": k, "prob": float(v)}
        for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    ]
