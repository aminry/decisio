# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A local System One endpoint in front of another player, so every demo can be played by it unchanged.

The three demos post `/v1/systemone` requests to a base URL. This gateway listens on 127.0.0.1 and answers them from:
  systemone  another System One API with a bearer key (for example a hosted decision API): the request is forwarded
             as it is, with `--model` set
  anthropic  Claude through the Anthropic SDK; openai through the OpenAI SDK; gemini through the Google GenAI SDK;
  openrouter any of them through OpenRouter's OpenAI-compatible API (the OpenAI SDK), with the provider pinned by
             `--provider` (fallbacks off, the structured-output parameter required) and the serving provider recorded.
             The request is rendered once for a chat model: the same state and the same questions (instructions and
             every option as `key: description`), with the answer constrained by a JSON schema whose properties are
             the questions, each an enum of its option keys, at temperature 0. A chat answer has no distribution, so
             the chosen key gets probability 1 and the answer says so (`probabilities_source`).
Keys are read from a file (`--key-file`) when the gateway starts and are never written anywhere. Every call is appended
to `--log` (the request, the answer, the upstream time), and `--max-calls` refuses calls beyond a budget (a ledger file
keeps the count across restarts). The upstream time is returned in `x-gateway-upstream-ms`; the demos time the whole
round trip.

    python systemone_gateway.py --backend anthropic --model claude-haiku-4-5 --key-file ~/.config/anthropic/key \\
        --port 18200 --log runs/<run>/gateway.jsonl
"""

from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

SYSTEM = (
    "You make one decision for each question below, about the state given. "
    "Answer with the key of exactly one option per question, as JSON, and nothing else."
)


def text_of(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def render_chat(body: dict) -> tuple[str, dict, dict]:
    """(user text, JSON schema of the answer, {question: [keys]}) for a request of choice questions."""
    lines = ["State:", text_of(body["state"]), "", "Questions:"]
    keys, props = {}, {}
    for name, q in body["questions"].items():
        if q.get("type") != "choice":
            raise ValueError(f"question {name!r}: only choice questions are rendered for a chat model")
        keys[name] = list(q["criteria"])
        lines += ["", f"{name}: {text_of(q.get('instructions')) or 'Which option applies?'}", "Options:"]
        for k, d in q["criteria"].items():
            lines.append(f"- {k}: {text_of(d)}" if text_of(d) else f"- {k}")
        props[name] = {"type": "string", "enum": keys[name]}
    schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return "\n".join(lines), schema, keys


def one_hot_answers(choices: dict, keys: dict) -> dict:
    out = {}
    for name, ks in keys.items():
        c = choices.get(name)
        if c not in ks:
            raise ValueError(f"the model answered {c!r} for {name!r}, not one of its options")
        out[name] = {
            "type": "choice",
            "choice": c,
            "confidence": 1.0,
            "probabilities": {k: float(k == c) for k in ks},
            "probabilities_source": "one-hot: a chat model's answer has no distribution",
        }
    return out


class Backend:
    """Turns a System One request into an answer body. `call` returns (answer body, upstream ms)."""

    def __init__(self, kind: str, model: str, key: str | None, base_url: str | None, extra: dict):
        self.kind, self.model, self.key, self.base_url, self.extra = kind, model, key, base_url, extra
        self.client = None
        if kind == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=key, max_retries=4)
        elif kind == "openai":
            import openai

            # base_url: another OpenAI-compatible server (used to check this path against a local model)
            self.client = openai.OpenAI(api_key=key or "none", base_url=base_url, max_retries=4)
        elif kind == "openrouter":
            import openai

            self.client = openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=4)
        elif kind == "gemini":
            from google import genai

            self.client = genai.Client(api_key=key)

    def call(self, body: dict) -> tuple[dict, float]:
        t = time.perf_counter()
        if self.kind == "systemone":
            out = self._systemone(body)
        else:
            text, schema, keys = render_chat(body)
            choices, usage = getattr(self, "_" + self.kind)(text, schema)
            out = {"model": self.model, "answers": one_hot_answers(choices, keys), "usage": usage}
            if self.kind == "openrouter":
                out["via"] = "OpenRouter"
        return out, (time.perf_counter() - t) * 1000

    def _systemone(self, body: dict) -> dict:
        u = urlparse(self.base_url)
        conn = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(
            u.netloc, timeout=90
        )
        payload = json.dumps({**body, "model": self.model}).encode()
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.key}"} if self.key else {})}
        try:
            conn.request("POST", (u.path.rstrip("/") or "") + "/v1/systemone", payload, headers)
            resp = conn.getresponse()
            raw = resp.read()
        finally:
            conn.close()
        if resp.status != 200:
            raise UpstreamError(resp.status, raw[:500].decode(errors="replace"))
        return json.loads(raw)

    def _anthropic(self, text: str, schema: dict) -> tuple[dict, dict]:
        r = self.client.messages.create(
            model=self.model,
            max_tokens=int(self.extra.get("max_tokens", 512)),
            temperature=0,
            system=SYSTEM,
            messages=[{"role": "user", "content": text}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if r.stop_reason == "refusal":
            raise UpstreamError(200, "refusal")
        out = json.loads(next(b.text for b in r.content if b.type == "text"))
        return out, {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}

    def _openai(self, text: str, schema: dict) -> tuple[dict, dict]:
        kw = {k: v for k, v in self.extra.items() if k in ("temperature", "reasoning_effort", "max_completion_tokens")}
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": schema, "strict": True},
            },
            **kw,
        )
        out = json.loads(r.choices[0].message.content)
        return out, {"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}

    def _openrouter(self, text: str, schema: dict) -> tuple[dict, dict]:
        provider = {"order": [self.extra["provider"]], "allow_fallbacks": False, "require_parameters": True}
        body = {"provider": provider, "usage": {"include": True}}
        if "reasoning" in self.extra:
            body["reasoning"] = self.extra["reasoning"]
        kw = {k: v for k, v in self.extra.items() if k in ("temperature", "max_tokens", "seed")}
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": schema, "strict": True},
            },
            extra_body=body,
            **kw,
        )
        extra = r.model_extra or {}
        u = r.usage
        usage = {
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "cost_usd": (u.model_extra or {}).get("cost"),
            "provider": extra.get("provider"),
            "generation_id": r.id,
        }
        return json.loads(r.choices[0].message.content), usage

    def _gemini(self, text: str, schema: dict) -> tuple[dict, dict]:
        cfg = {
            "system_instruction": SYSTEM,
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            **{k: v for k, v in self.extra.items() if k in ("thinking_config", "max_output_tokens")},
        }
        r = self.client.models.generate_content(model=self.model, contents=text, config=cfg)
        u = r.usage_metadata
        return json.loads(r.text), {"input_tokens": u.prompt_token_count, "output_tokens": u.candidates_token_count}


class UpstreamError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"upstream answered {status}: {detail}")
        self.status = status


class Ledger:
    """An append-only count of upstream calls per label, across restarts; refuses a call beyond the budget."""

    def __init__(self, path: Path | None, label: str, budget: int | None):
        self.path, self.label, self.budget, self.lock = path, label, budget, threading.Lock()

    def count(self) -> int:
        if not self.path or not self.path.exists():
            return 0
        return sum(1 for line in self.path.open() if json.loads(line).get("label") == self.label)

    def exhausted(self) -> bool:
        return self.budget is not None and self.count() >= self.budget

    def add(self, status: int) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.lock, self.path.open("a") as f:
                f.write(json.dumps({"t": time.time(), "label": self.label, "status": status}) + "\n")


def make_handler(backend: Backend, ledger: Ledger, log: Path | None):
    log_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict, extra: dict | None = None) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(
                    200, {"ok": True, "gateway": backend.kind, "model": backend.model, "calls": ledger.count()}
                )
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            status, out, ms = 200, None, None
            if ledger.exhausted():
                status, out = 429, {"error": f"the budget of {ledger.budget} calls for {ledger.label!r} is used"}
            else:
                try:
                    out, ms = backend.call(body)
                    ledger.add(200)
                except UpstreamError as e:
                    ledger.add(e.status)
                    status, out = (429 if e.status == 429 else 502), {"error": str(e)}
                except (ValueError, KeyError) as e:
                    ledger.add(422)
                    status, out = 422, {"error": str(e)}
            if log:
                with log_lock, log.open("a") as f:
                    row = {"t": time.time(), "status": status, "upstream_ms": ms, "request": body, "response": out}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._send(status, out, {"x-gateway-upstream-ms": f"{ms:.1f}"} if ms is not None else None)

        def log_message(self, *_):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["systemone", "anthropic", "openai", "gemini", "openrouter"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", default=None, help="a file holding the API key (read once, never written)")
    ap.add_argument(
        "--base-url",
        default=None,
        help="the upstream System One API (systemone), or another OpenAI-compatible server (openai)",
    )
    ap.add_argument("--extra", default="{}", help="backend options as JSON (reasoning_effort, thinking_config, ...)")
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--log", default=None, help="append every call here (request, answer, upstream time)")
    ap.add_argument("--ledger", default=None, help="append-only call count, kept across restarts")
    ap.add_argument("--label", default="calls", help="the ledger label the budget counts")
    ap.add_argument("--max-calls", type=int, default=None, help="refuse calls beyond this many for the label")
    a = ap.parse_args()
    key = Path(a.key_file).expanduser().read_text().strip() if a.key_file else None
    backend = Backend(a.backend, a.model, key, a.base_url, json.loads(a.extra))
    ledger = Ledger(Path(a.ledger) if a.ledger else None, a.label, a.max_calls)
    log = Path(a.log) if a.log else None
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(backend, ledger, log))
    print(
        f"System One gateway on 127.0.0.1:{a.port} -> {a.backend} {a.model} ({ledger.count()} calls on the ledger)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
