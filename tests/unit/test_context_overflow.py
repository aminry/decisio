# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""A prompt over the context is the caller's error: 422 with a detail that says so, on every base, and the engine stays
up. vLLM 0.30.0 raises `VLLMValidationError` (a `VLLMClientError`, which is not a ValueError) from its renderer when the
rendered prompt is over `max_model_len`; before this the server answered it with a bare 500 (the JevBench maintainer
served the 31B and sent a 40,000-token state).

X1  /v1/systemone: a state over the context gets 422 and a detail naming the context and the remedy, on each base's
    served name; the next request is answered and /health stays ok (the engine was not probed, not declared dead)
X2  the other routes that score a prompt: /v1/answer, POST /v1/tasks (an example over the context) and
    POST /v1/abstention/tasks answer 422 as well
X3  the other kinds of vLLM client error: an unprocessable entity is 422, any other VLLMClientError 400 with its
    message; an error that is not a client error is still a 500 and still probed
X4  the detail: the context and the length vLLM states are in it, and the parameter vLLM names

  uv run pytest -q tests/unit/test_context_overflow.py
"""

import pytest

from decisio.families import BASES

MODEL = "Qwen/Qwen3-0.6B-Base"
LIMIT = 400  # the stand-in's "context", in tokens (its start-up warm-up prompt is shorter)
CRIT = {f"option_{i}": n for i, n in enumerate(["card_arrival", "card_linking", "exchange_rate", "out of scope"])}


# vLLM 0.30.0's classes (vllm/exceptions.py), matched by name by decisio
class VLLMClientError(Exception):
    pass


class VLLMValidationError(VLLMClientError):
    def __init__(self, message, *, parameter=None, value=None):
        super().__init__(message)
        self.parameter, self.value = parameter, value

    def __str__(self):
        base = super().__str__()
        extras = [f"{k}={v}" for k, v in (("parameter", self.parameter), ("value", self.value)) if v is not None]
        return f"{base} ({', '.join(extras)})" if extras else base


class VLLMUnprocessableEntityError(VLLMClientError):
    pass


class VLLMNotFoundError(VLLMClientError):
    pass


def over_context(n):  # vllm/renderers/params.py, the token check
    return VLLMValidationError(
        f"This model's maximum context length is {LIMIT} tokens. However, you requested 1 output tokens and your "
        f"prompt contains {n} input tokens, for a total of {n + 1} tokens. Please reduce the length of the input "
        "prompt or the number of requested output tokens.",
        parameter="input_tokens",
        value=n,
    )


class Limited:
    """The CPU stand-in with a context limit, as vLLM's renderer enforces it. One engine serves the whole file: a
    stand-in holds a 0.6B model in float32 (2.4 GB), and one per test took 14 GB of the CI runner's 16 and got the
    job cancelled."""

    engine = None

    @classmethod
    def get(cls):
        if cls.engine is None:
            from decisio.serve.hf_letters import HFLettersEngine

            class Engine(HFLettersEngine):
                armed, raises = False, None  # armed after the start-up warm-up

                def score_prompts(self, rows, *args, **kwargs):
                    for ids, _ in rows:
                        if self.armed and self.raises is not None:
                            raise self.raises
                        if len(ids) > LIMIT:
                            raise over_context(len(ids))
                    return super().score_prompts(rows, *args, **kwargs)

            cls.engine = Engine(MODEL, pad_to=None)
            cls.engine.armed = True
        return cls.engine


@pytest.fixture(scope="module", autouse=True)
def _release_the_engine():
    yield
    Limited.engine = None


def served(base, raises=None):
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne
    from decisio.serve.tasks import TaskStore
    from decisio.serve.vllm_engine import make_app

    eng = Limited.get()
    eng.family, eng.raises = BASES[base], raises
    so = SystemOne(eng, BASES[base].served_name, task_store=TaskStore("fp"))
    return TestClient(make_app(eng, so), raise_server_exceptions=False), eng, so


def request(state):
    q = {"type": "choice", "instructions": "Which option fits?", "criteria": CRIT}
    return {"state": state, "model": "m", "questions": {"q1": q}}


SHORT = "My card has not arrived yet."
LONG = "word " * 600


@pytest.mark.parametrize("base", sorted(BASES))
def test_x1_a_state_over_the_context_is_a_422(base):
    client, _, _ = served(base)
    assert client.post("/v1/systemone", json=request(SHORT)).status_code == 200
    r = client.post("/v1/systemone", json=request(LONG))
    assert r.status_code == 422, (r.status_code, r.text)
    detail = r.json()["detail"]
    assert isinstance(detail, str) and str(LIMIT) in detail and "shorten" in detail.lower(), detail
    # not a death, not a probe: the next request is answered and /health is ok
    assert client.post("/v1/systemone", json=request(SHORT)).status_code == 200
    assert client.get("/health").status_code == 200  # not declared dead


def test_x2_the_other_routes_that_score_a_prompt():
    client, _, _ = served("gemma-4-31b")
    r = client.post(
        "/v1/answer",
        json={"state": LONG, "questions": [{"kind": "choice", "instructions": "Which?", "options": ["a", "b"]}]},
    )
    assert r.status_code == 422 and str(LIMIT) in r.json()["detail"], (r.status_code, r.text)
    example = {"request": request(LONG), "answer": "option_0"}
    r = client.post("/v1/tasks", json={"id": "t", "examples": [example] * 12})
    assert r.status_code == 422 and str(LIMIT) in r.json()["detail"], (r.status_code, r.text)
    body = {"id": "a", "option": {"key": "option_3"}, "examples": [example] * 12}
    r = client.post("/v1/abstention/tasks", json=body)
    assert r.status_code == 422 and str(LIMIT) in r.json()["detail"], (r.status_code, r.text)


def test_x3_the_other_client_errors_and_the_rest():
    for exc, status in (
        (VLLMUnprocessableEntityError("the image URL answered 404"), 422),
        (VLLMNotFoundError("no adapter"), 400),
    ):
        client, _, _ = served("gemma-4-12b", raises=exc)
        r = client.post("/v1/systemone", json=request(SHORT))
        assert r.status_code == status and str(exc) in r.json()["detail"], (r.status_code, r.text)
        assert client.get("/health").status_code == 200
    # a failure that is not the request's is still a 500 (and the engine is probed, as before)
    client, _, _ = served("gemma-4-12b", raises=IndexError("a bug"))
    assert client.post("/v1/systemone", json=request(SHORT)).status_code == 500


def test_x4_the_detail_carries_what_vllm_states():
    client, _, _ = served("qwen3.6-35b-a3b")
    detail = client.post("/v1/systemone", json=request(LONG)).json()["detail"]
    assert f"maximum context length is {LIMIT} tokens" in detail
    assert "input_tokens" in detail and "prompt contains" in detail
