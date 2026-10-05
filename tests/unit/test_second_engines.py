# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Every configuration that starts a second engine starts, on the CPU stand-in, with the served prompt.

S1  --image-model: the text engine and the image engine (decisio.serve.hf_letters.HFImageLettersEngine)
S2  --head-engine: the text engine and a second copy for the intent head (decisio.serve.hidden_engine.HFHiddenEngine),
    whose token rows must equal the text engine's at start-up; it is given the server's prompt format and base
S3  --one-engine --image-model: the image engine alone, serving both routes

The server's start-up runs for real (both engines load and build their prompts) up to the point where it would listen.
A change of the served prompt layout broke S2 silently from 0.3.0 to 0.6.0, so each configuration is started here.
Each engine also answers its probe (decisio.serve.engine_health), which tells a failed request from a dead engine.
The stand-in is the image route's (`DECISIO_IMAGE_TEST_MODEL`, a small checkpoint of the served model's family).

  uv run pytest -q tests/unit/test_second_engines.py
"""

import os
import sys

MODEL = os.environ.get("DECISIO_IMAGE_TEST_MODEL", "Qwen/Qwen3.5-0.8B-Base")


def start(monkeypatch, *flags):
    """The server's main() with the given flags on the CPU stand-in, up to where it would listen; returns (the text
    route's engine, the SystemOne it serves)."""
    import uvicorn

    from decisio.serve import vllm_engine

    served = {}
    monkeypatch.setattr(vllm_engine, "make_app", lambda engine, so: served.update(engine=engine, so=so) or object())
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    monkeypatch.setattr(sys, "argv", ["decisio", "--backend", "hf", "--model", MODEL, *flags])
    vllm_engine.main()
    return served["engine"], served["so"]


def test_s1_image_model_starts(monkeypatch):
    text, so = start(monkeypatch, "--image-model", MODEL)
    assert so.image_engine is not None and so.image_engine is not text
    text.probe()  # each engine answers the probe that tells a failed request from a dead engine (engine_health)
    so.image_engine.probe()


def test_s2_head_engine_starts_with_the_served_format(monkeypatch):
    text, so = start(monkeypatch, "--head-engine")
    hidden = so.hidden_engine
    assert type(hidden).__name__ == "HFHiddenEngine"
    assert hidden.fmt == text.fmt and hidden.family == text.family
    hidden.probe()


def test_s3_one_engine_starts(monkeypatch):
    text, so = start(monkeypatch, "--one-engine", "--image-model", MODEL)
    assert so.image_engine is text  # the image engine alone serves both routes
    text.probe()
