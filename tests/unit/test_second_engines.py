# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Every configuration that starts a second engine starts, on the CPU stand-in, with the served prompt.

S1  --image-model: the text engine and the image engine (decisio.serve.hf_letters.HFImageLettersEngine)
S2  the default start's intent head reads the serving engine itself (no second engine since two-engine mode was
    removed; decisio.serve.hidden_engine.HFReservedHiddenEngine on the CPU stand-in), with the server's prompt format
    and base; the removed --head-engine flag is refused with the upgrading note
S3  --one-engine --image-model: the image engine alone, serving both routes

The server's start-up runs for real (both engines load and build their prompts) up to the point where it would listen.
A change of the served prompt layout broke the head's rows silently from 0.3.0 to 0.6.0, so each configuration is
started here.
Each engine also answers its probe (decisio.serve.engine_health), which tells a failed request from a dead engine.
They load stand-in models on the CPU (the `slow` tier, one process per file): an --image-model start peaks at about
10.5 GB, which beside the fast tier's own process left a 16 GB runner without memory. Each configuration also starts in
a child process, whose exit returns all its memory before the next one starts.
The stand-in is the image route's (`DECISIO_IMAGE_TEST_MODEL`, a small checkpoint of the served model's family).

  uv run pytest -q tests/unit/test_second_engines.py
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow

MODEL = os.environ.get("DECISIO_IMAGE_TEST_MODEL", "Qwen/Qwen3.5-0.8B-Base")

# the server's main() with the given flags on the CPU stand-in, up to where it would listen; `text` is the text
# route's engine and `so` the SystemOne it serves, then the test's checks run
START = """
import sys

import uvicorn

from decisio.serve import vllm_engine

served = {}
vllm_engine.make_app = lambda engine, so: served.update(engine=engine, so=so) or object()
uvicorn.run = lambda app, **kw: None
sys.argv = ["decisio", "--backend", "hf", "--model", %r, *%r]
vllm_engine.main()
text, so = served["engine"], served["so"]
"""


def start_and_check(flags, checks: str):
    """Start the server with `flags` in a child process and run `checks` there; fail with its output otherwise."""
    code = START % (MODEL, list(flags)) + checks
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout[-4000:]}\n{r.stderr[-4000:]}"


def test_s1_image_model_starts():
    start_and_check(
        ["--image-model", MODEL],
        """
assert so.image_engine is not None and so.image_engine is not text
text.probe()  # each engine answers the probe that tells a failed request from a dead engine (engine_health)
so.image_engine.probe()
""",
    )


def test_s2_the_head_reads_the_serving_engine_with_the_served_format():
    start_and_check(
        [],
        """
hidden = so.hidden_engine
assert type(hidden).__name__ == "HFReservedHiddenEngine"
assert hidden.fmt == text.fmt and hidden.family == text.family
hidden.probe()
""",
    )
    code = START % (MODEL, ["--head-engine"])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 2 and "removed with two-engine mode" in r.stderr, r.stderr[-2000:]


def test_s3_one_engine_starts():
    start_and_check(
        ["--one-engine", "--image-model", MODEL],
        """
assert so.image_engine is text  # the image engine alone serves both routes
text.probe()
""",
    )
