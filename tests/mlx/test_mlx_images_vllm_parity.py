# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The Mac server's image answers against decisio's on vLLM, through the served request path (`/v1/systemone`).

Twenty noul and choice items with one picture each (the request bodies are `wire/<id>.json` of an image eval set, and
the vLLM answers are a JSON-lines file of `{"id", "answer"}` recorded from decisio on vLLM with the same base and
budget). The server is started here as a subprocess, so the whole path is the served one: prompt, image tokens, the
temperatures, the readout. Run under the machine's memory guard and on its own (the server holds the model):

  DECISIO_MLX_IMAGES_MODEL=<dir or repository> [DECISIO_MLX_IMAGES_VISION=<add-on>] [DECISIO_MLX_IMAGES_BASE=...] \\
  DECISIO_MLX_IMAGES_WIRE=<wire folder> DECISIO_MLX_IMAGES_VLLM_ANSWERS=<answers.jsonl> \\
  [DECISIO_MLX_IMAGES_BUDGET=280] uv run pytest -q -m mlx_model tests/mlx/test_mlx_images_vllm_parity.py

The bar is set from the measured spread: bf16 on two backends differ by a few hundredths at most. On the 12B base at 280
the 20 items differed by at most 0.024 and 0.004 on average, with no answer changed; the bar is twice the maximum, and a
changed answer fails it.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

pytestmark = [
    pytest.mark.mlx_model,
    pytest.mark.skipif(
        not all(os.environ.get(k) for k in ("DECISIO_MLX_IMAGES_MODEL", "DECISIO_MLX_IMAGES_VLLM_ANSWERS")),
        reason="DECISIO_MLX_IMAGES_MODEL and DECISIO_MLX_IMAGES_VLLM_ANSWERS are not set",
    ),
]
KINDS = (
    ("S-noul-presence", 5),
    ("R-noul-pos", 3),
    ("R-noul-neg", 2),
    ("S-choice-count", 3),
    ("S-choice-colour", 3),
    ("R-choice", 4),
)
MAX_BAR, MEAN_BAR = 0.05, 0.01


def probabilities(answer):
    if answer["type"] == "noul":
        return [answer["noul"], 1 - answer["noul"]]
    return [answer["probabilities"][k] for k in sorted(answer["probabilities"])]


def find_answer(o):
    if isinstance(o, dict):
        if "type" in o and ("noul" in o or "probabilities" in o):
            return o
        for v in o.values():
            if (f := find_answer(v)) is not None:
                return f
    return None


@pytest.fixture(scope="module")
def server():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    base = os.environ.get("DECISIO_MLX_IMAGES_BASE", "gemma-4-12b")
    cmd = [
        sys.executable,
        "-m",
        "decisio.serve.vllm_engine",
        "--backend",
        "mlx",
        "--base",
        base,
        "--model",
        os.environ["DECISIO_MLX_IMAGES_MODEL"],
        "--images",
        "on",
        "--image-budget",
        os.environ.get("DECISIO_MLX_IMAGES_BUDGET", "280"),
        "--port",
        str(port),
    ]
    if os.environ.get("DECISIO_MLX_IMAGES_VISION"):
        cmd += ["--vision-weights", os.environ["DECISIO_MLX_IMAGES_VISION"]]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            if proc.poll() is not None:
                pytest.fail(f"the server exited with {proc.returncode}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except OSError:
                time.sleep(2)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=60)


def test_p1_the_mac_server_answers_as_decisio_on_vllm_does(server):
    wire = os.environ["DECISIO_MLX_IMAGES_WIRE"]
    ref = {}
    for line in open(os.environ["DECISIO_MLX_IMAGES_VLLM_ANSWERS"]):
        a = json.loads(line)
        ref[a["id"]] = a["answer"]
    rows = []
    for kind, n in KINDS:
        for i in range(n):
            name = f"{kind}-{i:04d}"
            body = json.load(open(f"{wire}/{name}.json"))
            req = urllib.request.Request(
                f"http://127.0.0.1:{server}/v1/systemone",
                data=json.dumps(body).encode(),
                headers={"content-type": "application/json", "x-decisio-client": "test"},
            )
            mine = probabilities(find_answer(json.loads(urllib.request.urlopen(req, timeout=600).read())))
            theirs = probabilities(ref[name])
            rows.append(
                (
                    name,
                    float(np.abs(np.array(mine) - np.array(theirs)).max()),
                    int(np.argmax(mine) != np.argmax(theirs)),
                )
            )
    dp = [r[1] for r in rows]
    print(
        "PARITY",
        json.dumps({"n": len(rows), "max": max(dp), "mean": float(np.mean(dp)), "flips": sum(r[2] for r in rows)}),
    )
    assert sum(r[2] for r in rows) == 0, [r for r in rows if r[2]]
    assert max(dp) <= MAX_BAR and float(np.mean(dp)) <= MEAN_BAR, rows
