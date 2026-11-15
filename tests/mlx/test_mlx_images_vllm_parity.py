# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The Mac server's image answers against decisio's on vLLM, through the served request path (`/v1/systemone`).

Twenty noul and choice items with one picture each (the request bodies are `wire/<id>.json` of an image eval set, and
the vLLM answers are a JSON-lines file of `{"id", "answer"}` recorded from decisio on vLLM with the same base and
budget). The server is started here as a subprocess, so the whole path is the served one: prompt, image tokens, the
temperatures, the readout. Run under the machine's memory guard and on its own (the server holds the model):

  DECISIO_MLX_IMAGES_MODEL=<dir or repository> [DECISIO_MLX_IMAGES_VISION=<add-on>] \\
  [DECISIO_MLX_IMAGES_BASE=<base key, or a tuned repository owner/name[@revision] served with its own defaults>] \\
  DECISIO_MLX_IMAGES_WIRE=<wire folder> DECISIO_MLX_IMAGES_VLLM_ANSWERS=<answers.jsonl> \\
  [DECISIO_MLX_IMAGES_BUDGET=280] uv run pytest -q -m mlx_model tests/mlx/test_mlx_images_vllm_parity.py

Two references. Same weights on both sides (the 12B in bf16 against bf16): the strict rule. Other weights on the Mac
(the 31B's 4-bit build against the FP8 card, the tuned 12B's 6-bit MLX conversion against its bf16 merge): set
`DECISIO_MLX_IMAGES_REFERENCE=different` and `DECISIO_MLX_IMAGES_EVALSET=<eval set folder>`.
Then 60 reporting-split noul and choice items with one picture are read.
No answer may flip outside a near-tie (vLLM's top two within 0.10; for noul, P(yes) within 0.05 of 0.5).
The Mac may be right on at most one item fewer than vLLM.
The spread (max, mean and median |dp|) is reported as measured, with no bar on the mean.

A tuned repository's yes/no commit rule (`--noul-commit`: P(yes) between 0.20 and 0.80 is reported as exactly 0.80 or
0.20 by its side of 0.5) would turn a near-tie into a 0.60 gap, so in this mode the Mac server runs with
`--no-noul-commit` and reports P(yes) as read. vLLM's recorded answers are what the rule left: where one is exactly
0.80 or 0.20 its own P(yes) is only known to lie on that side of 0.5 and within the band, so that item is compared by
side alone (no |dp|), and a different side is allowed when the Mac's P(yes) is within 0.10 of 0.5 (twice the near-tie
width, as vLLM's own value is hidden). Items outside the band are compared as any others.

The strict rule's bar is set from the measured spread: bf16 on two backends differ by a few hundredths at most. On the
12B base at 280 the 20 items differed by at most 0.024 and 0.004 on average, with no answer changed; the bar is
twice the maximum (0.05, and 0.01 on the mean), and a changed answer fails it.
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
        *([] if "/" in base else ["--model", os.environ["DECISIO_MLX_IMAGES_MODEL"]]),  # a tuned repository is the base
        "--images",
        "on",
        "--image-budget",
        os.environ.get("DECISIO_MLX_IMAGES_BUDGET", "280"),
        "--port",
        str(port),
    ]
    if os.environ.get("DECISIO_MLX_IMAGES_REFERENCE") == "different":
        cmd += ["--no-noul-commit"]  # the commit rule hides a near-tie behind 0.80 and 0.20 (module docstring)
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


def pick_items(evalset, per=15):
    """60 reporting-split noul and choice items with one picture: `per` of each of S noul, S choice, R noul, R choice,
    evenly spaced through the ids in order. Returns [(id, truth index into the probabilities)]."""
    out = []
    for part in ("S", "R"):
        rows = [json.loads(line) for line in open(f"{evalset}/{part}/items.jsonl")]
        for kind in ("noul", "choice"):
            pool = sorted(
                (r for r in rows if r["kind"] == kind and r["split"] == "report" and len(r["request"]["assets"]) == 1),
                key=lambda r: r["id"],
            )
            step = max(1, len(pool) // per)
            for r in pool[::step][:per]:
                out.append((r["id"], r["truth"]["q"]["answer"], kind, r["request"]["questions"]["q"].get("criteria")))
    return out


def truth_index(kind, truth, criteria):
    if kind == "noul":
        return 0 if truth is True else 1
    return sorted(criteria).index(str(truth))


def committed(kind, theirs):
    """Whether a recorded noul answer is one the commit rule moved to the band's edge."""
    return kind == "noul" and any(abs(theirs[0] - edge) < 1e-9 for edge in (0.2, 0.8))


def near_tie(kind, theirs, mine=None):
    if committed(kind, theirs):
        return abs(mine[0] - 0.5) < 0.10
    if kind == "noul":
        return abs(theirs[0] - 0.5) < 0.05
    top = sorted(theirs, reverse=True)
    return top[0] - top[1] < 0.10


def post(server, wire, name):
    body = json.load(open(f"{wire}/{name}.json"))
    req = urllib.request.Request(
        f"http://127.0.0.1:{server}/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-decisio-client": "test"},
    )
    return probabilities(find_answer(json.loads(urllib.request.urlopen(req, timeout=600).read())))


def reference_answers():
    ref = {}
    for line in open(os.environ["DECISIO_MLX_IMAGES_VLLM_ANSWERS"]):
        a = json.loads(line)
        ref[a["id"]] = a["answer"]
    return ref


@pytest.mark.skipif(
    os.environ.get("DECISIO_MLX_IMAGES_REFERENCE") != "different", reason="the other-weights comparison"
)
def test_p2_other_weights_differ_by_a_measured_spread_and_no_answer_flips_outside_near_ties(server):
    evalset = os.environ["DECISIO_MLX_IMAGES_EVALSET"]
    wire, ref = f"{evalset}/wire", reference_answers()
    rows = []
    for name, truth, kind, criteria in pick_items(evalset):
        mine, theirs = post(server, wire, name), probabilities(ref[name])
        t = truth_index(kind, truth, criteria)
        rows.append(
            {
                "id": name,
                "dp": None if committed(kind, theirs) else float(np.abs(np.array(mine) - np.array(theirs)).max()),
                "committed": committed(kind, theirs),
                "flip": int(np.argmax(mine) != np.argmax(theirs)),
                "tie": near_tie(kind, theirs, mine),
                "mac_right": int(np.argmax(mine) == t),
                "vllm_right": int(np.argmax(theirs) == t),
            }
        )
    dp = [r["dp"] for r in rows if r["dp"] is not None]
    report = {
        "n": len(rows),
        "n_compared_by_dp": len(dp),
        "n_committed_by_side": sum(r["committed"] for r in rows),
        "max": max(dp),
        "mean": float(np.mean(dp)),
        "median": float(np.median(dp)),
        "flips": sum(r["flip"] for r in rows),
        "flips_in_near_ties": sum(r["flip"] and r["tie"] for r in rows),
        "mac_right": sum(r["mac_right"] for r in rows),
        "vllm_right": sum(r["vllm_right"] for r in rows),
    }
    print("PARITY", json.dumps(report))
    if os.environ.get("DECISIO_MLX_IMAGES_REPORT"):
        with open(os.environ["DECISIO_MLX_IMAGES_REPORT"], "w") as f:
            json.dump({"summary": report, "rows": rows}, f, indent=1)
    assert [r["id"] for r in rows if r["flip"] and not r["tie"]] == []
    assert report["mac_right"] >= report["vllm_right"] - 1, report


@pytest.mark.skipif(os.environ.get("DECISIO_MLX_IMAGES_REFERENCE") == "different", reason="the same-weights comparison")
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
