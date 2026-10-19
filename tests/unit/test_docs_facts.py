# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Facts the docs state about the code, kept true by a test.

F1  the response's `model` is the name the request sent, or the served name when it sent none (docs/api.md,
    docs/migrating.md)
F2  the README's and docs/running.md's clone commands pin the release tag of this version, inside the markers that
    release-please rewrites at release time, and both files are in its config
F3  the README's Docker line names the image by digest, not by a tag that can move
F4  docs/running.md does not name the old `aminry/decisio-*` model repositories as current ones

  uv run pytest -q tests/unit/test_docs_facts.py
"""

import json
import re
import zlib
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

import decisio
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore

ROOT = Path(__file__).parents[2]


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return [zlib.crc32(w.encode()) % 50000 for w in str(text).split()]


class _Engine:
    tok = _Tok()
    multi_question = "sequential"
    pad_policy = "always"
    adapters: dict = {}

    def facts(self):
        return {"engine": "fake", "base": "fake-base"}

    def _prepare_separate(self, state, questions):
        return ([1, 2, 3, len(questions)],)

    def answer(self, state, questions, adapter=None, images=None):
        return [np.array([0.7, 0.3]) for _ in questions], {}


def test_f1_the_response_model_is_the_requests_or_the_served_name():
    app = FastAPI()
    add_routes(app, SystemOne(_Engine(), "the-served-name", task_store=TaskStore("fp")))
    c = TestClient(app)
    body = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    assert c.post("/v1/systemone", json={**body, "model": "jev-latest"}).json()["model"] == "jev-latest"
    assert c.post("/v1/systemone", json=body).json()["model"] == "the-served-name"


def test_f2_the_clones_pin_the_release_tag_and_release_please_moves_it():
    for name in ("README.md", "docs/running.md"):
        text = (ROOT / name).read_text()
        m = re.search(
            r"<!-- x-release-please-start-version -->\n```bash\n(.*?)```\n<!-- x-release-please-end -->", text, re.S
        )
        assert m, name
        assert f"git clone --branch v{decisio.__version__} https://github.com/aminry/decisio\n" in m.group(1), name
    cfg = json.loads((ROOT / "release-please-config.json").read_text())
    extra = cfg["packages"]["."]["extra-files"]
    assert {"type": "generic", "path": "README.md"} in extra and {"type": "generic", "path": "docs/running.md"} in extra


def test_f3_the_readmes_docker_line_is_a_digest():
    text = (ROOT / "README.md").read_text()
    assert re.search(r"docker run [^\n]*ghcr\.io/aminry/decisio@sha256:[0-9a-f]{64}\n", text)
    assert not re.search(r"docker run [^\n]*ghcr\.io/aminry/decisio:[0-9]", text)


def test_f4_running_md_names_the_current_model_repositories():
    text = (ROOT / "docs" / "running.md").read_text()
    assert "aminry/decisio-" not in text.replace("aminry/decisio-examples", "")
