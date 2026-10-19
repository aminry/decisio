# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The generated reference documents (`decisio.serve.reference_docs`): `docs/reference/openapi.json` and
`docs/reference/flags.md`.

D1  the files in docs/reference are what the code gives (a change to a route, a model or a flag brings its regenerated
    files; release-please rewrites the version in both at release time)
D2  the OpenAPI document is complete: every route the app serves is in it, every reference resolves, the version is the
    package's, and System One's request and response models are in it
D3  the response models describe what the real route returns (noul, choice and score answers, usage, certified fields)
D4  the flags table has a row for every flag of the server's parser and one line per row
D5  release-please is told to rewrite the version in both files at release time

  uv run pytest -q tests/unit/test_reference_docs.py
  uv run python -m decisio.serve.reference_docs --write     # after changing a route, a model or a flag
"""

import json
import re
import zlib
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

import decisio
from decisio.serve import reference_docs as rd
from decisio.serve.systemone import SystemOne, add_routes
from decisio.serve.tasks import TaskStore

ROOT = Path(__file__).parents[2]
REF = ROOT / "docs" / "reference"


def test_d1_the_files_are_what_the_code_gives():
    assert rd.main(["--check", "--out", str(REF)]) == 0


def test_d2_the_openapi_document_is_complete():
    doc = rd.openapi()
    app_paths = {r.path for r in rd.build_app().routes if getattr(r, "methods", None)} - {
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
    assert set(doc["paths"]) == app_paths
    refs = set(re.findall(r'"\$ref": "#/components/schemas/([^"]+)"', json.dumps(doc)))
    assert refs <= set(doc["components"]["schemas"]), refs - set(doc["components"]["schemas"])
    assert doc["info"]["version"] == decisio.__version__
    so = doc["paths"]["/v1/systemone"]["post"]
    assert set(so["requestBody"]["content"]) == {"application/json", "multipart/form-data"}
    assert so["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SystemOneResponse"
    }
    assert {"x-decisio-server-ms", "x-decisio-route"} <= set(so["responses"]["200"]["headers"])
    assert {
        "SystemOneRequest",
        "SystemOneResponse",
        "NoulAnswer",
        "ChoiceAnswer",
        "ScoreAnswer",
        "TaskRegistration",
    } <= set(doc["components"]["schemas"])
    for path, ops in doc["paths"].items():
        for method, op in ops.items():
            assert op.get("summary"), (method, path)


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
        out = []
        for q in questions:
            k = len(q["options"]) if q.get("options") else 2
            p = np.full(k, 0.3 / (k - 1))
            p[0] = 0.7
            out.append(p)
        return out, {}


def test_d3_the_models_describe_the_routes_responses():
    app = FastAPI()
    add_routes(app, SystemOne(_Engine(), "decisio-test", task_store=TaskStore("fp")))
    c = TestClient(app)
    body = {
        "state": "s",
        "model": "m",
        "questions": {
            "n": {"type": "noul", "instructions": "Is it?"},
            "c": {"type": "choice", "criteria": {"a": None, "b": None}},
            "s": {"type": "score", "criteria": ["x", "y", "z"]},
        },
    }
    r = c.post("/v1/systemone", json=body)
    assert r.status_code == 200
    got = rd.SystemOneResponse.model_validate(r.json())
    assert [type(a).__name__ for a in got.answers.values()] == ["NoulAnswer", "ChoiceAnswer", "ScoreAnswer"]
    assert got.usage.output_tokens == 3
    assert rd.ModelList.model_validate(c.get("/v1/models").json()).models[0].name == "decisio-test"
    # the certified fields are optional extras of an answer
    with_cert = {
        **r.json(),
        "answers": {
            "n": {
                **r.json()["answers"]["n"],
                "certified": False,
                "certificate": "c1",
                "certified_reason": "below_threshold",
            }
        },
    }
    assert rd.SystemOneResponse.model_validate(with_cert).answers["n"].certified is False
    # the examples of the documentation validate
    rd.TaskRegistration.model_validate(
        {
            "id": "t",
            "examples": [{"request": {**body, "questions": {"q": body["questions"]["c"]}}, "answer": "a"}],
            "certify": {"alpha": 0.05},
        }
    )


def test_d4_the_flags_table_has_a_row_for_every_flag():
    from decisio.serve.vllm_engine import build_parser

    text = rd.flags_markdown()
    rows = [line for line in text.splitlines() if line.startswith("| `--")]
    flags = [a for a in build_parser()._actions if a.option_strings and a.dest != "help"]
    assert len(rows) == len(flags) > 40
    for a in flags:
        assert any(f"`{a.option_strings[0]}`" in row for row in rows), a.option_strings
    assert all(row.count("|") >= 5 for row in rows)
    assert f"decisio {decisio.__version__} <!-- x-release-please-version -->" in text


def test_d5_release_please_rewrites_both_files_at_release_time():
    cfg = json.loads((ROOT / "release-please-config.json").read_text())
    extra = cfg["packages"]["."]["extra-files"]
    assert {"type": "json", "path": "docs/reference/openapi.json", "jsonpath": "$.info.version"} in extra
    assert {"type": "generic", "path": "docs/reference/flags.md"} in extra
