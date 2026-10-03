# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""--noul-commit (`decisio.serve.systemone.commit_noul`), off by default.

N1  off by default: a yes/no answer is the tempered readout, unchanged;
N2  on: P(yes) strictly inside 0.20 to 0.80 goes to the band's edge on its own side, exactly 0.5 stays no, and the
    band's edges and anything outside are unchanged;
N3  on: choice and score answers are untouched, and the served answer (yes above 0.5) never changes.

  uv run pytest -q tests/unit/test_noul_commit.py
"""

import functools

import numpy as np
import pytest

from decisio.serve.systemone import SystemOne, SystemOneRequest, apply_temperature, commit_noul

TOKENIZER = "Qwen/Qwen3-0.6B-Base"  # usage counting only (the stand-in of tests/unit/test_ties.py)


@functools.cache
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


class Fixed:
    """A readout per question kind: yes/no (P_YES, 1 - P_YES); choice and score (0.4, 0.3, 0.2, 0.1) cut to the
    option count and renormalised."""

    P_YES = 0.6

    def __init__(self):
        self.adapters, self.tok = {}, tokenizer()

    def answer(self, state, questions, adapter=None):
        out = []
        for q in questions:
            if q["kind"] == "noul":
                out.append(np.array([self.P_YES, 1 - self.P_YES]))
            else:
                w = np.array([0.4, 0.3, 0.2, 0.1][: len(q["options"])])
                out.append(w / w.sum())
        return out, {}


NOUL = {"type": "noul", "instructions": "Is it urgent?"}
CHOICE = {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None, "d": None}}
SCORE = {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}


def ask(**flags):
    s1 = SystemOne(Fixed(), "m", **flags)
    body = {"state": "s", "questions": {"n": NOUL, "c": CHOICE, "s": SCORE}}
    return s1.answer(SystemOneRequest.model_validate(body))["answers"]


def test_n1_off_by_default():
    a = ask(temperature=1.307)
    assert a["n"]["noul"] == pytest.approx(float(apply_temperature(np.array([0.6, 0.4]), 1.307)[0]))


CASES = [(0.6, 0.8), (0.5001, 0.8), (0.5, 0.2), (0.45, 0.2), (0.79, 0.8), (0.21, 0.2)]
CASES += [(0.8, 0.8), (0.2, 0.2), (0.9, 0.9)]  # the band's edges and outside: unchanged


@pytest.mark.parametrize("py,want", CASES)
def test_n2_transform(py, want):
    out = commit_noul(np.array([py, 1 - py]))
    assert out[0] == pytest.approx(want) and out.sum() == pytest.approx(1.0)
    assert (out[0] > 0.5) == (py > 0.5)  # the served answer never changes


def test_n3_only_yes_no():
    on, off = ask(temperature=1.307, noul_commit=True), ask(temperature=1.307)
    assert on["n"]["noul"] == pytest.approx(0.8)
    assert on["c"] == off["c"] and on["s"] == off["s"]
