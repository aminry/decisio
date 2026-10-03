# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The calibration round's flags (`decisio.serve.systemone`), all off by default.

K1  with every flag at its default, the answers are exactly today's;
K2  --noul-commit: P(yes) strictly inside 0.20 to 0.80 goes to the band's edge on its own side (exactly 0.5 stays
    no), outside the band unchanged; choice and score answers untouched;
K3  --temperature-<type>: the type's temperature replaces the global one for that type only;
K4  --position-priors: the menu size's log prior subtracted in display order before the temperature; other sizes
    and score questions untouched; a yes/no question asked as letters takes the size-2 prior in display order;
K5  --score-rendering levels-noul: one yes/no row per level, the levels' P(yes) normalised; two-order requests keep
    letters.

  uv run pytest -q tests/unit/test_calibration_flags.py
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
    """A readout per question kind: yes/no (0.6, 0.4); choice and score by position, (0.4, 0.3, 0.2, 0.1) cut to the
    option count and renormalised; a level row 'Level: <name>' answers P(yes) from LEVEL_YES."""

    LEVEL_YES = {"low": 0.2, "mid": 0.6, "high": 0.2}

    def __init__(self):
        self.asked, self.adapters, self.tok = [], {}, tokenizer()

    def answer(self, state, questions, adapter=None):
        out = []
        for q in questions:
            self.asked.append(q)
            if q["kind"] == "noul":
                lvl = q["instructions"].rsplit("Level: ", 1)[-1] if "Level: " in q["instructions"] else None
                py = self.LEVEL_YES[lvl] if lvl else 0.6
                out.append(np.array([py, 1 - py]))
            else:
                w = np.array([0.4, 0.3, 0.2, 0.1][: len(q["options"])])
                out.append(w / w.sum())
        return out, {}


NOUL = {"type": "noul", "instructions": "Is it urgent?"}
CHOICE4 = {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None, "d": None}}
CHOICE2 = {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}
SCORE = {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}


def ask(questions, orders=None, **flags):
    eng = Fixed()
    s1 = SystemOne(eng, "m", **flags)
    body = {"state": "s", "questions": questions, **({"orders": orders} if orders else {})}
    return s1.answer(SystemOneRequest.model_validate(body))["answers"], eng.asked


def test_k1_defaults_unchanged():
    qs = {"n": NOUL, "c": CHOICE4, "s": SCORE}
    a, asked = ask(qs, temperature=1.307)
    assert [q["kind"] for q in asked] == ["noul", "choice", "score"]
    assert a["n"]["noul"] == pytest.approx(float(apply_temperature(np.array([0.6, 0.4]), 1.307)[0]))


COMMIT_CASES = [(0.6, 0.8), (0.5, 0.2), (0.45, 0.2), (0.9, 0.9), (0.1, 0.1), (0.8, 0.8), (0.2, 0.2)]


@pytest.mark.parametrize("py,want", COMMIT_CASES)
def test_k2_commit(py, want):
    assert commit_noul(np.array([py, 1 - py]))[0] == pytest.approx(want)


def test_k2_commit_only_noul():
    a, _ = ask({"n": NOUL, "c": CHOICE4}, noul_commit=True)
    assert a["n"]["noul"] == pytest.approx(0.8)
    assert a["c"] == ask({"n": NOUL, "c": CHOICE4})[0]["c"]


def test_k3_per_type_temperature():
    a, _ = ask({"n": NOUL, "c": CHOICE4}, temperature=1.307, temperatures={"noul": 2.0, "choice": None})
    assert a["n"]["noul"] == pytest.approx(float(apply_temperature(np.array([0.6, 0.4]), 2.0)[0]))
    base, _ = ask({"c": CHOICE4}, temperature=1.307)
    assert a["c"] == base["c"]  # unset type: the global temperature


def test_k4_position_priors():
    b4 = [0.3, 0.0, -0.1, -0.2]
    a, _ = ask({"c": CHOICE4, "c2": CHOICE2, "s": SCORE}, position_priors={"4": b4})
    lp = np.log(np.array([0.4, 0.3, 0.2, 0.1])) - np.array(b4)
    want = np.exp(lp - lp.max()) / np.exp(lp - lp.max()).sum()
    assert list(a["c"]["probabilities"].values()) == pytest.approx(list(want))
    base, _ = ask({"c2": CHOICE2, "s": SCORE})
    assert a["c2"] == base["c2"] and a["s"] == base["s"]  # other sizes and score questions untouched
    # a yes/no question asked as letters is a size-2 menu, false first: the prior applies in display order
    b2 = [0.5, 0.0]
    a, _ = ask({"n": NOUL}, noul_rendering="letters", position_priors={"2": b2})
    lp = np.log(np.array([0.4 / 0.7, 0.3 / 0.7])) - np.array(b2)  # display order (no, yes)
    py = float(np.exp(lp[1]) / np.exp(lp).sum())
    assert a["n"]["noul"] == pytest.approx(py)


def test_k5_score_levels_noul():
    a, asked = ask({"s": SCORE, "n": NOUL}, score_rendering="levels-noul")
    kinds = [q["kind"] for q in asked]
    assert kinds == ["noul", "noul", "noul", "noul"]  # three level rows, then the yes/no question
    assert asked[0]["instructions"].endswith("Does this level apply? Level: low")
    assert list(a["s"]["probabilities"].values()) == pytest.approx([0.2, 0.6, 0.2])
    assert a["n"]["noul"] == pytest.approx(0.6)
    _, asked2 = ask({"s": SCORE}, orders=2, score_rendering="levels-noul")
    assert {q["kind"] for q in asked2} == {"score"}  # two-order mode keeps letters
