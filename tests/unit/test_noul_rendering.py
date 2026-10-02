# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""--noul-rendering: how a yes/no question is asked (`decisio.serve.systemone.noul_as_letters`, `SystemOne`).

N1  words (the default): the engine gets today's yes/no question, unchanged;
N2  letters: a two-option choice, the false side first, each side its criteria description, "No" and "Yes" for a
    side without one; the answer is P(the true option), however the engine orders it;
N3  letters-keys: as N2 with the sides named, "No: <false>" and "Yes: <true>";
N4  choice and score questions are asked the same under every setting;
N5  sides that would read the same, two-order requests and the abstain option keep words;
N6  the rendering enters a yes/no question's task key, never a choice question's; registration scores as served.

  uv run pytest -q tests/unit/test_noul_rendering.py
"""

import functools

import numpy as np
import pytest

from decisio.serve.systemone import SystemOne, SystemOneRequest, noul_as_letters, render_text
from decisio.serve.tasks import task_key

CRIT = {"true": "The text states that this is so", "false": "The text states that this is not so"}
TOKENIZER = "Qwen/Qwen3-0.6B-Base"  # usage counting only (the stand-in of tests/unit/test_ties.py)


@functools.cache
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


class TextEngine:
    """Answers from the options' text alone, so a reversed readout would show: the true side (its description, or
    "Yes") gets 0.8; a words-form yes/no question gets (0.7, 0.3)."""

    def __init__(self):
        self.asked, self.adapters, self.tok = [], {}, tokenizer()

    def answer(self, state, questions, adapter=None):
        out = []
        for q in questions:
            self.asked.append(q)
            if q["kind"] == "noul":
                out.append(np.array([0.7, 0.3]))
            else:
                w = np.array([0.8 if ("Yes" in o or o == CRIT["true"]) else 0.2 for o in q["options"]])
                out.append(w / w.sum())
        return out, {}


def ask(rendering, question, **extra):
    eng = TextEngine()
    s1 = SystemOne(eng, "m", noul_rendering=rendering, abstain_option=extra.pop("abstain_option", None))
    body = {"state": "s", "questions": {"q": question}, **extra}
    out = s1.answer(SystemOneRequest.model_validate(body), imajev_ext=extra.get("images") is not None)
    return out["answers"]["q"], eng.asked


def noul(criteria=None):
    q = {"type": "noul", "instructions": "Has the order been shipped?"}
    return {**q, "criteria": criteria} if criteria is not None else q


def test_n1_words_is_the_default():
    a, asked = ask("words", noul(CRIT))
    assert asked[0]["kind"] == "noul" and "Yes means: " + CRIT["true"] in asked[0]["instructions"]
    assert a == {"type": "noul", "noul": 0.7}
    assert SystemOne(TextEngine(), "m").noul_rendering == "words"


def test_n2_letters():
    a, asked = ask("letters", noul(CRIT))
    assert asked[0] == {
        "kind": "choice",
        "instructions": "Has the order been shipped?",
        "options": [CRIT["false"], CRIT["true"]],
    }
    assert a["type"] == "noul" and a["noul"] == pytest.approx(0.8)  # P(true option), the engine's order reversed
    _, asked = ask("letters", noul())
    assert asked[0]["options"] == ["No", "Yes"]
    _, asked = ask("letters", noul({"true": "Shipped", "false": None}))
    assert asked[0]["options"] == ["No", "Shipped"]


def test_n3_letters_keys():
    a, asked = ask("letters-keys", noul(CRIT))
    assert asked[0]["options"] == ["No: " + CRIT["false"], "Yes: " + CRIT["true"]]
    assert a["noul"] == pytest.approx(0.8)
    _, asked = ask("letters-keys", noul())
    assert asked[0]["options"] == ["No", "Yes"]


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Invoices", "access": None}},
        {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
    ],
)
def test_n4_other_questions_unchanged(question):
    seen = [ask(r, question) for r in ("words", "letters", "letters-keys")]
    assert seen[0] == seen[1] == seen[2]


def test_n5_words_kept():
    same = {"true": "Same", "false": "Same"}
    assert (
        noul_as_letters(SystemOneRequest.model_validate({"state": "s", "questions": {"q": noul(same)}}).questions["q"])
        is None
    )
    _, asked = ask("letters", noul(same))
    assert asked[0]["kind"] == "noul"
    _, asked = ask("letters", noul(CRIT), orders=2)
    assert {q["kind"] for q in asked} == {"noul"}
    _, asked = ask("letters", noul(CRIT), abstain_option="can't tell", images=[])
    assert asked[0]["kind"] == "choice" and asked[0]["options"][:2] == ["yes", "no"]  # the words form with the option


def test_n6_task_key_and_registration():
    q = SystemOneRequest.model_validate({"state": "s", "questions": {"q": noul(CRIT)}}).questions["q"]
    c = SystemOneRequest.model_validate(
        {
            "state": "s",
            "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}},
        }
    ).questions["q"]
    words, letters = SystemOne(TextEngine(), "m"), SystemOne(TextEngine(), "m", noul_rendering="letters")
    assert words.noul_rendered(q) is None and letters.noul_rendered(q) == "letters"
    assert task_key(q, render_text, False, letters.noul_rendered(q)) != task_key(q, render_text)
    assert letters.noul_rendered(c) is None

    class Store:  # records what registration scores, as served
        def register(self, task_id, examples, score, hidden=None, key_fn=None):
            self.scored, self.key = score([(s, q) for q, s, _ in examples]), key_fn(examples[0][0])

    letters.task_store = Store()
    letters.register_readout_task("t", [("s", q, True)])
    assert letters.task_store.scored[0][0] == pytest.approx(0.8)  # (yes, no) order, as answers are read
    assert letters.task_store.key == task_key(q, render_text, False, "letters")
