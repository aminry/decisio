# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""--noul-rendering: how a yes/no question is asked (`decisio.serve.systemone.noul_as_letters`, `SystemOne`).

N1  words (SystemOne's default; the server's is letters-keys since 2026-10-03): the engine gets the words-form yes/no
    question, unchanged;
N2  letters: a two-option choice, the false side first, each side its criteria description, "No" and "Yes" for a
    side without one; the answer is P(the true option), however the engine orders it;
N3  letters-keys: as N2 with the sides named, "No: <false>" and "Yes: <true>";
N4  choice and score questions are asked the same under every setting;
N5  sides that would read the same, two-order requests and the abstain option keep words;
N7  the image route keeps words, the prompt its figures were measured with;
N6  the rendering enters a yes/no question's task key, never a choice question's; registration scores as served.
N8  a yes/no head reads the served (letters) question, its label log-probabilities in key order (yes, no), at
    registration, certification and serving, and in the debug readout

  uv run pytest -q tests/unit/test_noul_rendering.py
"""

import functools
import os

import numpy as np
import pytest

from decisio.serve.systemone import SystemOne, SystemOneRequest, noul_as_letters, render_text
from decisio.serve.tasks import task_key

CRIT = {"true": "The text states that this is so", "false": "The text states that this is not so"}
# usage counting only (the stand-in of tests/unit/test_ties.py)
TOKENIZER = os.environ.get("DECISIO_STAND_IN_MODEL", "Qwen/Qwen3-0.6B-Base")


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
        def register(self, task_id, examples, score, hidden=None, key_fn=None, temperature=1.0, **kw):
            self.scored, self.key = score([(s, q) for q, s, _ in examples]), key_fn(examples[0][0])
            return {}, {}  # (task, fitted), as TaskStore.register returns

    letters.task_store = Store()
    letters.register_readout_task("t", [("s", q, True)])
    assert letters.task_store.scored[0][0] == pytest.approx(0.8)  # (yes, no) order, as answers are read
    assert letters.task_store.key == task_key(q, render_text, False, "letters")


def test_n7_image_route_keeps_words():
    text, image = TextEngine(), TextEngine()
    s1 = SystemOne(text, "m", noul_rendering="letters-keys", image_engine=image)
    req = SystemOneRequest.model_validate({"state": "s", "questions": {"q": noul(CRIT)}})
    out = s1.answer(req, route="image")
    assert not text.asked and image.asked[0]["kind"] == "noul" and out["answers"]["q"]["noul"] == pytest.approx(0.7)
    s1.answer(req)  # the text route: letters-keys
    assert text.asked[0]["kind"] == "choice"


class HiddenEngine:
    """The hidden-state readout of TextEngine's prompts: lp the log of TextEngine's answer, in engine order, and h a
    marker of the form read ([1, 0] the letters choice, [0, 1] the words form), so a head that read another prompt, or
    the letters in engine order, would show."""

    reader = None

    def __init__(self):
        self.text = TextEngine()

    def readout(self, state, questions):
        probs, _ = self.text.answer(state, questions)
        return [
            (np.log(p), np.array([0.0, 1.0]) if q["kind"] == "noul" else np.array([1.0, 0.0]))
            for p, q in zip(probs, questions)
        ]

    def read_info(self):
        return None


def test_n8_a_yes_no_head_reads_the_served_question_in_key_order(monkeypatch):
    """A yes/no head under letters is fitted, certified and served on the letters question (the one served), with
    its label log-probabilities reversed to (yes, no), the task's key order; the debug readout's hidden_lp is in that
    order too. (Heads are fitted for 10 or more options; the floor is lowered here to reach the path.)"""
    from decisio.readout import intent_head
    from decisio.serve import tasks
    from decisio.serve.systemone import key_order

    assert np.array_equal(key_order(np.array([1.0, 2.0]), True), [2.0, 1.0])
    seen = {}

    def fit(logps, H, labels, options, **kw):  # a head that adds nothing: it serves softmax(lp), so the order shows
        seen.update(lps=np.asarray(logps), H=np.asarray(H))
        return {
            "format": intent_head.FORMAT, "applied": True, "K": 2, "options": options,
            "options_sha256": intent_head.options_digest(options), "A": np.zeros((2, 2)), "c": np.zeros(2),
        }  # fmt: skip

    monkeypatch.setattr(tasks, "MIN_OPTIONS_FOR_HEAD", 2)
    monkeypatch.setattr(intent_head, "fit_intent_head", fit)
    s1 = SystemOne(TextEngine(), "m", noul_rendering="letters", debug_readout=True)
    s1.hidden_engine = HiddenEngine()
    s1.task_store = tasks.TaskStore("fp")
    q = SystemOneRequest.model_validate({"state": "s", "questions": {"q": noul(CRIT)}}).questions["q"]
    task, _ = s1.register_readout_task("t", [(f"s{i}", q, i % 2 == 0) for i in range(4)])
    assert task["head"]["applied"] and task["options"] == ["yes", "no"]
    assert np.allclose(np.exp(seen["lps"]), [0.8, 0.2]) and np.array_equal(seen["H"], [[1.0, 0.0]] * 4)
    assert np.allclose(s1.served_rows(task, [("s", q)])[0], [0.8, 0.2])  # certification reads as serving does
    body = {"state": "s", "questions": {"q": noul(CRIT)}}
    out = s1.answer(SystemOneRequest.model_validate(body), debug="readout")
    assert out["answers"]["q"]["noul"] == pytest.approx(0.8)  # P(yes): the head read (yes, no)
    dbg = out["decisio_debug"]["q"]
    assert dbg["path"] == "head" and np.allclose(np.exp(dbg["hidden_lp"]), [0.8, 0.2])
    out = s1.answer(SystemOneRequest.model_validate(body), debug="hidden")
    assert np.allclose(np.exp(out["decisio_debug"]["q"]["hidden_lp"]), [0.8, 0.2])
