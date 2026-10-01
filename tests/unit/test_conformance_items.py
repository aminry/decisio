# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The conformance items generator (benchmarks/make_conformance_items.py), offline, on a few made-up rows.

I1  every item carries both request forms, the systemone form validates against the served schema, and its question
    renders to the same options, in the same order, as the /v1/answer form (the condition C3 checks with a tokenizer)
I2  the draw is deterministic: the same rows give the same items, in the same order
"""

import importlib.util
from pathlib import Path

from decisio.serve.systemone import SystemOneRequest, to_engine_question

SPEC = importlib.util.spec_from_file_location(
    "make_conformance_items", Path(__file__).resolve().parents[2] / "benchmarks" / "make_conformance_items.py"
)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)

BOOLQ = [{"question": f"is {i} even", "passage": f"The number {i}.", "answer": i % 2 == 0} for i in range(12)]
CATEGORIES = ["card_arrival", "exchange_rate", "card_payment_wrong_exchange_rate", "age_limit"]
BANKING = [(f"message {i}", CATEGORIES[i % 4]) for i in range(12)]


def test_i1_both_forms_agree():
    items = gen.build(BOOLQ, BANKING, 5, 5, "m")
    assert len(items) == 10 and {i["task"] for i in items} == {"boolq", "banking77"}
    for it in items:
        req = SystemOneRequest.model_validate(it["systemone"])
        eq, keys = to_engine_question(req.questions["q"])
        q = it["answer"]["questions"][0]
        if q["kind"] == "choice":
            assert eq["options"] == q["options"]
            assert keys == list(it["systemone"]["questions"]["q"]["criteria"])
            assert q["options"][it["label"]] == it["answer"]["questions"][0]["options"][it["label"]]
        else:
            assert it["label"] in (0, 1) and eq["instructions"] == q["instructions"]
        assert req.state == it["answer"]["state"]


def test_i2_the_draw_is_deterministic():
    a, b = gen.build(BOOLQ, BANKING, 5, 5, "m"), gen.build(BOOLQ, BANKING, 5, 5, "m")
    assert a == b and [i["i"] for i in a if i["task"] == "boolq"] == sorted(i["i"] for i in a if i["task"] == "boolq")
