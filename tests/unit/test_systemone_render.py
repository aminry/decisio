# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The served renderer's index-key rule (`decisio.serve.systemone.index_keys`,
`to_engine_question(..., hide_index_keys)`).

  R1  the Decision Index's intent format (keys `option_0` ... `option_N`, intent names as descriptions) shows the names
      alone, and the answer keys stay the wire keys; with the flag off, `key: description` as before;
  R2  keys that carry meaning are kept: not exactly 0..n-1 or 1..n, mixed stems, identifiers with letters after the
      number, a missing description, a single option;
  R4  de-snaking, label by label: a bare snake_case label (the Decision Index's BANKING77 names behind hidden index
      keys, or snake_case keys without descriptions) is shown with spaces; capitalised, numbered, punctuated or natural
      labels, and every `key: description` rendering, are left as they are; the answer keys never change.

    uv run pytest -q tests/unit/test_systemone_render.py
"""

import pytest

from decisio.serve.systemone import SystemOneRequest, index_keys, snake_label, to_engine_question  # noqa: E402

NAMES = ["card_arrival", "card_linking", "exchange_rate", "card_payment_wrong_exchange_rate"]


def question(criteria, instructions="Classify the banking intent of this user request:\nHow do I locate my card?"):
    req = SystemOneRequest.model_validate(
        {
            "state": {},
            "model": "m",
            "questions": {"q1": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        }
    )
    return req.questions["q1"]


def test_r1_decision_index_format():
    q = question({f"option_{i}": n for i, n in enumerate(NAMES)})
    eq, keys = to_engine_question(q, desnake_labels=False)
    assert eq["options"] == NAMES and keys == [f"option_{i}" for i in range(len(NAMES))]
    eq_both, keys_both = to_engine_question(q)  # the served default: both fixes
    assert eq_both["options"] == [n.replace("_", " ") for n in NAMES] and keys_both == keys
    eq_off, keys_off = to_engine_question(q, hide_index_keys=False)
    assert eq_off["options"] == [f"option_{i}: {n}" for i, n in enumerate(NAMES)] and keys_off == keys
    assert eq["instructions"] == eq_off["instructions"]


@pytest.mark.parametrize(
    "keys",
    [
        ["0", "1", "2"],
        ["1", "2", "3"],
        ["Option 1", "Option 2"],
        ["opt-0", "opt-1"],
        ["choice1", "choice2", "choice3"],
        ["label_0", "label_1"],
        ["option_2", "option_0", "option_1"],
    ],
)  # shuffled: still only an index
def test_r1_enumerations_are_hidden(keys):
    assert index_keys({k: f"text {i}" for i, k in enumerate(keys)})


@pytest.mark.parametrize(
    "criteria",
    [
        {"sku_12": "red mug", "sku_40": "blue mug"},  # identifiers, not a run from 0 or 1
        {"option_0": "a", "option_2": "b"},  # a gap
        {"option_0": "a", "option_1": "b", "option_1 ": "c"},  # a repeated number (a trailing space is not a key)
        {"option_0": "a", "option_1": "b", "option_3": "c"},  # not a full enumeration
        {"a0": "x", "b1": "y"},  # two stems
        {"2xl": "size", "3xl": "size"},  # letters after the number
        {"option_0": "a", "option_1": None},  # nothing to show for one option
        {"option_0": "a", "option_1": "  "},
        {"option_0": "only"},  # one option
        {"positive": None, "negative": None},  # ordinary label keys
        {"room 101": "north wing", "room 102": "south wing"},  # starts at 101
    ],
)
def test_r2_meaningful_keys_are_kept(criteria):
    assert not index_keys(criteria)
    q = question(criteria)
    assert to_engine_question(q, desnake_labels=False) == to_engine_question(
        q, hide_index_keys=False, desnake_labels=False
    )


@pytest.mark.parametrize(
    "label,snake",
    [
        ("card_arrival", True),
        ("card_payment_wrong_exchange_rate", True),
        ("balance", False),  # one word: nothing to change
        ("card arrival", False),  # already words
        ("Refund_not_showing_up", False),  # a capital (BANKING77's own label)
        ("reverted_card_payment?", False),  # punctuation (likewise)
        ("plan_2024", False),  # digits
        ("_leading", False),
        ("trailing_", False),
        ("double__underscore", False),
    ],
)
def test_r4_snake_rule(label, snake):
    assert snake_label(label) == snake


def test_r4_desnake_rendering():
    # snake_case keys without descriptions are bare labels too
    q = question({"billing_issue": None, "tech_support": None, "other": None, "Refund_not_showing_up": None})
    eq, keys = to_engine_question(q)
    assert eq["options"] == ["billing issue", "tech support", "other", "Refund_not_showing_up"]
    assert keys == ["billing_issue", "tech_support", "other", "Refund_not_showing_up"]
    assert to_engine_question(q, desnake_labels=False)[0]["options"] == keys
    # `key: description` renderings are never touched, even with snake_case descriptions
    q = question({"a": "card_arrival", "b": "card_linking"})
    assert to_engine_question(q)[0]["options"] == ["a: card_arrival", "b: card_linking"]
    # index keys shown (flag off) keep their snake_case descriptions
    q = question({f"option_{i}": n for i, n in enumerate(NAMES)})
    assert to_engine_question(q, hide_index_keys=False)[0]["options"][0] == "option_0: card_arrival"


def test_r5_describe_options_rule():
    """--describe-options: word keys with descriptions show the description alone, keys stay the answer's keys."""
    crit = {
        "CLICK": "Click the element named in the target",
        "TYPE_TEXT": "Type text into the target field",
        "DONE": "The goal is complete; stop",
        "BLOCKED": None,
    }
    q = question(crit)
    off, keys = to_engine_question(q)
    assert off["options"] == [
        "CLICK: Click the element named in the target",
        "TYPE_TEXT: Type text into the target field",
        "DONE: The goal is complete; stop",
        "BLOCKED",
    ]
    on, keys_on = to_engine_question(q, describe_options=True)
    assert on["options"] == [
        "Click the element named in the target",
        "Type text into the target field",
        "The goal is complete; stop",
        "BLOCKED",
    ]
    assert keys_on == keys == list(crit)
    # a snake_case key without a description is de-snaked as a bare label is
    on2, _ = to_engine_question(question({"card_arrival": None, "refund": "Money back"}), describe_options=True)
    assert on2["options"] == ["card arrival", "Money back"]
    # two options that would read the same: the question is rendered as without the rule
    dup = question({"a": "Same text", "b": "Same text"})
    assert to_engine_question(dup, describe_options=True) == to_engine_question(dup)
    # index keys, keys without descriptions, yes/no and score questions: unchanged by the rule
    for c in ({f"option_{i}": n for i, n in enumerate(NAMES)}, {"billing": None, "access": None}):
        assert to_engine_question(question(c), describe_options=True) == to_engine_question(question(c))
