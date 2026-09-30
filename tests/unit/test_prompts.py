# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The letters engine's prompt construction, with the served model's tokenizer and no model:
  C1  separate-mode prompts equal the letters prompt in chat format, with the right label tokens;
  C2  the state split depends on the state alone, and padding inserts exactly enough tokens at the chosen place;
  C3  a packed prompt's first turn equals its separate prompt, and every read position is the final ':' of "Answer:".
The GPU tier's serving gates (tests/gpu/serving_gates.py) import these checks and their state and questions from here.

    uv run pytest -q tests/unit/test_prompts.py        (downloads the tokenizer of Qwen/Qwen3.6-35B-A3B once)
"""

import os

from decisio.readout.letters import chat_wrap, fmt_state, letters_prompt
from decisio.serve import vllm_engine as sv

TOKENIZER = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")

STATE = (
    "Ticket 8842. Customer writes: our API integration started returning 500 errors on every "
    "request about 20 minutes ago, and we cannot process customer orders until this is fixed. "
    "Account tier: enterprise. Previous tickets this month: 3. Last deploy: 40 minutes ago."
)
QUESTIONS = [
    {
        "kind": "choice",
        "instructions": "Which team should handle this?",
        "options": ["billing", "technical", "sales", "account"],
    },
    {"kind": "noul", "instructions": "Is the customer reporting an outage?"},
    {"kind": "score", "instructions": "How urgent is this?", "options": ["low", "medium", "high", "critical"]},
    {"kind": "noul", "instructions": "Is this urgent?"},
]


class _Tokonly(sv.LettersEngine):
    """The engine's prompt logic with a tokenizer and no model."""

    def __init__(self, tok, pad_unit=None, pad_where="between"):
        self.tok, self.pad_unit, self.pad_token, self.pad_where = tok, pad_unit, sv.PAD_TOKEN, pad_where


def cpu_gates(model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    ok = True

    eng = _Tokonly(tok)
    rows, P = eng._prepare_separate(STATE, QUESTIONS)
    for q, (ids, lab) in zip(QUESTIONS, rows):
        kind = q["kind"]
        opts = ["yes", "no"] if kind == "noul" else q["options"]
        prompt, cands = letters_prompt(tok, kind, fmt_state(STATE), q["instructions"], opts)
        ref = enc(chat_wrap(tok, prompt, "chat"))
        ok &= _check(f"C1 {kind:6s} prompt identical to eval", ids == ref)
        ok &= _check(f"C1 {kind:6s} labels", lab == [enc(c)[0] for c in cands])
    n_state = len(enc("<|im_start|>user\n" + STATE))
    ok &= _check(
        f"C2 split {P} just past the state's blank line (state alone encodes to {n_state})",
        P in (n_state, n_state + 1) and tok.decode(rows[0][0][:P]).endswith(STATE + "\n\n"),
    )
    other, P2 = eng._prepare_separate(STATE, QUESTIONS[::-1][:2])
    ok &= _check("C2 split depends on the state alone, not the question mix", P2 == P)

    for unit in (16, 544, 1056):
        padded, Pp = _Tokonly(tok, unit)._prepare_separate(STATE, QUESTIONS)
        k = Pp - P
        same = all(
            pi[:P] == ri[:P] and pi[P:Pp] == [sv.PAD_TOKEN] * k and pi[Pp:] == ri[P:]
            for (pi, _), (ri, _) in zip(padded, rows)
        )
        ok &= _check(
            f"C2 pad to {unit}: prefix {Pp} = multiple, {k} pads at the split, rest unchanged",
            Pp % unit == 0 and 0 <= k < unit and same,
        )

    head = len(enc(sv.USER_HEAD))
    for where, at in (("front", 0), ("user", head)):
        padded, Pp = _Tokonly(tok, 1056, where)._prepare_separate(STATE, QUESTIONS)
        k = Pp - P
        same = all(
            pi[at : at + k] == [sv.PAD_TOKEN] * k and pi[:at] + pi[at + k :] == ri
            for (pi, _), (ri, _) in zip(padded, rows)
        )
        shared = all(pi[:Pp] == padded[0][0][:Pp] for pi, _ in padded)
        ok &= _check(
            f"C2 pad {where}: prefix {Pp} = 1056 multiple, {k} pads at position {at}, prompt otherwise "
            f"unchanged, prefix shared",
            Pp % 1056 == 0 and same and shared,
        )

    turns = [(f"{STATE}\n\n{sv.question_text(tok, QUESTIONS[0])[0]}", rows[0][1])] + [
        (sv.question_text(tok, q)[0], lab) for q, (_, lab) in zip(QUESTIONS[1:], rows[1:])
    ]
    ids, pos = [], []
    for j, (content, _) in enumerate(turns):
        ids += enc((sv.TURN_SEP if j else "") + sv.user_turn(tok, content))
        pos.append(len(ids) - 1)
    whole = enc(sv.user_turn(tok, turns[0][0]) + "".join(sv.TURN_SEP + sv.user_turn(tok, c) for c, _ in turns[1:]))
    ok &= _check("C3 pack: piecewise encoding = whole-string encoding", ids == whole)
    ok &= _check("C3 pack: first turn = separate prompt", ids[: pos[0] + 1] == rows[0][0])
    colon = enc("Answer:")[-1]
    ok &= _check("C3 pack: every read position is the final ':' of 'Answer:'", all(ids[p] == colon for p in pos))
    print("CPU GATES", "PASS" if ok else "FAIL")
    return ok


def _check(name, cond):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")
    return bool(cond)


def test_c1_c2_c3_prompt_construction(capsys):
    ok = cpu_gates(TOKENIZER)
    assert ok, capsys.readouterr().out
