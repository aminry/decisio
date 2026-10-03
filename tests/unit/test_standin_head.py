# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The CPU stand-in's head engine (`decisio.serve.hidden_engine.HFHiddenEngine`) builds the served rows with the text
engine's own row builder, so it must carry every attribute that builder reads. Since `--pad-policy` (#35) it reads
`pad_policy`, which the stand-in lacked: registering a task with a head on `--backend hf` failed with HTTP 500.

    uv run pytest -q tests/unit/test_standin_head.py      (downloads the tokenizer of Qwen/Qwen3.6-35B-A3B once)
"""

import os
import threading

from decisio.serve import vllm_engine as sv
from decisio.serve.hidden_engine import HFHiddenEngine, HFReservedHiddenEngine

TOKENIZER = os.environ.get("DECISIO_TOKENIZER", "Qwen/Qwen3.6-35B-A3B")
STATE = "Ticket 8842. The customer cannot log in since this morning and asks for a password reset."
QUESTIONS = [
    {"kind": "choice", "instructions": "Which team?", "options": ["billing", "access", "sales"]},
    {"kind": "noul", "instructions": "Is this urgent?"},
]


def standin(cls, tok, pad_to):
    """The stand-in's attributes without loading a model (its __init__ sets exactly these besides the model)."""
    eng = cls.__new__(cls)
    eng.mode, eng.pad_token, eng.pad_where, eng.adapters, eng.tok = "hidden", sv.PAD_TOKEN, "front", {}, tok
    eng.block_size = eng.match_unit = 64
    eng.pad_unit = None if not pad_to else eng.block_size
    eng._lock = threading.Lock()
    return eng


class Text(sv.LettersEngine):
    def __init__(self, tok, pad_unit):
        self.tok, self.pad_unit, self.pad_token, self.pad_where = tok, pad_unit, sv.PAD_TOKEN, "front"
        self.block_size = self.match_unit = 64


def test_standin_head_builds_the_served_rows():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    for cls in (HFHiddenEngine, HFReservedHiddenEngine):
        for pad_to in (None, "block"):
            assert standin(cls, tok, pad_to).pad_policy == sv.LettersEngine.pad_policy
            got = standin(cls, tok, pad_to)._prepare_separate(STATE, QUESTIONS)
            assert got == Text(tok, 64 if pad_to else None)._prepare_separate(STATE, QUESTIONS)
