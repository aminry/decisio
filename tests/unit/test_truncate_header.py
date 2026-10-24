# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Truncating a state under the header layout, with Gemma 4's real tokenizer (skipped where it is not cached).

The 0.26.0 failure (RLCD experiments/2026-10-23_lab2_r3ch_session, serve_422_evidence): on a 65,536 ceiling, a state of
83,000 tokens cut to 65,379 and a row of 65,538 tokens, the engine's 422. The same question (a yes/no with a 34-token
header) is checked here at a small ceiling, where the header is the same size; `test_over_limit.py` O8 checks the rule
with a stand-in tokenizer.

  uv run pytest -q tests/unit/test_truncate_header.py
"""

import pytest

from decisio.readout.letters import PromptFormat
from decisio.serve import vllm_engine
from decisio.serve.systemone import SystemOne, SystemOneRequest
from decisio.serve.tasks import TaskStore

QUESTION = {
    "type": "noul",
    "instructions": (
        "Does the abstract support or refute the claim (i.e. is it relevant evidence for judging the claim)?"
    ),
}
SENTENCE = "The trial enrolled adults with moderate disease and followed them for twelve weeks. "


def gemma_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it", local_files_only=True)
    except Exception as e:  # not in the local cache
        pytest.skip(f"the Gemma 4 tokenizer is not cached: {e}")


class Stub:
    """The parts of the letters engine that build a request's rows."""

    over_limit_tokens = None
    pad_unit = 0

    def __init__(self, tok, fmt, limit):
        self.tok, self.fmt, self._limit = tok, fmt, limit

    def prompt_limit(self):
        return self._limit


for _name in ("_build_rows", "_template_tail", "_labels"):
    setattr(Stub, _name, getattr(vllm_engine.LettersEngine, _name))


@pytest.mark.parametrize("layout", ["plain", "header"])
@pytest.mark.parametrize("context", [2048, 4096])
def test_t1_every_row_of_a_truncated_request_fits_the_context_under_either_layout(layout, context):
    tok = gemma_tokenizer()
    fmt = PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout=layout)
    eng = Stub(tok, fmt, context - 1)  # the server's ceiling is one under the context
    so = SystemOne(eng, "x", task_store=TaskStore("fp"), noul_rendering="letters", over_limit="truncate")
    for repeats in (context // 10, context // 12, context // 6, context):  # a state under, around and over the ceiling
        req = SystemOneRequest.model_validate(
            {"state": {"claim": "x", "abstract": SENTENCE * repeats}, "model": "m", "questions": {"q": QUESTION}}
        )
        qs = [so.engine_question(q)[0] for q in req.questions.values()]
        cut, report = so.truncated_request(req, qs, eng)
        rows, _, _ = eng._build_rows(cut.state if report else req.state, qs)
        assert max(len(ids) for ids, _ in rows) <= context, (layout, context, repeats, report)
