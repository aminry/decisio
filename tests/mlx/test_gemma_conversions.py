# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The MLX engine on Gemma 4 12B: the mlx-community 4-bit and 6-bit conversions, loaded from the local Hugging Face
cache at pinned revisions (skipped where they are not there; nothing is downloaded). What Gemma adds to the engine:

  - the family's official tokenizer (google/gemma-4-12B-it at 707f0a3b), not the conversion's;
  - the sliding-window layers' cache (40 of 48 layers, a 1,024-token window), copied for every question;
  - the final-logit softcap (30) on the label logits, and so on the intent head's base readout.

The prompts are main's served rows on Gemma's tokenizer; the Gemma prompt format and flags come with the model family.

    uv run pytest -q -m mlx_model tests/mlx/test_gemma_conversions.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))

pytestmark = pytest.mark.mlx_model

CONVERSIONS = {
    4: ("mlx-community/gemma-4-12B-it-4bit", "73bcf09092aa277861d5a191b989b666f7f32e8f"),
    6: ("mlx-community/gemma-4-12B-it-6bit", "2fe53aeb9b8686eefa543a60d6aa364c5b219aac"),
}
STATE = (
    "Order 4471 shipped on Monday. The customer writes: 'My parcel arrived with the box crushed and the lamp inside "
    "broken.'"
)
QUESTION = {
    "kind": "choice",
    "instructions": "What does the customer want?",
    "options": ["track_order", "report_damage", "cancel_order", "change_address"],
}
# a state past the sliding window (1,024 tokens), so the prefix's sliding layers have rotated before the question
LONG_STATE = " ".join(f"Line {i}: the order was scanned at depot {i % 7}." for i in range(160)) + " " + STATE
# the capped logits recomputed from the last position alone against the model's own: the matrix-vector kernel rounds
# differently from the model's matrix-matrix one, by 1 to 3.5 bf16 steps near 30 (0.125 each); measured 0.125 to 0.44
BF16_STEPS = 0.5


@pytest.fixture(scope="module", params=sorted(CONVERSIONS))
def engine(request):
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from huggingface_hub import snapshot_download

    from decisio.serve.mlx_engine import MLXLettersEngine

    repo, rev = CONVERSIONS[request.param]
    try:
        path = snapshot_download(repo, revision=rev, local_files_only=True)
        snapshot_download(
            "google/gemma-4-12B-it",
            revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            allow_patterns=["*.json", "*.jinja"],
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001
        pytest.skip(f"{repo}@{rev[:8]} or the official tokenizer is not in the local Hugging Face cache")
    eng = MLXLettersEngine(path, pad_to=None, prefix_cache_mb=0, warm_up=False)
    eng.pad_policy = "none"
    return eng


def _whole(engine, ids, lab):
    """The label log-probabilities with the whole prompt in one pass from an empty cache (never served)."""
    import mlx.core as mx

    from decisio.readout.letters import allowed_ids

    h = engine._feed(ids, engine.model.make_cache())
    z = engine._head(h[None])[0][mx.array(allowed_ids(lab))]
    z = np.array((mx.tanh(z / engine.softcap) * engine.softcap).astype(mx.float32), dtype=np.float64)
    z -= z.max()
    return z - np.log(np.exp(z).sum())


def test_family_tokenizer_and_softcap(engine):
    f = engine.facts()
    assert f["tokenizer"] == "google/gemma-4-12B-it"
    assert f["tokenizer_revision"] == "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    assert f["official_tokenizer"] and f["final_logit_softcap"] == 30.0
    kinds = [type(c).__name__ for c in engine.model.make_cache()]
    assert kinds.count("RotatingKVCache") == 40 and kinds.count("KVCache") == 8


@pytest.mark.parametrize("state", [STATE, LONG_STATE], ids=["short", "past_the_window"])
def test_prefix_path_equals_the_whole_prompt(engine, state):
    rows, P = engine._prepare_separate(state, [QUESTION])
    assert P > 0 and (state is STATE or P > 1024)
    ((lp, _),) = engine._score(rows, P)[0]
    whole = _whole(engine, rows[0][0], rows[0][1])
    assert np.argmax(lp) == np.argmax(whole) == QUESTION["options"].index("report_damage")
    # the prefix path and one pass differ by bf16 chunking only (G1 of tests/mlx/serving_gates.py, as for Qwen)
    assert np.abs(np.exp(lp) - np.exp(whole)).max() < 1e-2


def test_capped_last_position_logits_match_the_model(engine):
    import mlx.core as mx

    rows, _ = engine._prepare_separate(STATE, [QUESTION])
    x = mx.array([rows[0][0]])
    h = engine._feed(rows[0][0], engine.model.make_cache())
    capped = mx.tanh(engine._head(h[None])[0] / engine.softcap) * engine.softcap
    own = engine.model(x, cache=engine.model.make_cache())[0, -1]
    mx.eval(capped, own)
    assert float(mx.abs(capped.astype(mx.float32) - own.astype(mx.float32)).max()) <= BF16_STEPS
    raw = engine._head(h[None])[0]
    assert float(mx.abs(raw.astype(mx.float32) - own.astype(mx.float32)).max()) > 1.0  # the cap is not a no-op


def test_the_head_reads_the_capped_readout(engine):
    from decisio.serve.mlx_engine import MLXHiddenReadout

    ((lp, h),) = MLXHiddenReadout(engine).readout(STATE, [QUESTION])
    p = engine.answer(STATE, [QUESTION])[0][0]
    assert np.array_equal(np.exp(lp) / np.exp(lp).sum(), p) and h.shape == (3840,) and h.dtype == np.float32


def test_answers_are_isolated_and_repeat_bit_for_bit(engine):
    other = {"kind": "noul", "instructions": "Is the customer likely to ask for a refund or replacement?"}
    alone = engine.answer(LONG_STATE, [QUESTION])[0][0]
    together = engine.answer(LONG_STATE, [other, QUESTION])[0][1]
    again = engine.answer(LONG_STATE, [QUESTION])[0][0]
    assert np.array_equal(alone, together) and np.array_equal(alone, again)
