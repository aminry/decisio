# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Box questions through the served route (`--images on`, decisio.serve.gemma_images) on the CPU stand-in.

The engine is the stand-in with a tiny random Gemma 4 and the real tokenizer; its generation is scripted (digit
distributions per step), so the checks are about the wire, the prompt, the batch, the presence read, the confidence,
the usage and the refusals, not about any model's boxes.

  X1  the wire to the answer: a scripted reply becomes a box in the image's fractions, rounded, in the request's order,
      beside the other questions, which are answered as without the Box
  X2  the engine gets one batch for all the Boxes, each row the shared prefix plus the question's own text and the
      forced reply prefix, with the question's own images; the presence read reaches the engine on the same prefix
  X3  confidence is the presence probability times the raw edge confidence until a map is loaded, then the map's
  X4  usage and timing: generated tokens are output tokens; flags, tokens and the uncalibrated mark are reported
  X5  the rules: each is a 400 `api_usage_error` that says what to change
  X6  an unreadable reply is a 502 naming the question; no box is made up
  X7  an adapter reaches both reads; the same request twice gives the same answer
  X8  (needs DECISIO_BOX_REAL_MODEL, a local Gemma 4 checkpoint) squares in the four corners are found, IoU >= 0.5
  X9  branch expansion: a real decode of a likely alternative off the greedy path moves the edge past the plain
      tail approximation; usage counts the branch's tokens; a cap on branches is flagged
  X10 the exact-tree debug readout: refused without --debug-box-tree; with it, each edge carries the exact value,
      the served (fast) estimator's and the plain greedy one, and the tree mass covered

  uv run pytest -q tests/unit/test_box_serving.py
"""

import base64
import io
import math
import os

import numpy as np
import pytest
from PIL import Image

from decisio.families import GEMMA4
from decisio.readout import box as B
from decisio.readout.letters import PromptFormat
from decisio.serve import image_marks as marks
from decisio.serve import vllm_engine as sv

PLAIN = PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout="plain")
HEADER = PromptFormat(tail="spaced", slot="template", variants="summed", system_prompt=True, layout="header")


def asset(w=96, h=96, seed=0):
    rng = np.random.default_rng(seed)
    b = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB").save(b, "PNG")
    return {"type": "image", "format": "png", "data_base64": base64.b64encode(b.getvalue()).decode()}


def sure(tok):
    return (tok, [(tok, 0.0)])


def script(nums, tail=True):
    """The steps of a reply `y_min, x_min, y_max, x_max` (digits one per step, a space token after each comma)."""
    steps = []
    for i, n in enumerate(nums):
        if i:
            steps.append(sure("▁"))
        steps += [sure(d) for d in str(n)]
        steps.append(sure("]," if i == 3 else ","))
    if tail:
        steps += [sure("▁"), sure('"'), sure("label"), sure('":')]
    return steps


def make_engine(fmt=PLAIN, scripted=True):
    from decisio.serve.gemma_images import BOX_TOP_LOGPROBS
    from decisio.serve.hf_letters import HFGemmaImageLettersEngine

    class Scripted(HFGemmaImageLettersEngine):
        reply = None  # callable(index, row ids) -> steps: the main forced reply
        branch_reply = None  # callable(index, row ids) -> steps: a branch's continuation (top=1)
        tree_reply = None  # callable(index, row ids) -> steps: an exact-tree node decode (top=BOX_TREE_TOP_LOGPROBS)

        def generate_rows(self, rows, adapter=None, **kw):
            if not scripted:
                return super().generate_rows(rows, adapter, **kw)
            self.box_calls.append({"rows": [list(r) for r in rows], "adapter": adapter, **kw})
            top = kw.get("top")
            if top == 1 and self.branch_reply is not None:
                source = self.branch_reply
            elif top not in (None, 1, BOX_TOP_LOGPROBS) and self.tree_reply is not None:
                source = self.tree_reply
            else:
                source = self.reply
            gens = [source(i, r) for i, r in enumerate(rows)]
            return gens, {
                "generate_ms": 2.0,
                "generated_tokens": [len(g) for g in gens],
                "cached_tokens": [0] * len(gens),
                "engine_prompt_tokens": [len(r) for r in rows],
            }

    e = Scripted(GEMMA4.model, kind="unified", budget=70, max_images=4, revision=GEMMA4.revision)
    e.fmt, e.family = fmt, GEMMA4
    e.box_calls = []
    e.reply = lambda i, r: script([100, 200, 500, 800])
    return e


@pytest.fixture(scope="module")
def engine():
    return make_engine()


@pytest.fixture(autouse=True)
def _reset(engine):
    engine.box_calls.clear()
    engine.scored_rows.clear()
    engine.adapter_calls.clear()
    engine.reply = lambda i, r: script([100, 200, 500, 800])
    engine.branch_reply = None
    engine.tree_reply = None
    engine.fmt = PLAIN


def make_client(engine, **kw):
    from fastapi.testclient import TestClient

    from decisio.serve.systemone import SystemOne

    so = SystemOne(engine, "decisio-test", **kw)
    return TestClient(sv.make_app(engine, so), headers={"x-decisio-client": "test"}), so


def body(**kw):
    out = {
        "state": {"listing": "ITEM", "title": "Blue ceramic mug, 350 ml"},
        "questions": {
            "where": {"type": "box", "instructions": "the mug"},
            "colour": {
                "type": "choice",
                "instructions": "Which colour is the mug?",
                "criteria": {"blue": None, "red": None},
            },
        },
        "assets": {"ITEM": asset(200, 150, 1)},
    }
    out.update(kw)
    return out


def text_of(engine, ids):
    return engine.tok.decode(ids)


# ------------------------------------------------------------------------------------------------------ X1


def test_x1_a_scripted_reply_is_a_box_in_request_order_beside_other_questions(engine):
    client, _ = make_client(engine)
    r = client.post("/v1/systemone", json=body())
    assert r.status_code == 200, r.text
    ans = r.json()["answers"]
    assert list(ans) == ["where", "colour"]
    assert ans["where"]["type"] == "box"
    assert ans["where"]["box"] == {"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.8}
    assert 0.0 < ans["where"]["confidence"] <= 1.0
    assert ans["colour"]["type"] == "choice"
    assert r.headers["x-decisio-box"] == "calibrated=false; tokens=%d; flags=none" % len(script([100, 200, 500, 800]))
    assert "box=2.0" in r.headers["x-decisio-stages"]
    # the other questions are read as without the Box
    plain = body()
    del plain["questions"]["where"]
    r2 = client.post("/v1/systemone", json=plain)
    assert r2.json()["answers"]["colour"] == ans["colour"]


def test_x1_two_boxes_one_batch_and_a_mixed_order(engine):
    engine.reply = lambda i, r: script([[100, 200, 500, 800], [0, 0, 1000, 1000]][i])
    client, _ = make_client(engine)
    b = body()
    b["questions"] = {
        "a": {"type": "box", "instructions": "the mug"},
        "n": {"type": "noul", "instructions": "Is it blue?"},
        "b": {"type": "box", "instructions": "the saucer"},
    }
    r = client.post("/v1/systemone", json=b)
    assert r.status_code == 200, r.text
    ans = r.json()["answers"]
    assert list(ans) == ["a", "n", "b"]
    assert ans["b"]["box"] == {"top": 0.0, "bottom": 1.0, "left": 0.0, "right": 1.0}
    assert len(engine.box_calls) == 1 and len(engine.box_calls[0]["rows"]) == 2


# ------------------------------------------------------------------------------------------------------ X2


def test_x2_rows_share_the_prefix_and_carry_the_forced_reply(engine):
    client, _ = make_client(engine)
    b = body()
    b["questions"]["where"]["instructions"] = "the mug in QPIC"
    b["assets"]["QPIC"] = asset(64, 64, 2)
    r = client.post("/v1/systemone", json=b)
    assert r.status_code == 200, r.text
    (call,) = engine.box_calls
    (row,) = call["rows"]
    text = text_of(engine, row)
    assert text.endswith(B.BOX_REPLY_PREFIX)
    assert "Detect the mug in" in text and "Output only the JSON." in text
    assert "<|turn>model" in text and "Answer:" not in text
    assert row.count(engine.image_token_id) == 2  # the state's image and the question's own
    assert call["max_tokens"] == 32 and call["top"] == 20
    assert [m["image"] and len(m["image"]) for m in call["row_mm"]] == [2]
    # the presence read is a yes/no row on the same prefix
    yn = [ids for ids, _ in engine.scored_rows if "Is this in the picture: the mug in" in text_of(engine, ids)]
    assert len(yn) == 1
    n = 0
    while row[n] == yn[0][n]:
        n += 1
    assert 100 <= n < len(row)  # the state is shared, the question's own text differs


def test_x2_the_presence_read_follows_the_instructions(engine):
    client, _ = make_client(engine)
    b = body()
    b["questions"]["where"]["instructions"] = {"object": "mug"}
    client.post("/v1/systemone", json=b)
    assert any('Is this in the picture: {"object":"mug"}?' in text_of(engine, ids) for ids, _ in engine.scored_rows)


# ------------------------------------------------------------------------------------------------------ X3, X4


def request_for(engine, nums=None):
    from decisio.serve.systemone import SystemOneRequest

    tag = "0123456789abcdef"
    return SystemOneRequest(
        state="Photo:" + marks.mark(1, tag) + "Blue mug",
        questions={"where": {"type": "box", "instructions": "the mug"}, "n": {"type": "noul", "instructions": "Blue?"}},
    ), tag


def test_x3_confidence_is_presence_times_raw_until_a_map_is_loaded(engine):
    img = Image.new("RGB", (96, 96), "white")
    from decisio.serve.systemone import SystemOne

    req, tag = request_for(engine)
    # a reply whose y_min is 100 with probability 0.8 and 150 with 0.2 lowers the raw confidence below 1
    split = script([100, 200, 500, 800])
    split[1] = ("0", [("0", math.log(0.8)), ("5", math.log(0.2))])
    engine.reply = lambda i, r: split
    so = SystemOne(engine, "decisio-test")
    out = so.answer(req, None, [img], image_tag=tag, debug="readout")
    d = out["_timing"]["box"]["questions"]["where"]
    assert d["calibrated"] is False
    assert d["raw_confidence"] == pytest.approx(0.8**0.25, abs=1e-6)
    assert out["answers"]["where"]["confidence"] == pytest.approx(
        round(d["presence"] * d["raw_confidence"], 12), abs=1e-9
    )

    cmap = B.fit_confidence_map([0.1, 0.2, 0.5, 0.9], [0, 0, 1, 1])
    so2 = SystemOne(engine, "decisio-test", box_confidence=cmap)
    out2 = so2.answer(req, None, [img], image_tag=tag, debug="readout")
    d2 = out2["_timing"]["box"]["questions"]["where"]
    assert d2["calibrated"] is True
    assert out2["answers"]["where"]["confidence"] == pytest.approx(cmap.apply(d2["presence"] * d2["raw_confidence"]))


def test_x4_usage_and_timing(engine):
    from decisio.serve.systemone import SystemOne

    img = Image.new("RGB", (96, 96), "white")
    req, tag = request_for(engine)
    steps = script([100, 200, 500, 800])
    engine.reply = lambda i, r: steps
    so = SystemOne(engine, "decisio-test")
    out = so.answer(req, None, [img], image_tag=tag)
    assert out["usage"]["output_tokens"] == 1 + len(steps)  # the noul's one and the Box's written tokens
    t = out["_timing"]
    assert t["box"]["generate_ms"] == 2.0 and t["stages"]["box"] == 2.0
    q = t["box"]["questions"]["where"]
    assert q["tokens"] == len(steps) and q["flags"] == [] and q["calibrated"] is False
    assert "presence" not in q  # debug only
    nobox = SystemOne(engine, "decisio-test").answer(
        req.model_copy(update={"questions": {"n": req.questions["n"]}}), None, [img], image_tag=tag
    )
    assert out["usage"]["input_tokens"] > nobox["usage"]["input_tokens"]


def test_x4_flags_of_a_swapped_box(engine):
    from decisio.serve.systemone import SystemOne

    req, tag = request_for(engine)
    engine.reply = lambda i, r: script([500, 800, 100, 200])
    out = SystemOne(engine, "decisio-test").answer(req, None, [Image.new("RGB", (96, 96))], image_tag=tag)
    assert out["_timing"]["box"]["questions"]["where"]["flags"] == ["swapped_y", "swapped_x"]
    assert out["answers"]["where"]["box"] == {"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.8}


# ------------------------------------------------------------------------------------------------------ X5


def refused(client, b, needle):
    r = client.post("/v1/systemone", json=b)
    assert r.status_code == 400, r.text
    d = r.json()["detail"]
    assert d["error_type"] == "api_usage_error" and needle in d["message"], d


def test_x5_no_image(engine):
    client, _ = make_client(engine)
    b = body()
    b["state"], b["assets"] = "A mug.", None
    del b["assets"]
    refused(client, b, "needs an image")


def test_x5_two_orders_a_task_id_and_abstention(engine):
    client, _ = make_client(engine)
    refused(client, body(orders=2), "one order")
    refused(client, body(task="t1"), "no registered task")
    from decisio.serve.systemone import BoxRequestError, SystemOne

    req, tag = request_for(engine)
    so = SystemOne(engine, "decisio-test", abstain_option="none of these")
    with pytest.raises(BoxRequestError, match="abstention"):
        so.answer(req, None, [Image.new("RGB", (96, 96))], imajev_ext=True, image_tag=tag)


def header_text():
    from decisio.readout.letters import QUESTIONS_HEAD

    return QUESTIONS_HEAD


def test_x5_a_header_default_server_answers_a_box_without_layout(engine):
    engine.fmt = HEADER
    client, so = make_client(engine)
    assert so.layout == "header"
    only_box = body()
    del only_box["questions"]["colour"]
    r = client.post("/v1/systemone", json=only_box)
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["where"]["type"] == "box"
    assert r.headers["x-decisio-prompt"].startswith("layout=plain;")
    (call,) = engine.box_calls
    assert header_text() not in text_of(engine, call["rows"][0])
    assert engine.scored_rows and all(header_text() not in text_of(engine, ids) for ids, _ in engine.scored_rows)


def test_x5_a_mixed_request_on_a_header_server_reads_each_under_its_layout(engine):
    engine.fmt = HEADER
    client, _ = make_client(engine)
    r = client.post("/v1/systemone", json=body())
    assert r.status_code == 200, r.text
    ans = r.json()["answers"]
    assert list(ans) == ["where", "colour"] and ans["where"]["type"] == "box"
    assert r.headers["x-decisio-prompt"].startswith("layout=header;")  # the decisions' layout
    rows = [text_of(engine, ids) for ids, _ in engine.scored_rows]
    choice = [t for t in rows if "Which colour is the mug?" in t and "Is this in the picture" not in t]
    presence = [t for t in rows if "Is this in the picture: the mug" in t]
    assert choice and all(header_text() in t for t in choice)
    assert presence and all(header_text() not in t for t in presence)
    (call,) = engine.box_calls
    assert header_text() not in text_of(engine, call["rows"][0])
    assert r.json()["usage"]["input_tokens"] > 0


def test_x5_an_explicit_header_layout_with_a_box_is_refused(engine):
    engine.fmt = HEADER
    client, _ = make_client(engine)
    refused(client, body(layout="header"), 'send the request with "layout": "plain" or without "layout"')
    engine.fmt = PLAIN
    refused(client, body(layout="header"), "read under the plain layout")
    assert not engine.box_calls


def test_x5_an_explicit_plain_layout_still_works(engine):
    engine.fmt = HEADER
    client, _ = make_client(engine)
    r = client.post("/v1/systemone", json=body(layout="plain"))
    assert r.status_code == 200, r.text
    assert r.headers["x-decisio-prompt"].startswith("layout=plain;")
    assert all(header_text() not in text_of(engine, ids) for ids, _ in engine.scored_rows)
    assert header_text() not in text_of(engine, engine.box_calls[0]["rows"][0])


def test_x5_servers_that_cannot_answer_a_box(engine):
    from decisio.serve.hf_letters import HFLettersEngine

    # text only: no images on
    text_only = object.__new__(HFLettersEngine)
    text_only.__dict__.update(
        {
            "mode": "separate",
            "adapters": {},
            "tok": engine.tok,
            "model": engine.model,
            "model_name": "x",
            "revision": None,
        }
    )
    text_only.block_size = text_only.match_unit = 64
    text_only.pad_unit, text_only.pad_token, text_only.pad_where = None, engine.pad_token, "front"
    text_only.scored_rows, text_only.fmt, text_only.family = [], PLAIN, GEMMA4
    from decisio.serve.combine import FairLock

    text_only._lock = FairLock()
    client, _ = make_client(text_only)
    refused(client, body(), "--images on")
    # the Qwen image route
    client, _ = make_client(engine, image_engine=object())
    refused(client, body(), "--image-model")


def test_x5_an_engine_without_box_generation(engine, monkeypatch):
    from decisio.serve.gemma_images import GemmaImageRequests

    monkeypatch.delattr(GemmaImageRequests, "box")
    client, _ = make_client(engine)
    refused(client, body(), "does not answer Box")


# ------------------------------------------------------------------------------------------------------ X6


def test_x6_an_unreadable_reply_is_a_502_naming_the_question(engine):
    engine.reply = lambda i, r: [sure("Sorry"), sure(",")]
    client, _ = make_client(engine)
    r = client.post("/v1/systemone", json=body())
    assert r.status_code == 502
    assert 'Box question "where"' in r.json()["detail"]["message"]


# ------------------------------------------------------------------------------------------------------ X7


def test_x7_an_adapter_reaches_both_reads_and_the_same_request_twice_agrees(engine):
    engine.add_adapter("a", seed=3)
    client, _ = make_client(engine)
    b = body(adapter="a")
    r1 = client.post("/v1/systemone", json=b)
    assert r1.status_code == 200, r1.text
    assert engine.box_calls[0]["adapter"] == "a"
    assert engine.adapter_calls == ["a"]  # the presence read (the scripted generation is the call recorded above)
    r2 = client.post("/v1/systemone", json=b)
    assert r1.json()["answers"] == r2.json()["answers"]
    assert r1.json()["adapter"] == "a"


def test_x7_the_stand_in_generation_is_deterministic():
    e = make_engine(scripted=False)
    client, _ = make_client(e)
    b = body()
    first = client.post("/v1/systemone", json=b)
    again = client.post("/v1/systemone", json=b)
    # a random model writes no box: the same refusal, word for word, twice
    assert first.status_code == again.status_code == 502
    assert first.json() == again.json()
    ids = [0] * 5
    a, _ = e.generate_rows([ids], None, max_tokens=3, top=5)
    c, _ = e.generate_rows([ids], None, max_tokens=3, top=5)
    assert a == c and len(a[0]) == 3 and len(a[0][0][1]) == 5


# ------------------------------------------------------------------------------------------------------ X8


@pytest.mark.skipif(not os.environ.get("DECISIO_BOX_REAL_MODEL"), reason="needs DECISIO_BOX_REAL_MODEL (a Gemma 4)")
def test_x8_squares_in_the_four_corners_are_found():
    from decisio.serve.hf_letters import HFGemmaImageLettersEngine
    from decisio.serve.systemone import SystemOne, SystemOneRequest

    model = os.environ["DECISIO_BOX_REAL_MODEL"]
    e = HFGemmaImageLettersEngine(model, model=model, budget=280)
    e.fmt = PLAIN
    so = SystemOne(e, "decisio-box")
    tag = "0123456789abcdef"
    bad = []
    for top in (True, False):
        for left in (True, False):
            img = Image.new("RGB", (480, 480), "white")
            x, y = (40 if left else 320), (40 if top else 320)
            img.paste(Image.new("RGB", (120, 120), "red"), (x, y))
            truth = (y / 480, (y + 120) / 480, x / 480, (x + 120) / 480)  # top, bottom, left, right
            req = SystemOneRequest(
                state=marks.mark(1, tag), questions={"b": {"type": "box", "instructions": "the red square"}}
            )
            got = so.answer(req, None, [img], image_tag=tag)["answers"]["b"]["box"]
            g = (got["top"], got["bottom"], got["left"], got["right"])
            ih = max(0.0, min(g[1], truth[1]) - max(g[0], truth[0]))
            iw = max(0.0, min(g[3], truth[3]) - max(g[2], truth[2]))
            inter = ih * iw
            union = (g[1] - g[0]) * (g[3] - g[2]) + (truth[1] - truth[0]) * (truth[3] - truth[2]) - inter
            if inter / union < 0.5:
                bad.append((top, left, g, truth))
    assert not bad, bad


# ------------------------------------------------------------------------------------------------------ X9


def script_branch_main():
    """y_min is a single digit, "9,", with a 0.1 chance its terminator was really one more digit, "5"; the other
    three coordinates are certain. One branch off the greedy path, worth a real decode (>= 0.05)."""
    steps = [sure("9")]
    steps.append((",", [(",", math.log(0.9)), ("5", math.log(0.1))]))
    for i, n in enumerate([200, 500, 800], start=1):
        steps.append(sure("▁"))
        steps += [sure(d) for d in str(n)]
        steps.append(sure("]," if i == 3 else ","))
    steps += [sure("▁"), sure('"'), sure("label"), sure('":')]
    return steps


def test_x9_a_decoded_branch_moves_the_edge_past_the_tail_approximation(engine):
    engine.reply = lambda i, r: script_branch_main()
    engine.branch_reply = lambda i, r: [sure("0"), sure(",")]  # the branch continues "95" with "0", then ends: 950
    client, _ = make_client(engine)
    r = client.post("/v1/systemone", json=body(), headers={"x-decisio-client": "test"})
    assert r.status_code == 200, r.text

    from decisio.readout import box as B

    pairs, _ = B.parse_coordinates(B._normalise_steps(script_branch_main()), prefix_in_steps=False)
    plain = B.estimate_coordinate(*pairs[0]).expected
    expanded = B.estimate_coordinate(*pairs[0], branch_values={(1, 5): 950.0}).expected
    assert expanded != pytest.approx(plain)

    dbg = (
        make_client(engine, debug_readout=True)[0]
        .post("/v1/systemone", json=body(), headers={"x-decisio-debug": "readout", "x-decisio-client": "test"})
        .json()["decisio_debug"]["where"]["box"]
    )
    assert dbg["edges"]["y_min"]["expected"] == pytest.approx(round(expanded, 3))
    assert dbg["branches_decoded"] == 1 and "branches_capped" not in dbg


def test_x9_usage_counts_the_branch_tokens_and_the_cap_is_flagged(engine):
    engine.reply = lambda i, r: script_branch_main()
    engine.branch_reply = lambda i, r: [sure("0"), sure(",")]
    client, so = make_client(engine)
    from decisio.serve.systemone import SystemOne

    req, tag = request_for(engine)
    out = so.answer(req, None, [Image.new("RGB", (96, 96))], image_tag=tag)
    # the state's "n" and the Box's presence read (1 each), the main reply, and the one branch's 2 tokens
    assert out["usage"]["output_tokens"] == 2 + (len(script_branch_main()) - 1) + 2

    # a tiny cap forces the fallback and flags it
    import decisio.serve.gemma_images as gi

    old = gi.BOX_MAX_BRANCHES
    gi.BOX_MAX_BRANCHES = 0
    try:
        out2 = SystemOne(engine, "decisio-test").answer(req, None, [Image.new("RGB", (96, 96))], image_tag=tag)
    finally:
        gi.BOX_MAX_BRANCHES = old
    assert out2["_timing"]["box"]["questions"]["where"]["branches_capped"] is True
    assert out2["_timing"]["box"]["questions"]["where"]["branches_decoded"] == 0


# ------------------------------------------------------------------------------------------------------ X10


def test_x10_the_exact_tree_debug_needs_the_server_flag(engine):
    engine.reply = lambda i, r: script([100, 200, 500, 800])
    client, _ = make_client(engine)
    r = client.post("/v1/systemone", json=body(), headers={"x-decisio-debug": "box_tree"})
    assert r.status_code == 422 and "--debug-box-tree" in r.json()["detail"]


def test_x10_the_exact_tree_debug_readout(engine):
    engine.reply = lambda i, r: script_branch_main()
    engine.branch_reply = lambda i, r: [sure("0"), sure(",")]
    engine.tree_reply = lambda i, r: [sure(",")]  # every real decode node ends at once (a flat, simple tree)
    client, _ = make_client(engine, debug_box_tree=True)
    r = client.post("/v1/systemone", json=body(), headers={"x-decisio-debug": "box_tree"})
    assert r.status_code == 200, r.text
    tree = r.json()["decisio_debug"]["where"]["box_tree"]["y_min"]
    assert set(tree) == {"exact", "fast", "greedy", "tree_mass", "decodes"}
    assert tree["greedy"] == pytest.approx(0.009)
    assert 0.0 <= tree["tree_mass"] <= 1.0
    assert tree["decodes"] >= 1
