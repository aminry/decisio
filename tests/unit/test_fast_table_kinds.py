# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The fast table's student kinds (`"fast_table": true | {"kind": "static" | "encoder" | "auto", "static_model": ...}`).
No model and no download except K9: fakes stand in for both students (`FT.static_embed` and `FT.embed`), and the
virtual clock of tests/unit/test_fast_table_task.py sets how long each takes.

K1  the request: `true` is the static student on potion-base-8M; the object form and its refusals (a bad kind, the
    reserved `hashed`, a bad static model, a static model with the encoder, an unknown key)
K2  the default kind is static: the record carries its feature function, the timing stack its student (no ONNX facts,
    threads or pool), and the table is served with the static student; `static_model: multilingual` is recorded
K3  auto: the kind with the larger saving is kept, the static one on a tie within a point, none when neither clears
    the guard; the report lists every kind tried with its share, error, bound, saving and time; the rule (`FT.choose`)
K4  a record is only served with its own feature function: a server with another static revision, or another
    long-state rule, serves through decisio and says why; the encoder kind records HEAD_TAIL and both files' hashes
K5  a record made before the kinds (no "features") is the encoder's with TRUNC512: it is served, exported, imported and
    re-certified exactly as before, with TRUNC512
K6  the encoder's head-and-tail reading of a long text: the token selection, and what the session is fed (a small
    tokenizer, a fake session)
K7  the static student's arithmetic and files: mean of the token vectors, the unknown token dropped, an empty text
    zeros, model2vec's normalisation; a file that is not the pinned one is refused, files that cannot be fetched say
    what to run (tiny files written by the test)
K8  the timing stack of each kind, and a replica whose static student differs re-times
K9  [LOCAL ONLY: needs model2vec 0.10.0 and the pinned potion files in the local Hugging Face cache] decisio's static
    encoding equals model2vec's `StaticModel.encode(texts, max_length=None)` bit for bit, for both static models, in a
    batch and one text at a time. Run it with

      hf download minishlab/potion-base-8M model.safetensors tokenizer.json config.json \\
          --revision bf8b056651a2c21b8d2565580b8569da283cab23
      hf download minishlab/potion-multilingual-128M model.safetensors tokenizer.json config.json \\
          --revision 73908c3438cf03b6a01bcb9611d62b23d0726f08
      uv run --extra dev --with model2vec==0.10.0 pytest -q -p no:cacheprovider \\
          tests/unit/test_fast_table_kinds.py -k k9

  uv run pytest -q tests/unit/test_fast_table_kinds.py
"""

import json
import zlib

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("obstore")
tokenizers = pytest.importorskip("tokenizers")

from test_fast_table_task import (  # noqa: E402
    CLOCK,
    FakeEngine,
    ask,
    examples,
    fake_embed,
    make,
    virtual_time,  # noqa: F401  (the virtual clock, the fake encoder and CPU: autouse)
)

from decisio.serve import fast_table as FT  # noqa: E402

CALLS = {"static": [], "encoder": []}


def fake_static_embed(name, texts):
    """The static student's stand-in: the encoder's hashed bag of words, or (STATIC_BLIND) one vector for every text,
    so no table threshold holds; it takes `STATIC_MS`."""
    CALLS["static"].append((name, list(texts)))
    CLOCK.spend(STATIC[0])
    if STATIC_BLIND[0]:
        return np.tile(np.eye(1, 256, dtype=np.float32), (len(texts), 1))
    return _bag(texts)


def _bag(texts):
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        body = str(t).split("|", 1)[-1]
        for w in body.replace('"', " ").replace("}", " ").split():
            out[i, zlib.crc32(w.encode()) % 256] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(n == 0, 1.0, n)


def counting_embed(texts, long_state=FT.TRUNC512):
    CALLS["encoder"].append((long_state, list(texts)))
    return fake_embed(texts, long_state)


STATIC = [1.0]  # ms a static featurisation takes
STATIC_BLIND = [False]


@pytest.fixture(autouse=True)
def students(monkeypatch, virtual_time):  # noqa: F811
    CALLS["static"].clear()
    CALLS["encoder"].clear()
    STATIC[0], STATIC_BLIND[0] = 1.0, False
    monkeypatch.setattr(FT, "static_embed", fake_static_embed)
    monkeypatch.setattr(FT, "static_model", lambda name: object())
    monkeypatch.setattr(FT, "embed", counting_embed)


def register(client, fast_table=True, n=800, seed=0):
    body = {"id": "queue", "examples": examples(n, seed), "certify": {"alpha": 0.1, "delta": 0.1}}
    if fast_table is not None:
        body["fast_table"] = fast_table
    return client.post("/v1/tasks", json=body)


# ---- K1 -------------------------------------------------------------------------------------------------------------


def test_k1_true_is_the_static_student_on_potion_base():
    m = FT.MIN_SAVING  # every request records the saving it asks for (the default, when it names none)
    assert FT.parse_request(True) == {"kind": "static", "static_model": "base", "min_saving": m}
    assert FT.parse_request(False) is None and FT.parse_request(None) is None
    assert FT.parse_request({}) == {"kind": "static", "static_model": "base", "min_saving": m}
    assert FT.parse_request({"static_model": "multilingual"}) == {
        "kind": "static",
        "static_model": "multilingual",
        "min_saving": m,
    }
    assert FT.parse_request({"kind": "encoder"}) == {"kind": "encoder", "min_saving": m}
    assert FT.parse_request({"kind": "auto"}) == {"kind": "auto", "static_model": "base", "min_saving": m}
    assert FT.parse_request({"kind": "auto", "static_model": "multilingual"})["static_model"] == "multilingual"


@pytest.mark.parametrize(
    "value, words",
    [
        ({"kind": "hashed"}, "'hashed' is not available in this version"),
        ({"kind": "bge"}, "fast_table.kind: one of 'static', 'encoder', 'auto', not 'bge'"),
        ({"kind": "static", "static_model": "large"}, "fast_table.static_model: one of 'base', 'multilingual'"),
        ({"kind": "encoder", "static_model": "base"}, "for the 'static' and 'auto' kinds only"),
        ({"kind": "static", "speed": 1}, "not 'speed'"),
        ("yes", "true or false, or an object"),
        (1, "true or false, or an object"),
    ],
)
def test_k1_the_object_form_is_checked(value, words):
    with pytest.raises(ValueError, match=None) as e:
        FT.parse_request(value)
    assert words in str(e.value)
    client, so, _ = make()
    r = register(client, value)
    assert r.status_code == 422 and words in r.json()["detail"] and not so.task_store.by_key


# ---- K2 -------------------------------------------------------------------------------------------------------------


def test_k2_the_default_kind_is_static_and_its_feature_function_is_recorded_and_served():
    client, so, _ = make()
    r = register(client)
    assert r.status_code == 200, r.text
    t = r.json()
    ft, rep = t["fast_table"], t["fast_table_report"]
    spec = FT.STATIC_MODELS["base"]
    assert ft["features"] == {
        "kind": "static",
        "model": "minishlab/potion-base-8M",
        "revision": spec["revision"],
        "files_sha256": spec["files_sha256"],
        "long_state": "whole",
    }
    assert ft["encoder"] == "minishlab/potion-base-8M" and ft["revision"] == spec["revision"]
    assert rep["kept"] and rep["kind"] == "static"
    assert rep["request"] == {"kind": "static", "static_model": "base", "min_saving": FT.MIN_SAVING}
    assert [k["kind"] for k in rep["kinds"]] == ["static"] and rep["kinds"][0]["chosen"]
    stack = ft["timing"]["stack"]
    assert stack["kind"] == "static" and stack["model"] == "minishlab/potion-base-8M"
    assert not {"onnx_sha256", "threads", "encoders", "onnxruntime", "encoder"} & set(stack)
    assert CALLS["encoder"] == []  # the encoder was never asked
    n_static, calls = len(CALLS["static"]), FakeEngine.calls
    a = ask(client, "billing", right=False, conf=0.4)
    assert a["answered_by"] == "table" and a["choice"] == "billing" and FakeEngine.calls == calls
    assert len(CALLS["static"]) == n_static + 1 and CALLS["static"][-1][0] == "base" and CALLS["encoder"] == []
    assert t["status"]["fast_table"]["state"] == "active"


def test_k2_the_multilingual_static_model_is_recorded():
    client, so, _ = make()
    t = register(client, {"kind": "static", "static_model": "multilingual"}).json()
    f = t["fast_table"]["features"]
    assert (
        f["model"] == "minishlab/potion-multilingual-128M"
        and f["revision"] == FT.STATIC_MODELS["multilingual"]["revision"]
    )
    assert {c[0] for c in CALLS["static"]} == {"multilingual"}
    assert ask(client, "billing", right=False, conf=0.4)["answered_by"] == "table"
    assert CALLS["static"][-1][0] == "multilingual"


# ---- K3 -------------------------------------------------------------------------------------------------------------


def test_k3_auto_keeps_the_kind_that_saves_more_and_lists_both():
    STATIC_BLIND[0] = True  # the static student sees nothing: no threshold holds for it
    client, so, _ = make()
    t = register(client, {"kind": "auto"}).json()
    rep = t["fast_table_report"]
    assert rep["kept"] and rep["kind"] == "encoder" and t["fast_table"]["features"]["kind"] == "encoder"
    assert t["fast_table"]["features"]["long_state"] == FT.HEAD_TAIL
    assert [k["kind"] for k in rep["kinds"]] == ["static", "encoder"]
    st, en = rep["kinds"]
    assert not st["eligible"] and not st["chosen"] and "no confidence threshold" in st["why_not"]
    assert en["eligible"] and en["chosen"] and "largest expected saving" in rep["why"]
    for k in rep["kinds"]:
        assert {
            "table_share",
            "errors_at_threshold",
            "error_rate_at_threshold",
            "upper_bound_at_threshold",
            "expected_saving",
            "table_ms_mean",
            "model",
            "revision",
            "long_state",
        } <= set(k)
    assert en["table_share"] > 0.3 and en["upper_bound_at_threshold"] >= en["error_rate_at_threshold"]
    assert rep["rule"].startswith("auto: each kind (static minishlab/potion-base-8M, encoder BAAI/bge-base-en-v1.5)")
    assert "static one when the two savings are within 1 point" in rep["rule"]
    assert t["fast_table"]["timing"]["stack"]["encoders"] == FT.ENCODERS  # the encoder's stack, as today
    calls = len(CALLS["static"])
    assert ask(client, "billing", right=False, conf=0.4)["answered_by"] == "table"
    assert len(CALLS["static"]) == calls and CALLS["encoder"][-1][0] == FT.HEAD_TAIL  # served by the encoder


def test_k3_auto_keeps_the_faster_static_student_when_both_certify_alike():
    STATIC[0] = 1.0  # the encoder takes CLOCK.table_ms = 5 ms on the same features
    client, so, _ = make()
    rep = register(client, {"kind": "auto"}).json()["fast_table_report"]
    st, en = rep["kinds"]
    assert st["table_share"] == en["table_share"] and st["expected_saving"] > en["expected_saving"] + FT.TIE
    assert rep["kind"] == "static" and st["chosen"] and "largest expected saving" in rep["why"]


def test_k3_auto_prefers_static_on_a_tie():
    STATIC[0] = CLOCK.table_ms  # the same features and the same time: the same saving
    client, so, _ = make()
    t = register(client, {"kind": "auto"}).json()
    rep = t["fast_table_report"]
    st, en = rep["kinds"]
    assert st["expected_saving"] == en["expected_saving"] and en["eligible"]
    assert rep["kind"] == "static" and "tie" in rep["why"] and t["fast_table"]["features"]["kind"] == "static"


def test_k3_auto_keeps_no_table_when_neither_clears_the_guard():
    STATIC[0], CLOCK.table_ms = 55.0, 55.0  # both nearly decisio's own 60 ms
    client, so, _ = make()
    t = register(client, {"kind": "auto"}).json()
    rep = t["fast_table_report"]
    assert not rep["kept"] and rep["kind"] is None and "no kind has a table threshold" in rep["reason"]
    assert "fast_table" not in t and [k["eligible"] for k in rep["kinds"]] == [False, False]
    assert all("under the 10% required" in k["why_not"] for k in rep["kinds"])
    assert t["status"]["fast_table"]["state"] == "off" and t["certificate"]["certified"]


def test_k3_the_rule():
    def c(kind, saving, holds=True):
        return {"kind": kind, "saving": saving, "holds": holds}

    assert FT.choose([c("static", 0.50), c("encoder", 0.52)])[0] == 1
    assert FT.choose([c("static", 0.50), c("encoder", 0.505)])[0] == 0  # within a point: static
    assert FT.choose([c("static", 0.50), c("encoder", 0.51)])[0] == 0  # exactly a point: still a tie
    assert FT.choose([c("static", 0.60), c("encoder", 0.30)])[0] == 0
    assert FT.choose([c("static", 0.60, holds=False), c("encoder", 0.30)])[0] == 1
    assert FT.choose([c("static", 0.09), c("encoder", 0.08)])[0] is None  # under the default 10%
    assert FT.choose([c("static", 0.09), c("encoder", 0.095)])[0] is None  # a tie under the guard is no table
    assert FT.choose([c("static", 0.24), c("encoder", 0.20)], min_saving=0.25)[0] is None  # a task's own bar
    assert FT.choose([c("static", 0.24), c("encoder", 0.20)])[0] == 0
    assert FT.choose([c("encoder", 0.40)])[0] == 0


# ---- K4 -------------------------------------------------------------------------------------------------------------


def test_k4_the_encoder_kind_records_head_and_tail_and_both_files():
    client, so, _ = make()
    t = register(client, {"kind": "encoder"}).json()
    f = t["fast_table"]["features"]
    assert f == {
        "kind": "encoder",
        "model": FT.ENCODER,
        "revision": FT.REVISION,
        "files_sha256": {FT.ONNX_FILE: FT.ONNX_SHA256, FT.TOKENIZER_FILE: FT.TOKENIZER_SHA256},
        "long_state": "head256+tail254",
    }
    assert {ls for ls, _ in CALLS["encoder"]} == {FT.HEAD_TAIL}


def test_k4_a_server_with_another_static_revision_serves_through_decisio(monkeypatch):
    from obstore.store import MemoryStore

    store = MemoryStore()
    a, _, _ = make(store, "a")
    assert register(a).status_code == 200
    other = {**FT.STATIC_MODELS["base"], "revision": "9" * 40}
    monkeypatch.setitem(FT.STATIC_MODELS, "base", other)
    b, _, _ = make(store, "b")
    st = b.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "unavailable" and "potion-base-8M at revision bf8b0566" in st["reason"]
    assert "register the task again" in st["reason"]
    calls = len(CALLS["static"])
    assert ask(b, "billing", right=True, conf=0.97)["answered_by"] == "decisio" and len(CALLS["static"]) == calls
    r = b.post("/v1/tasks/queue/certify", json={"examples": examples(300, seed=98)})
    assert r.status_code == 422 and "register the task again" in r.json()["detail"]


def test_k4_a_rule_this_server_does_not_compute_is_not_served():
    assert FT.cannot_compute({**FT.student_features("encoder"), "long_state": "chunk512x8"})
    assert FT.cannot_compute({**FT.student_features("static"), "long_state": FT.HEAD_TAIL})
    assert FT.cannot_compute({"kind": "hashed"})
    assert FT.cannot_compute({**FT.student_features("encoder"), "files_sha256": {FT.ONNX_FILE: "0" * 64}})
    for kind in ("static", "encoder"):
        assert FT.cannot_compute(FT.student_features(kind)) is None
    assert FT.cannot_compute(FT.student_features("static", "multilingual")) is None


# ---- K5 -------------------------------------------------------------------------------------------------------------


def legacy_export(monkeypatch):
    """A task registered as decisio registered it before the kinds: the encoder, read by TRUNC512, no "features"."""
    legacy = FT.record_features({"encoder": FT.ENCODER, "revision": FT.REVISION})
    fit = FT.fit
    with monkeypatch.context() as m:
        m.setattr(FT, "requested_features", lambda request: [legacy])
        m.setattr(FT, "fit", lambda X, y, k, features=None: fit(X, y, k))
        client, so, _ = make()
        t = register(client, {"kind": "encoder"}).json()
        full = client.get("/v1/tasks?full=1").json()["tasks"][0]
    return client, t, full


def test_k5_a_record_without_features_is_served_imported_and_exported_as_before(monkeypatch):
    old, t, full = legacy_export(monkeypatch)
    assert "features" not in full["fast_table"] and full["fast_table"]["encoder"] == FT.ENCODER
    assert t["fast_table_report"]["features"]["long_state"] == FT.TRUNC512
    assert set(full["fast_table"]["timing"]["stack"]) == set(FT.timing_stack({}))  # the stack of before the kinds
    new, so, _ = make()
    assert new.post("/v1/tasks/import", json={"tasks": [full]}).status_code == 200
    got = new.get("/v1/tasks/queue").json()
    assert got["status"]["fast_table"]["state"] == "active" and got["certificate"]["id"] == t["certificate"]["id"]
    CALLS["encoder"].clear()
    for g, right, conf in (("billing", False, 0.4), ("shipping", True, 0.97), ("access", True, 0.6)):
        assert ask(new, g, right, conf, seed=7) == ask(old, g, right, conf, seed=7)
    assert {ls for ls, _ in CALLS["encoder"]} == {FT.TRUNC512} and CALLS["static"] == []
    again = new.get("/v1/tasks?full=1").json()["tasks"][0]
    assert json.dumps(again["fast_table"], sort_keys=True) == json.dumps(full["fast_table"], sort_keys=True)
    r = new.post("/v1/tasks/queue/certify", json={"examples": examples(300, seed=99)})
    assert r.status_code == 200, r.text
    rc = r.json()
    assert "features" not in rc["fast_table"] and rc["fast_table"]["coef"] == t["fast_table"]["coef"]
    assert rc["fast_table_report"]["features"]["long_state"] == FT.TRUNC512
    assert {ls for ls, _ in CALLS["encoder"]} == {FT.TRUNC512}


# ---- K6 -------------------------------------------------------------------------------------------------------------


def test_k6_head_and_tail_selection():
    seq = list(range(600))  # [CLS] = 0, tokens 1..598, [SEP] = 599
    got = FT.head_tail(seq)
    assert len(got) == FT.MAX_SEQ_LENGTH == 512 and got[0] == 0 and got[-1] == 599
    assert got[1:257] == list(range(1, 257)) and got[257:511] == list(range(345, 599))
    assert FT.head_tail(list(range(512))) == list(range(512))  # fits the window: whole
    assert FT.head_tail(list(range(513)))[1:257] == list(range(1, 257))
    assert FT.head_tail(list(range(513)))[257:] == list(range(258, 513))
    assert FT.HEAD_TAIL == "head256+tail254" and FT.HEAD_TOKENS + FT.TAIL_TOKENS + 2 == FT.MAX_SEQ_LENGTH


def test_k6_the_encoder_is_fed_the_head_and_the_tail():
    from test_fast_table_onnx import FakeSession
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{f"w{i}": 4 + i for i in range(1000)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    sess = FakeSession()
    enc = FT.OnnxEncoder(sess, tok)
    long = " ".join(f"w{i}" for i in range(700))
    enc.encode([long], long_state=FT.HEAD_TAIL)
    ids = sess.fed[-1]["input_ids"][0].tolist()
    assert ids == [2, *range(4, 4 + 256), *range(4 + 700 - 254, 4 + 700), 3]
    enc.encode([long])  # TRUNC512, as before
    assert sess.fed[-1]["input_ids"][0].tolist() == [2, *range(4, 4 + 510), 3]
    short = "w5 w6 w7"
    a, b = enc.encode([short]), enc.encode([short], long_state=FT.HEAD_TAIL)
    assert np.array_equal(a, b)  # a text that fits is read whole by both rules
    enc.encode([long, short], long_state=FT.HEAD_TAIL)  # one batch: padded to its longest, masked
    fed = sess.fed[-1]
    assert fed["input_ids"].shape == (2, 512) and fed["attention_mask"][1].tolist() == [1] * 5 + [0] * 507
    assert fed["input_ids"][1, 5:].tolist() == [0] * 507 and fed["token_type_ids"].sum() == 0
    with pytest.raises(ValueError, match="long text"):
        enc.encode([short], long_state="chunk512x8")


# ---- K7 -------------------------------------------------------------------------------------------------------------


def tiny_static(tmp_path, monkeypatch, tensors=None):
    """A tiny static model written to tmp_path and pinned as "base"; returns its matrix."""
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    E = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    save_file(tensors or {"embeddings": E}, str(tmp_path / "model.safetensors"))
    files = {f: FT.file_sha256(tmp_path / f) for f in ("model.safetensors", "tokenizer.json")}
    monkeypatch.setitem(FT.STATIC_MODELS, "base", {"repo": "t/tiny", "revision": "1" * 40, "files_sha256": files})
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, **k: str(tmp_path / f))
    return E


def test_k7_the_static_arithmetic(tmp_path, monkeypatch):
    E = tiny_static(tmp_path, monkeypatch)
    m = FT.StaticEncoder.load("base")
    got = m.encode(["a b", "a zzz b", "", "zzz", "d d d c"])
    assert got.dtype == np.float32 and got.shape == (5, 8)
    mean = E[[1, 2]].mean(axis=0)
    assert np.allclose(got[0], mean / np.linalg.norm(mean)) and np.array_equal(got[0], got[1])  # [UNK] dropped
    assert not got[2].any() and not got[3].any()  # nothing known: zeros, not NaN
    mean = E[[4, 4, 4, 3]].mean(axis=0)
    assert np.allclose(got[4], mean / np.linalg.norm(mean))
    long = " ".join(["a", "b"] * 5000)  # no truncation: the whole text
    assert np.allclose(m.encode([long])[0], got[0], atol=1e-4)  # float32 sums of 10,000 vectors


def test_k7_a_static_file_that_is_not_the_pinned_one_is_refused(tmp_path, monkeypatch):
    tiny_static(tmp_path, monkeypatch)
    spec = FT.STATIC_MODELS["base"]
    monkeypatch.setitem(spec["files_sha256"], "tokenizer.json", "0" * 64)
    with pytest.raises(FT.TableUnavailable, match="not the pinned"):
        FT.StaticEncoder.load("base")


def test_k7_a_static_file_of_another_layout_is_refused(tmp_path, monkeypatch):
    E = np.zeros((5, 8), np.float32)
    tiny_static(tmp_path, monkeypatch, tensors={"embeddings": E, "mapping": np.zeros(5, np.int64)})
    with pytest.raises(FT.TableUnavailable, match="not the one 'embeddings' matrix"):
        FT.StaticEncoder.load("base")


def test_k7_static_files_that_cannot_be_fetched_say_what_to_run(monkeypatch):
    import huggingface_hub

    def offline(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(
        FT.TableUnavailable,
        match=r"hf download minishlab/potion-base-8M model.safetensors tokenizer.json --revision bf8b0566",
    ):
        FT.StaticEncoder.load("base")


def test_k7_a_missing_static_student_refuses_the_registration_and_auto_names_it(monkeypatch):
    def missing(name):
        raise FT.TableUnavailable("the static model files cannot be fetched (OSError: offline)")

    monkeypatch.setattr(FT, "static_model", missing)
    client, so, _ = make()
    r = register(client)
    assert r.status_code == 422 and "cannot be fetched" in r.json()["detail"] and not so.task_store.by_key
    t = register(client, {"kind": "auto"}).json()
    rep = t["fast_table_report"]
    assert rep["kind"] == "encoder" and [k["kind"] for k in rep["kinds"]] == ["encoder"]
    assert rep["unavailable"][0]["kind"] == "static" and "cannot be fetched" in rep["unavailable"][0]["reason"]


# ---- K8 -------------------------------------------------------------------------------------------------------------


def test_k8_the_timing_stack_of_each_kind(monkeypatch):
    enc = FT.timing_stack({"engine": "fake"}, FT.student_features("encoder"))
    assert enc == FT.timing_stack({"engine": "fake"})  # the encoder's stack is the one of before the kinds
    st = FT.timing_stack({"engine": "fake"}, FT.student_features("static"))
    assert (
        st["kind"] == "static" and st["runtime"] == "numpy" and st["revision"] == FT.STATIC_MODELS["base"]["revision"]
    )
    ml = FT.timing_stack({"engine": "fake"}, FT.student_features("static", "multilingual"))
    assert FT.stack_differences(st, ml) == ["model", "revision"]
    assert {"encoder", "onnx_sha256", "threads", "encoders"} <= set(FT.stack_differences(st, enc))


def test_k8_a_replica_on_another_cpu_re_times_a_static_table(monkeypatch):
    from obstore.store import MemoryStore

    store = MemoryStore()
    a, _, _ = make(store, "a")
    assert register(a).status_code == 200
    monkeypatch.setattr(FT, "cpu_model", lambda: "cpu-B")
    b, _, _ = make(store, "b")
    st = b.get("/v1/tasks/queue").json()["status"]["fast_table"]
    assert st["state"] == "timing" and "cpu" in st["reason"]
    for i in range(FT.SHADOW_EVERY * FT.SHADOW_N):
        ask(b, "access", right=True, conf=0.95, seed=100 + i, confident_text=False)
    assert b.get("/v1/tasks/queue").json()["status"]["fast_table"]["state"] == "active"


# ---- K9: the real potion files and model2vec, when they are here ----------------------------------------------------

PARITY_TEXTS = [
    "what is my account balance",
    "cancel my subscription",
    "",
    "a",
    "Ünïcödé tëxt with accents, 日本語のテキスト, and emoji 🙂 mixed in",
    "주문한 상품이 아직 도착하지 않았어요. 배송 조회를 해 주세요.",
    "Привет, где моя посылка? Je voudrais un remboursement, s'il vous plaît.",
    "zzzzqqqxxj 🙂🙂🙂",
    "TRANSFER 200 DOLLARS TO SAVINGS",
    "the parcel has not arrived yet and the tracking page says it is still at the depot",
    " ".join(f"invoice {i} refund payment receipt for order number {i * 7}" for i in range(300)),  # about 3,000 tokens
    '{"message": "please reset my password", "account": {"id": 42, "plan": "pro"}}',
]


def _potion_cached(name):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    spec = FT.STATIC_MODELS[name]
    return all(
        isinstance(try_to_load_from_cache(spec["repo"], f, revision=spec["revision"]), str)
        for f in ("model.safetensors", "tokenizer.json", "config.json")
    )


@pytest.mark.parametrize("name", ["base", "multilingual"])
def test_k9_the_static_encoding_equals_model2vecs(name, monkeypatch):
    m2v = pytest.importorskip(
        "model2vec", reason="local-only: needs model2vec 0.10.0 (uv run --with model2vec==0.10.0)"
    )
    if m2v.__version__ != "0.10.0":
        pytest.skip(f"local-only: the parity is with model2vec 0.10.0, not {m2v.__version__}")
    if not _potion_cached(name):
        pytest.skip(f"local-only: the pinned {FT.STATIC_MODELS[name]['repo']} files are not in the local cache")
    import os

    from huggingface_hub import try_to_load_from_cache

    spec = FT.STATIC_MODELS[name]
    folder = os.path.dirname(try_to_load_from_cache(spec["repo"], "model.safetensors", revision=spec["revision"]))
    ref_model = m2v.StaticModel.from_pretrained(folder, force_download=False)
    ours = FT.StaticEncoder.load(name, local_files_only=True)
    long_tokens = len(ours.tokenizer.encode(PARITY_TEXTS[10], add_special_tokens=False).ids)
    assert long_tokens > 2500  # the long text is long
    ref = ref_model.encode(PARITY_TEXTS, max_length=None)
    batch = ours.encode(PARITY_TEXTS)
    one = np.vstack([ours.encode([t]) for t in PARITY_TEXTS])
    assert ref.dtype == batch.dtype == np.float32
    assert np.array_equal(batch, ref) and np.array_equal(one, ref)
    assert not ref[2].any() and np.isclose(np.linalg.norm(ref[0]), 1.0)  # the empty text is zeros


def test_k_an_earlier_server_refuses_a_head_and_tail_encoder_table():
    """An earlier server compares only `encoder` and `revision` with its own (bge at REVISION) before serving a table:
    a new encoder table names its encoder with the head-and-tail rule, so that check fails there and the earlier server
    serves the task through decisio instead of reading the table's features with its 512-token cut."""
    feat = FT.student_features("encoder")
    assert FT.legacy_encoder_name(feat) == f"{FT.ENCODER}#{FT.HEAD_TAIL}" != FT.ENCODER
    assert FT.legacy_encoder_name(None) == FT.ENCODER
    static = FT.student_features("static")
    assert FT.legacy_encoder_name(static) == static["model"] != FT.ENCODER
    # this server still reads the record by its features, not by that name
    assert FT.cannot_compute(feat) is None
