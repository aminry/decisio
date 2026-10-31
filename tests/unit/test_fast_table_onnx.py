# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The fast table's encoder: the model's published ONNX export run by ONNX Runtime (`decisio.serve.fast_table`).

O1  the encoder's arithmetic around the session: [CLS] pooling, L2 normalisation, batches in order, padding and the
    attention mask, truncation at 512 tokens, the empty text (a fake session and a small tokenizer; no model)
O2  the pinned files: a file whose sha256 is not the pinned one is refused; files that cannot be fetched say what to
    run; a missing extra says what to install
O3  the timing stack records the runtime, its version, the ONNX file's hash and the threads, and a replica that differs
    in any of them differs in the stack (so it re-times)
O4  [LOCAL ONLY: needs the pinned files and the torch weights in the local Hugging Face cache, which CI does not have;
    run it before a change to the encoder] the ONNX embeddings equal the torch
    model's within 1e-5 on fixed texts (a text past 512 tokens, unicode, the empty text), alone and in batches
O5  [same] a table fitted on the ONNX embeddings and one fitted on torch's give the same cascade: the same thresholds,
    the same answers, the same certified share and errors

  uv run pytest -q tests/unit/test_fast_table_onnx.py
"""

import builtins

import numpy as np
import pytest

from decisio.serve import fast_table as FT

tokenizers = pytest.importorskip("tokenizers")
sklearn = pytest.importorskip("sklearn")

TEXTS = [
    "what is my account balance",
    "play some jazz music",
    "set an alarm for seven tomorrow morning",
    "turn off the living room lights",
    "How do I reset my password if I no longer have access to the email address on file?",
    "cancel my subscription",
    "the parcel has not arrived yet and the tracking page says it is still at the depot",
    "refund",
    "a",
    "",
    "Ünïcödé tëxt with accents, 日本語のテキスト, and emoji 🙂 mixed in",
    "TRANSFER 200 DOLLARS TO SAVINGS",
    "i would like to book a table for four people at an italian restaurant on friday evening around eight",
    "what's the weather like in San Francisco this weekend",
    "remind me to call mom at 5 pm",
    "translate good morning into french",
    "the quick brown fox jumps over the lazy dog " * 3,
    "x " * 700,  # past 512 tokens: truncated
    "A longer passage about invoices and refunds and payment receipts that goes on for a while. " * 40,
    "tell me a joke",
    "how many calories are in an avocado",
    "is the store open on sunday",
    "what time does the next train leave",
    "can you help me find a good recipe for chicken soup",
    "order more paper towels",
    "who won the game last night",
    "wire transfer failed twice, the bank says the account number is invalid",
    "spam spam spam win a free prize now",
    "short",
    "two words",
    "Hello, world!",
    "set a timer for ten minutes",
    "what is the meaning of life",
    "navigate to the nearest gas station",
]


# ---- O1 -------------------------------------------------------------------------------------------------------------


class FakeSession:
    """Hidden states that depend on the token ids only (position-wise), so padding and batching cannot change a row's
    [CLS] vector unless the encoder feeds it wrongly; records what it was fed."""

    def __init__(self, inputs=("input_ids", "attention_mask", "token_type_ids")):
        self.names, self.fed = inputs, []

    def get_inputs(self):
        return [type("I", (), {"name": n}) for n in self.names]

    def run(self, _outputs, feed):
        self.fed.append({k: v.copy() for k, v in feed.items()})
        ids = feed["input_ids"].astype(np.float64)
        h = np.zeros((*ids.shape, 768))
        h[:, :, 0] = ids  # the vector of position 0 is the first token's id in dimension 0, and its row's length in 1
        h[:, :, 1] = feed["attention_mask"].sum(1, keepdims=True)
        h[:, :, 2] = 1.0
        return [h.astype(np.float32)]


def small_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "a": 4, "b": 5, "c": 6}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    from tokenizers.processors import TemplateProcessing

    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return tok


def test_o1_pooling_normalisation_batches_padding_and_truncation():
    sess = FakeSession()
    enc = FT.OnnxEncoder(sess, small_tokenizer())
    texts = ["a b c", "a", "", "b b b b", "c"]
    one = np.vstack([enc.encode([t]) for t in texts])
    many = enc.encode(texts, batch=2)  # batches of 2, 2 and 1
    assert many.shape == (5, 768) and many.dtype == np.float32
    assert np.allclose(np.linalg.norm(many, axis=1), 1.0, atol=1e-6)
    assert np.allclose(one, many, atol=1e-6)  # the batch a text is in, and so its padding, changes nothing
    # [CLS] pooling: dimension 0 holds the first token's id ([CLS] = 2) over the vector's norm, for every row
    n = np.sqrt(2.0**2 + np.array([len_ for len_ in (5, 3, 2, 6, 3)], dtype=float) ** 2 + 1.0)
    assert np.allclose(many[:, 0], 2.0 / n, atol=1e-6)
    # the model is fed the attention mask and the token types, and a batch is padded to its longest text
    first = sess.fed[len(texts)]
    assert set(first) == {"input_ids", "attention_mask", "token_type_ids"}
    assert first["input_ids"].shape == (2, 5) and first["attention_mask"].tolist() == [[1] * 5, [1, 1, 1, 0, 0]]
    assert first["input_ids"].dtype == np.int64
    # a text past the limit is cut to MAX_SEQ_LENGTH tokens, [SEP] kept
    enc.encode(["a " * 700])
    assert sess.fed[-1]["input_ids"].shape == (1, FT.MAX_SEQ_LENGTH) and sess.fed[-1]["input_ids"][0, -1] == 3
    assert enc.encode([]).shape == (0, 768)


def test_o1_an_export_without_token_types_is_fed_what_it_takes():
    sess = FakeSession(("input_ids", "attention_mask"))
    FT.OnnxEncoder(sess, small_tokenizer()).encode(["a b"])
    assert set(sess.fed[0]) == {"input_ids", "attention_mask"}


# ---- O2 -------------------------------------------------------------------------------------------------------------


def test_o2_a_file_that_is_not_the_pinned_one_is_refused(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    bad = tmp_path / "model.onnx"
    bad.write_bytes(b"not the pinned export")
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(bad))
    with pytest.raises(FT.TableUnavailable, match="not the pinned"):
        FT.OnnxEncoder.load()
    assert FT.file_sha256(bad) != FT.ONNX_SHA256


def test_o2_files_that_cannot_be_fetched_say_what_to_run(monkeypatch):
    pytest.importorskip("onnxruntime")
    import huggingface_hub

    def offline(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(FT.TableUnavailable, match=r"hf download BAAI/bge-base-en-v1.5 onnx/model.onnx tokenizer.json"):
        FT.OnnxEncoder.load()


def test_o2_a_missing_extra_says_what_to_install(monkeypatch):
    real = builtins.__import__

    def refuse(name, *a, **k):
        if name.startswith("onnxruntime"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(FT.TableUnavailable, match=r"decisio\[table\]"):
        FT.OnnxEncoder.load()


# ---- O3 -------------------------------------------------------------------------------------------------------------


def test_o3_the_timing_stack_records_the_runtime_and_a_difference_in_it_is_a_difference_in_the_stack(monkeypatch):
    a = FT.timing_stack({"engine": "fake"})
    assert a["runtime"] == "onnxruntime-fp32" and a["threads"] == FT.THREADS == 4
    assert a["onnx_sha256"] == FT.ONNX_SHA256[:16] and "torch" not in a and "sentence_transformers" not in a
    assert {"onnxruntime", "tokenizers", "encoder", "revision"} <= set(a)
    monkeypatch.setattr(FT, "THREADS", 8)
    assert FT.stack_differences(a, FT.timing_stack({"engine": "fake"})) == ["threads"]
    monkeypatch.setattr(FT, "THREADS", 4)
    real_sha = FT.ONNX_SHA256
    monkeypatch.setattr(FT, "ONNX_SHA256", "0" * 64)
    assert FT.stack_differences(a, FT.timing_stack({"engine": "fake"})) == ["onnx_sha256"]
    monkeypatch.setattr(FT, "ONNX_SHA256", real_sha)
    from importlib import metadata

    real = metadata.version
    monkeypatch.setattr(metadata, "version", lambda n: "9.9.9" if n == "onnxruntime" else real(n))
    assert FT.stack_differences(a, FT.timing_stack({"engine": "fake"})) == ["onnxruntime"]


# ---- O4 and O5: the real files, when they are in the local cache ----------------------------------------------------


def _cached(name):
    from huggingface_hub import try_to_load_from_cache

    got = try_to_load_from_cache(FT.ENCODER, name, revision=FT.REVISION)
    return isinstance(got, str)


needs_files = pytest.mark.skipif(
    not all(_cached(f) for f in (FT.ONNX_FILE, FT.TOKENIZER_FILE, "model.safetensors")),
    reason="local-only: the pinned ONNX file, tokenizer and torch weights are not in the local Hugging Face cache",
)


@pytest.fixture(scope="module")
def encoders():
    pytest.importorskip("onnxruntime")
    import torch
    from transformers import AutoModel, AutoTokenizer

    onnx = FT.OnnxEncoder.load(local_files_only=True)
    tok = AutoTokenizer.from_pretrained(FT.ENCODER, revision=FT.REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(FT.ENCODER, revision=FT.REVISION, local_files_only=True).eval()
    torch.set_num_threads(4)

    def torch_embed(texts):
        out = []
        for i in range(0, len(texts), 32):
            batch = tok(list(texts[i : i + 32]), padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                h = model(**batch).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=1).numpy())
        return np.vstack(out).astype(np.float32)

    return onnx, torch_embed


@needs_files
def test_o4_the_onnx_embeddings_equal_the_torch_models(encoders):
    onnx, torch_embed = encoders
    ref = torch_embed(TEXTS)
    for got in (onnx.encode(TEXTS), np.vstack([onnx.encode([t]) for t in TEXTS])):
        assert float(np.abs(got - ref).max()) < 1e-5
        assert float((got * ref).sum(1).min()) > 0.999999


WORDS = {
    "billing": ["invoice", "refund", "charge", "payment", "receipt", "price", "bill", "card"],
    "shipping": ["parcel", "courier", "delivery", "tracking", "address", "package", "post", "arrive"],
    "access": ["password", "login", "account", "reset", "locked", "token", "sign", "email"],
}
FILLER = ["please", "today", "again", "quickly", "thanks", "hello", "need", "help", "about", "with", "my", "the"]


def corpus(n, seed):
    rng = np.random.default_rng(seed)
    labels = list(WORDS)
    ys, texts = [], []
    for _ in range(n):
        y = int(rng.integers(0, 3))
        k = int(rng.integers(0, 4))  # some texts name no word of their class: the table is unsure of them
        words = list(rng.choice(WORDS[labels[y]], k, replace=False)) + list(rng.choice(FILLER, 4, replace=False))
        rng.shuffle(words)
        texts.append(" ".join(words))
        ys.append(y)
    return texts, np.array(ys)


@needs_files
def test_o5_a_table_fitted_on_onnx_embeddings_is_torchs_table(encoders):
    onnx, torch_embed = encoders
    texts, y = corpus(900, 1)
    fit_t, fit_y, cert_t, cert_y, test_t, test_y = (
        texts[:300],
        y[:300],
        texts[300:600],
        y[300:600],
        texts[600:],
        y[600:],
    )
    tables, served = {}, {}
    rng = np.random.default_rng(7)
    dec = {  # the same decisio side for both: confidences and predictions drawn once
        w: (rng.uniform(0.5, 0.99, 300), np.where(rng.random(300) < 0.9, 0, 1)) for w in ("cert", "test")
    }
    for name, embed in (("onnx", onnx.encode), ("torch", torch_embed)):
        rec = FT.fit(embed(fit_t), fit_y, 3)
        Pc, Pt = FT.predict(rec, embed(cert_t)), FT.predict(rec, embed(test_t))
        cTc, pTc = FT.confidence(Pc)
        cTt, pTt = FT.confidence(Pt)
        th = FT.cascade(cTc, pTc, cert_y, dec["cert"][0], np.where(dec["cert"][1] == 0, cert_y, 1), 0.1, 0.1)
        by_table, certified, pred = FT.served(cTt, pTt, dec["test"][0], np.where(dec["test"][1] == 0, test_y, 1), th)
        tables[name], served[name] = (rec, th), (by_table, certified, pred)
    (ro, to), (rt, tt) = tables["onnx"], tables["torch"]
    assert ro["C"] == rt["C"] and ro["classes"] == rt["classes"]
    assert float(np.abs(FT.unpack(ro["coef"]) - FT.unpack(rt["coef"])).max()) < 1e-3
    assert to == pytest.approx(tt, abs=1e-4) or {k: (None if v is None else round(v, 3)) for k, v in to.items()} == {
        k: (None if v is None else round(v, 3)) for k, v in tt.items()
    }
    for a, b in zip(served["onnx"], served["torch"]):
        assert np.array_equal(a, b)  # the same requests answered by the table, certified, and the same options
    # registration embeds the fit examples in batches and one held-out example at a time; serving one at a time. A text
    # embedded alone and in a batch differs by float noise, which moves no probability of the table by more than 1e-5
    # and so decides no request that the table answers
    alone = np.vstack([onnx.encode([t]) for t in test_t[:60]])
    batched = onnx.encode(test_t[:60])
    assert float(np.abs(FT.predict(ro, alone) - FT.predict(ro, batched)).max()) < 1e-5
    assert served["onnx"][0].mean() > 0.2  # the comparison is not vacuous: the table answers a share of the requests
