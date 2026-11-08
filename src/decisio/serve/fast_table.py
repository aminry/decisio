# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""An opt-in fast table for a fixed-option task: a student's features and a per-task classifier answer first.

For short texts that arrive in volume (an assistant's commands, spam, a fixed set of intents) a small classifier trained
on the task's own labelled examples answers most requests faster than decisio's own path, on the host's CPU. The table
is a student's features of the request's text, a multinomial logistic regression on the task's examples, and a
confidence threshold; a request whose confidence reaches the threshold is answered by the table and decisio's model is
not asked. Every other request goes to decisio as it would without the table.

The table is off unless the registration asks for it (`"fast_table": true`, or `{"kind": ..., "static_model": ...}`),
and it keeps itself only where it pays: at registration the server times the table and decisio's own path on the task's
certification examples and keeps the table only if the expected mean time per request falls by at least the task's
`min_saving` (`"fast_table": {"min_saving": x}`, `MIN_SAVING` by default; `MIN_SAVING_RANGE`). Where it does not, the
registration says so and the task is served as it would be without a table. A table recorded before the setting existed
has none, and is held to the bar it was kept under, `LEGACY_MIN_SAVING` (`record_min_saving`).

A yes/no task (type "noul") gets a two-class table: the same certificate, guard and rules, with the table's P(yes)
calibrated by one temperature on its logit, fitted on the fit examples by cross-fitting (`cross_fit_temperature`), so
that the probability it reports is its own and calibrated, and the score is fixed before any certification label is
read; the temperature changes neither its decision nor the order of its confidences. Under --noul-commit its confidence
is read after the rule, as decisio's certified yes/no answers are (`top_confidence`).

The student is a kind (`KINDS`):

- `static` (the default): a Model2Vec static embedding (`STATIC_MODELS`: potion-base-8M, English, or
  potion-multilingual-128M): the mean of the text's token vectors, L2-normalised, computed here exactly as model2vec
  0.10.0's `StaticModel.encode(texts, max_length=None)` does. It reads the whole text and costs well under a millisecond
  on a short one.
- `encoder`: `BAAI/bge-base-en-v1.5` at a pinned revision, the model's own published ONNX export (`onnx/model.onnx`,
  sha256 pinned below) run by ONNX Runtime in fp32 on `THREADS` threads with the model's own fast tokenizer, [CLS]
  pooling and L2 normalisation, as the model's sentence-transformers configuration does. It gives the embeddings of the
  torch model to about 1.5e-6. It costs about 15 ms a short text and well over 100 on a long one, so it pays only where
  it certifies much more than the static student. A text longer than its 512-token window is read as its first
  `HEAD_TOKENS` and last `TAIL_TOKENS` tokens (`HEAD_TAIL`); tables made before the kinds read the first 510
  (`TRUNC512`) and are served so.
- `auto`: both are fitted and certified, each is timed on this host, and the one that saves more is kept (`choose`).
- `hashed` is a reserved name, not available in this version.

Every table records its feature function (`"features"`: the kind, the model, its revision, the files' sha256 and the
long-state rule) and is only ever served with it: a server that computes features otherwise serves through decisio and
says why. A table recorded before the kinds has no "features" and is the encoder's with `TRUNC512` (`record_features`).

The table and decisio form a two-stage cascade with one certificate (the way the certificate of a task is made, on
examples held out of the fit): the table's threshold is the lowest at which at most `alpha` of its answers are wrong,
with probability at least 1 - delta/2; decisio's threshold is made the same way on the requests the table defers. An
answer of either stage whose confidence reaches its threshold is certified, and the promise is the cascade's: at most
`alpha` wrong among the certified answers, with probability at least 1 - delta. Every kind has the same certificate;
only the features differ.

A table can also be learned from a task's own traffic, with decisio's answers as its labels and an agreement
certificate (`"fast_table": {"learn": true}`): decisio.serve.fast_table_learn.

Needs the `table` extra (scikit-learn for the fit; onnxruntime for the encoder). Each student's files are fetched from
the Hub at the pinned revision on first use and checked against the pinned hashes. Standard library, numpy, tokenizers,
safetensors, huggingface_hub and the extra.
"""

from __future__ import annotations

import base64
import collections
import math
import os
import platform
import queue
import threading
import warnings
from pathlib import Path

import numpy as np

from decisio.names import record_format
from decisio.readout import certify
from decisio.serve import monitor
from decisio.serve.stack import sha256

FORMAT = record_format("fast-table")
ENCODER = "BAAI/bge-base-en-v1.5"
REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
MAX_SEQ_LENGTH = 512
ONNX_FILE = "onnx/model.onnx"  # in the model's repository at REVISION
ONNX_SHA256 = "9bc579acdba21c253c62a9bf866891355a63ffa3442b52c8a37d75b2ccb91848"
TOKENIZER_FILE = "tokenizer.json"
TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"  # tokenizer.json at REVISION
THREADS = (
    4  # intra-op threads of the ONNX session: explicit and recorded in the timing stack (8, 16: slower at batch 1)
)
# encoder sessions that run at once, each with THREADS threads: two kept one request's time (15.3 against 16.3 ms on an
# Apple M5 Pro) and raised the rate at 16 in flight from 68 to 91 a second; four single-threaded ones reached 99 a
# second but made one request slower (26 ms). Batching texts in one call was slower per text on the CPU (batch 8 14.6
# ms, batch 32 18.6, against 13.9 one at a time) and is not used. Recorded in the timing stack.
ENCODERS = 2
CS = (0.1, 1.0, 10.0, 100.0)  # the L2 strength of the classifier, chosen by 5-fold cross-validated log loss
MIN_SAVING = 0.10  # the table is kept when it cuts the expected mean time per request by at least this share (default)
MIN_SAVING_RANGE = (0.01, 0.90)  # a task's own `min_saving`, inclusive
# a table record (or a learning task) made before `min_saving` was a setting was kept under this bar, and is held to it
LEGACY_MIN_SAVING = 0.25
TEMPERATURES = (0.02, 50.0)  # the range a yes/no table's calibration temperature is searched in
MIN_EXAMPLES = 100  # fewer fit examples than this fit no table
FOLDS = 5
WINDOW = 50  # table-path latencies kept for the rolling median
MIN_LATENCY_SAMPLES = 20  # a rolling median is judged only from this many table answers
SHADOW_EVERY = 10  # while the table is off, every 10th request also times the table (after decisio has answered)
SHADOW_N = 30  # and the table is put back when 30 such pairs show it pays (`timing_guard`)
SAMPLE_EVERY = 50  # while on, every 50th table answer is also read by decisio and compared


# the long-state rules: how a text longer than the encoder's window is read. A table records its rule and is served with
# it only. TRUNC512: the first MAX_SEQ_LENGTH - 2 tokens (tables made before the kinds). HEAD_TAIL: the first
# HEAD_TOKENS and the last TAIL_TOKENS of the text, between [CLS] and [SEP], when it does not fit the window, else the
# whole text (MAX_SEQ_LENGTH is the model's position limit, so the tail is 254 and not 256). WHOLE: the whole text
# (static).
TRUNC512 = "trunc512"
HEAD_TOKENS = 256
TAIL_TOKENS = MAX_SEQ_LENGTH - 2 - HEAD_TOKENS
HEAD_TAIL = f"head{HEAD_TOKENS}+tail{TAIL_TOKENS}"
WHOLE = "whole"

# The static students: Model2Vec potion models, MIT licensed (https://huggingface.co/minishlab/potion-base-8M and
# https://huggingface.co/minishlab/potion-multilingual-128M, model cards, "license: mit"; the model2vec library that
# made them is MIT too, https://github.com/MinishLab/model2vec). Pinned at their main revisions of 2026-11-05; the
# sha256 of each file is the Hub's LFS hash where the file is in LFS, else the hash of the file at the revision.
# where an image with the students built in keeps their files (docker/Dockerfile.table; docs/running.md): a student's
# pinned file is read from <dir>/<repo>/<revision>/<file> when it is there, before the Hugging Face cache or the Hub
STUDENTS_DIR_ENV = "DECISIO_STUDENTS_DIR"


def student_file(repo: str, filename: str, revision: str, local_files_only: bool = False) -> str:
    """The path of a student's pinned file: under $DECISIO_STUDENTS_DIR when it holds it, else from the Hugging Face
    cache or the Hub (`hf_hub_download`). The caller checks its sha256 either way, so a file in the directory that is
    not the pinned one is refused, never served."""
    root = os.environ.get(STUDENTS_DIR_ENV)
    if root:
        path = Path(root) / repo / revision / filename
        if path.is_file():
            return str(path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, revision=revision, local_files_only=local_files_only)


def pinned_files(encoder: bool = False) -> list[dict]:
    """Every student file decisio pins: {"repo", "revision", "file", "sha256"}, the static students' and, with
    `encoder`, the encoder's (what docker/fetch_students.py puts in an image)."""
    out = [
        {"repo": spec["repo"], "revision": spec["revision"], "file": f, "sha256": digest}
        for spec in STATIC_MODELS.values()
        for f, digest in spec["files_sha256"].items()
    ]
    if encoder:
        out += [
            {"repo": ENCODER, "revision": REVISION, "file": ONNX_FILE, "sha256": ONNX_SHA256},
            {"repo": ENCODER, "revision": REVISION, "file": TOKENIZER_FILE, "sha256": TOKENIZER_SHA256},
        ]
    return out


STATIC_MODELS = {
    "base": {
        "repo": "minishlab/potion-base-8M",
        "revision": "bf8b056651a2c21b8d2565580b8569da283cab23",
        "files_sha256": {
            "model.safetensors": "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2",
            "tokenizer.json": "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50",
        },
    },
    "multilingual": {
        "repo": "minishlab/potion-multilingual-128M",
        "revision": "73908c3438cf03b6a01bcb9611d62b23d0726f08",
        "files_sha256": {
            "model.safetensors": "14b5eb39cb4ce5666da8ad1f3dc6be4346e9b2d601c073302fa0a31bf7943397",
            "tokenizer.json": "19f1909063da3cfe3bd83a782381f040dccea475f4816de11116444a73e1b6a1",
        },
    },
}
STATIC_TENSOR = "embeddings"  # the embedding matrix's name in model.safetensors (model2vec.persistence)

KINDS = ("static", "encoder", "auto")
RESERVED_KINDS = ("hashed",)  # names kept for kinds to come: refused, not available in this version
DEFAULT_KIND = "static"
DEFAULT_STATIC_MODEL = "base"
TIE = 0.01  # auto: two kinds whose expected savings are this close are a tie, and the static one is kept

_lock = threading.Lock()
_encoder = None  # the first of the pool's encoders, once loaded
_pool: queue.Queue | None = None  # the ENCODERS encoders, each used by one encode at a time
_static_lock = threading.Lock()
_static: dict = {}  # the static students loaded, by name in STATIC_MODELS; one each, read by any thread


class TableUnavailable(ImportError):
    """The `table` extra is not installed, or the encoder's files cannot be had or are not the pinned ones."""


def _need(what: str) -> TableUnavailable:
    return TableUnavailable(f"the fast table needs the `table` extra ({what}): pip install 'decisio[table]'")


class OnnxEncoder:
    """The pinned encoder: `encode(texts, long_state=...)` gives L2-normalised float32 [CLS] embeddings, one row each.
    Not thread-safe (the tokenizer's padding is set on it): `embed` takes one of the pool's encoders for each call."""

    def __init__(self, session, tokenizer, files=None):
        from tokenizers import Tokenizer

        self.files = files  # (the ONNX file, the tokenizer file) it was loaded from, checked: `another` reuses them
        self.session = session
        self.inputs = [i.name for i in session.get_inputs()]
        self.whole = Tokenizer.from_str(tokenizer.to_str())  # no truncation, no padding: for HEAD_TAIL
        self.whole.no_truncation()
        self.whole.no_padding()
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.tokenizer = tokenizer  # TRUNC512, as the encoder has read texts since 0.38.0
        tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        tokenizer.enable_padding(pad_id=self.pad_id, pad_token="[PAD]")

    @classmethod
    def load(cls, local_files_only: bool = False) -> OnnxEncoder:
        """The session and tokenizer of the pinned files, fetched from the Hub at REVISION (or taken from the local
        cache) and checked against ONNX_SHA256."""
        try:
            import onnxruntime  # noqa: F401  (here only to say what to install before anything is fetched)
        except ImportError as e:
            raise _need("onnxruntime") from e
        try:
            import huggingface_hub  # noqa: F401  (student_file fetches with it)
            from tokenizers import Tokenizer  # noqa: F401  (as above; from_files loads it)
        except ImportError as e:
            raise _need("tokenizers and huggingface_hub") from e
        try:
            model = student_file(ENCODER, ONNX_FILE, REVISION, local_files_only)
            tok = student_file(ENCODER, TOKENIZER_FILE, REVISION, local_files_only)
        except Exception as e:  # no network, no such file, a gated or moved repository
            raise TableUnavailable(
                f"the fast table's encoder files cannot be fetched ({type(e).__name__}: {e}); fetch them where the "
                f"server can read them: hf download {ENCODER} {ONNX_FILE} {TOKENIZER_FILE} --revision {REVISION}"
            ) from e
        for path, pinned in ((model, ONNX_SHA256), (tok, TOKENIZER_SHA256)):
            digest = file_sha256(path)
            if digest != pinned:
                raise TableUnavailable(
                    f"{path} has sha256 {digest}, not the pinned {pinned}: the fast table refuses it"
                )
        return cls.from_files(model, tok)

    @classmethod
    def from_files(cls, model: str, tok: str) -> OnnxEncoder:
        """A session and a tokenizer of files already fetched and checked."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = THREADS, 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
        return cls(session, Tokenizer.from_file(tok), files=(model, tok))

    def another(self) -> OnnxEncoder:
        """A second session and tokenizer of the same checked files (a pool's next encoder)."""
        return type(self).from_files(*self.files)

    def encode(self, texts, batch: int = 32, long_state: str = TRUNC512) -> np.ndarray:
        """The texts' embeddings, a long one read by the rule `long_state` (TRUNC512 or HEAD_TAIL)."""
        if long_state not in (TRUNC512, HEAD_TAIL):
            raise ValueError(f"the encoder reads a long text by {TRUNC512!r} or {HEAD_TAIL!r}, not {long_state!r}")
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            if long_state == TRUNC512:
                enc = self.tokenizer.encode_batch(chunk)
                rows = [(e.ids, e.attention_mask, e.type_ids) for e in enc]
            else:
                rows = [
                    tuple(head_tail(seq) for seq in (e.ids, e.attention_mask, e.type_ids))
                    for e in self.whole.encode_batch(chunk)
                ]
            width = max(len(r[0]) for r in rows)
            feed = {  # padded to the longest of the batch, as the tokenizer pads
                name: np.array([list(r[j]) + [fill] * (width - len(r[j])) for r in rows], dtype=np.int64)
                for j, (name, fill) in enumerate(
                    (("input_ids", self.pad_id), ("attention_mask", 0), ("token_type_ids", 0))
                )
            }
            cls_ = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][:, 0]
            out.append(cls_ / np.maximum(np.linalg.norm(cls_, axis=1, keepdims=True), 1e-12))
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, 768), dtype=np.float32)


def head_tail(seq, limit: int = MAX_SEQ_LENGTH, head: int = HEAD_TOKENS):
    """The HEAD_TAIL selection of one encoded text (a list that starts with [CLS] and ends with [SEP], or any per-token
    list aligned with it): the whole when it fits `limit`, else [CLS], the first `head` tokens, the last
    `limit - 2 - head`, [SEP]."""
    seq = list(seq)
    if len(seq) <= limit:
        return seq
    tail = limit - 2 - head
    return [seq[0], *seq[1 : 1 + head], *seq[len(seq) - 1 - tail : len(seq) - 1], seq[-1]]


class StaticEncoder:
    """A static student (a Model2Vec potion model): `encode(texts)` gives the mean of each text's token vectors,
    L2-normalised, float32, one row each, bit for bit as model2vec 0.10.0's `StaticModel.encode(texts,
    max_length=None)`: the tokenizer with no truncation and no padding, no special tokens, the unknown token dropped, an
    empty text zeros. Thread-safe: nothing is set on it after it is made."""

    def __init__(self, embeddings, tokenizer, name: str | None = None):
        self.name = name
        self.embeddings = np.asarray(embeddings)
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self.tokenizer = tokenizer
        self.unk_id = _unk_token_id(tokenizer)
        if self.embeddings.dtype == np.int8:
            raise TableUnavailable("a quantised static model is not supported")

    @classmethod
    def load(cls, name: str, local_files_only: bool = False) -> StaticEncoder:
        """The static model `name` of STATIC_MODELS, its files fetched from the Hub at the pinned revision (or taken
        from the local cache) and checked against their pinned sha256."""
        spec = STATIC_MODELS[name]
        try:
            import huggingface_hub  # noqa: F401  (student_file fetches with it)
            from safetensors.numpy import load_file
            from tokenizers import Tokenizer
        except ImportError as e:
            raise _need("tokenizers, safetensors and huggingface_hub") from e
        files = list(spec["files_sha256"])
        try:
            paths = {f: student_file(spec["repo"], f, spec["revision"], local_files_only) for f in files}
        except Exception as e:  # no network, no such file, a gated or moved repository
            raise TableUnavailable(
                f"the fast table's static model files cannot be fetched ({type(e).__name__}: {e}); fetch them where "
                f"the server can read them: hf download {spec['repo']} {' '.join(files)} --revision {spec['revision']}"
            ) from e
        for f, path in paths.items():
            digest = file_sha256(path)
            if digest != spec["files_sha256"][f]:
                raise TableUnavailable(
                    f"{path} has sha256 {digest}, not the pinned {spec['files_sha256'][f]}: the fast table refuses it"
                )
        tensors = load_file(paths["model.safetensors"])
        if STATIC_TENSOR not in tensors or set(tensors) != {STATIC_TENSOR}:
            raise TableUnavailable(
                f"{spec['repo']}: model.safetensors holds {sorted(tensors)}, not the one {STATIC_TENSOR!r} matrix"
            )
        tok = Tokenizer.from_file(paths["tokenizer.json"])
        if tok.get_vocab_size() != tensors[STATIC_TENSOR].shape[0]:
            raise TableUnavailable(f"{spec['repo']}: the tokenizer's vocabulary and the embedding matrix differ")
        return cls(tensors[STATIC_TENSOR], tok, name)

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        E = self.embeddings
        out = np.zeros((len(texts), E.shape[1]), dtype=E.dtype)
        for i, enc in enumerate(self.tokenizer.encode_batch(texts, add_special_tokens=False)):
            ids = enc.ids if self.unk_id is None else [t for t in enc.ids if t != self.unk_id]
            if ids:
                out[i] = E[np.array([ids], dtype=np.int64)].mean(axis=1)[0]
        out32 = out.astype(np.float32)
        return out32 / (np.linalg.norm(out32, axis=1, keepdims=True) + 1e-32)


def _unk_token_id(tokenizer) -> int | None:
    """The id of the tokenizer model's unknown token, found as model2vec finds it: the model's `unk_token` (WordPiece,
    BPE, WordLevel), else the Unigram model's `unk_id`."""
    import json

    model = tokenizer.model
    if hasattr(model, "unk_token"):
        return None if model.unk_token is None else tokenizer.token_to_id(model.unk_token)
    return json.loads(tokenizer.to_str())["model"].get("unk_id")


def file_sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def encoder() -> OnnxEncoder:
    """The pinned encoder, loaded once on the CPU with the rest of the pool (ENCODERS sessions of the same files)."""
    global _encoder, _pool
    with _lock:
        if _encoder is None:
            first = OnnxEncoder.load()
            pool = queue.Queue()
            pool.put(first)
            for _ in range(ENCODERS - 1):
                pool.put(first.another() if first.files else first)
            _encoder, _pool = first, pool
    return _encoder


def embed(texts: list[str], long_state: str = TRUNC512) -> np.ndarray:
    """L2-normalised embeddings of the texts by the encoder, float32, one row each, a long text read by `long_state`, on
    a free encoder of the pool (at most ENCODERS encodes run at once; a further one waits for the first that ends)."""
    encoder()
    enc = _pool.get()
    try:
        return np.asarray(enc.encode(list(texts), long_state=long_state), dtype=np.float32)
    finally:
        _pool.put(enc)


def static_model(name: str) -> StaticEncoder:
    """The static student `name` of STATIC_MODELS, loaded once."""
    with _static_lock:
        if name not in _static:
            _static[name] = StaticEncoder.load(name)
        return _static[name]


def static_embed(name: str, texts: list[str]) -> np.ndarray:
    """The static student `name`'s embeddings of the texts, float32, one row each."""
    return np.asarray(static_model(name).encode(list(texts)), dtype=np.float32)


# ---- the student kinds and their feature functions ------------------------------------------------------------------


def parse_request(value) -> dict | None:
    """The fast table a registration asks for, or None: `true` is the default kind; an object names the kind
    (`KINDS`), for `static` or `auto` the static model (`STATIC_MODELS`), and the saving the table must time
    (`min_saving`, `parse_min_saving`; `MIN_SAVING` when not given, and always in the result, so the task records the
    bar it was registered under). ValueError says what is wrong."""
    if value is None or value is False:
        return None
    if value is True:
        return {"kind": DEFAULT_KIND, "static_model": DEFAULT_STATIC_MODEL, "min_saving": MIN_SAVING}
    shape = (
        f"fast_table: true or false, or an object {{'kind': {' | '.join(map(repr, KINDS))}, "
        f"'static_model': {' | '.join(map(repr, STATIC_MODELS))}, 'min_saving': <a share>}}"
    )
    if not isinstance(value, dict):
        raise ValueError(shape)
    extra = sorted(set(value) - {"kind", "static_model", "min_saving"})
    if extra:
        raise ValueError(f"{shape}; not {', '.join(map(repr, extra))}")
    ms = parse_min_saving(value.get("min_saving", MIN_SAVING))
    kind = value.get("kind", DEFAULT_KIND)
    if kind in RESERVED_KINDS:
        raise ValueError(
            f"fast_table.kind: {kind!r} is not available in this version (the kinds are {', '.join(map(repr, KINDS))})"
        )
    if kind not in KINDS:
        raise ValueError(f"fast_table.kind: one of {', '.join(map(repr, KINDS))}, not {kind!r}")
    if kind == "encoder":
        if "static_model" in value:
            raise ValueError("fast_table.static_model: for the 'static' and 'auto' kinds only")
        return {"kind": kind, "min_saving": ms}
    sm = value.get("static_model", DEFAULT_STATIC_MODEL)
    if sm not in STATIC_MODELS:
        raise ValueError(f"fast_table.static_model: one of {', '.join(map(repr, STATIC_MODELS))}, not {sm!r}")
    return {"kind": kind, "static_model": sm, "min_saving": ms}


def parse_min_saving(value) -> float:
    """A task's `min_saving`: the share of the expected time per request its table must save to be kept, and to stay
    on under load, within `MIN_SAVING_RANGE` inclusive. ValueError says what is wrong."""
    lo, hi = MIN_SAVING_RANGE
    try:  # an integer too large for a float overflows in isfinite: not a number this setting can take
        number = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        number = False
    if not number or not lo <= value <= hi:
        raise ValueError(
            f"fast_table.min_saving: the share of the expected time per request the table must save to be kept, a "
            f"number from {lo:g} to {hi:g} (default {MIN_SAVING:g}), not {value!r}"
        )
    return float(value)


def record_min_saving(rec: dict | None) -> float:
    """The bar a stored table (or a learning task's request) is held to: its own `min_saving`, or for one recorded
    before the setting existed the `LEGACY_MIN_SAVING` it was kept under."""
    if rec and rec.get("min_saving") is not None:
        return float(rec["min_saving"])
    return LEGACY_MIN_SAVING


def student_features(kind: str, static_model: str = DEFAULT_STATIC_MODEL) -> dict:
    """The feature function this server fits a new table of `kind` ("static" or "encoder") with."""
    if kind == "static":
        spec = STATIC_MODELS[static_model]
        return {
            "kind": "static",
            "model": spec["repo"],
            "revision": spec["revision"],
            "files_sha256": dict(spec["files_sha256"]),
            "long_state": WHOLE,
        }
    if kind == "encoder":
        return {
            "kind": "encoder",
            "model": ENCODER,
            "revision": REVISION,
            "files_sha256": {ONNX_FILE: ONNX_SHA256, TOKENIZER_FILE: TOKENIZER_SHA256},
            "long_state": HEAD_TAIL,
        }
    raise ValueError(f"no student of kind {kind!r}")


def requested_features(request: dict) -> list[dict]:
    """The feature functions a registration's request (`parse_request`) fits: one, or for `auto` the static then the
    encoder."""
    if request["kind"] == "auto":
        return [student_features("static", request["static_model"]), student_features("encoder")]
    return [student_features(request["kind"], request.get("static_model", DEFAULT_STATIC_MODEL))]


def legacy_encoder_name(features: dict | None) -> str:
    """The `encoder` field of a new table record: the student's model, and for an encoder read with another long-state
    rule than TRUNC512 the model and the rule ("BAAI/bge-base-en-v1.5#head256+tail254"), which no earlier server has."""
    if not features:
        return ENCODER
    if features["kind"] == "encoder" and features["long_state"] != TRUNC512:
        return f"{features['model']}#{features['long_state']}"
    return features["model"]


def record_features(rec: dict) -> dict:
    """The feature function of a stored table: its own, or for a table recorded before the kinds (no "features") the
    encoder it names with the TRUNC512 rule it was made with."""
    if rec.get("features"):
        return rec["features"]
    return {
        "kind": "encoder",
        "model": rec["encoder"],
        "revision": rec["revision"],
        "files_sha256": None,  # not recorded then; the model and revision pin the files
        "long_state": TRUNC512,
    }


def static_name(feat: dict) -> str | None:
    """The name in STATIC_MODELS of a static feature function this server computes exactly, or None."""
    for name, spec in STATIC_MODELS.items():
        if (spec["repo"], spec["revision"], spec["files_sha256"]) == (
            feat.get("model"),
            feat.get("revision"),
            feat.get("files_sha256"),
        ):
            return name
    return None


def cannot_compute(feat: dict) -> str | None:
    """Why this server cannot compute a table's feature function exactly (another model, revision, file or rule), or
    None when it can."""
    kind = feat.get("kind")
    if kind == "encoder":
        files = {ONNX_FILE: ONNX_SHA256, TOKENIZER_FILE: TOKENIZER_SHA256}
        if (feat.get("model"), feat.get("revision")) != (ENCODER, REVISION) or feat.get("files_sha256") not in (
            None,
            files,
        ):
            return (
                f"fitted with {feat.get('model')} at revision {str(feat.get('revision'))[:8]}; this server has "
                f"{ENCODER} at {REVISION[:8]}: register the task again"
            )
        if feat.get("long_state") not in (TRUNC512, HEAD_TAIL):
            return f"fitted with the long-state rule {feat.get('long_state')!r}, which this server does not compute"
        return None
    if kind == "static":
        if static_name(feat) is None or feat.get("long_state") != WHOLE:
            have = ", ".join(f"{s['repo']} at {s['revision'][:8]}" for s in STATIC_MODELS.values())
            return (
                f"fitted with the static model {feat.get('model')} at revision {str(feat.get('revision'))[:8]}; this "
                f"server has {have}: register the task again"
            )
        return None
    return f"fitted with features of kind {kind!r}, which this server does not compute: register the task again"


def load_student(feat: dict) -> None:
    """Load the student of a feature function this server computes (TableUnavailable when its files or its runtime
    cannot be had)."""
    if feat["kind"] == "encoder":
        encoder()
    else:
        static_model(static_name(feat))


def featurize(feat: dict, texts: list[str]) -> np.ndarray:
    """The features of the texts by a table's feature function (one `cannot_compute` passes)."""
    if feat["kind"] == "encoder":
        return embed(texts, long_state=feat["long_state"])
    return static_embed(static_name(feat), texts)


def require_fit() -> None:
    """TableUnavailable when the fit's library (scikit-learn) is missing."""
    try:
        import sklearn  # noqa: F401
    except ImportError as e:
        raise _need("scikit-learn") from e


def choose(candidates: list[dict], min_saving: float = MIN_SAVING) -> tuple[int | None, str]:
    """auto's rule: (the index of the kind kept, or None, and why). Each candidate has `kind`, `holds` (a table
    threshold holds) and `saving` (the timing guard's expected saving). Eligible: its threshold holds and it saves at
    least `min_saving`. The eligible one with the largest saving is kept; when a static one is within TIE of the
    largest, the static one is."""
    ok = [i for i, c in enumerate(candidates) if c["holds"] and c["saving"] >= min_saving]
    if not ok:
        return None, (
            f"no kind has a table threshold that holds and an expected saving of at least {min_saving:.0%}: no table"
        )
    best = max(ok, key=lambda i: candidates[i]["saving"])
    top = candidates[best]["saving"]

    def near(i):
        return round(top - candidates[i]["saving"], 6) <= TIE

    statics = [i for i in ok if candidates[i]["kind"] == "static" and near(i)]
    keep = max(statics, key=lambda i: candidates[i]["saving"]) if statics else best
    tied = [i for i in ok if i != keep and near(i) and candidates[i]["kind"] != "static"]
    if tied and candidates[keep]["kind"] == "static":
        others = ", ".join(f"{candidates[i]['kind']} {candidates[i]['saving']:.1%}" for i in tied)
        return keep, (
            f"static saves {candidates[keep]['saving']:.1%} and {others}, within {TIE * 100:.0f} point: static is kept "
            "on a tie"
        )
    others = [f"{candidates[i]['kind']} {candidates[i]['saving']:.1%}" for i in ok if i != keep]
    return keep, (
        f"{candidates[keep]['kind']} has the largest expected saving ({top:.1%}"
        + (f", against {', '.join(others)}" if others else "")
        + f") of the kinds whose threshold holds and that save at least {min_saving:.0%}"
    )


def _pack(a) -> dict:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"dtype": "float32", "shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode()}


def unpack(d) -> np.ndarray:
    if isinstance(d, np.ndarray):
        return d
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.float32).reshape(d["shape"]).copy()


def fit(X: np.ndarray, y, k: int, features: dict | None = None, sample_weight=None) -> dict:
    """The stored table from features X (n x d) and gold option indices y (below k): a multinomial logistic regression
    with L2, the strength chosen from `CS` by 5-fold stratified cross-validated log loss. Needs scikit-learn. With
    `features` (the feature function X was made with) the record carries it, and its model and revision in "encoder"
    and "revision"; without it the record names the encoder, as before the kinds. `sample_weight`: each row's weight
    in the fit and in the cross-validated loss (a table learned from traffic weights a row by one over the chance it
    was logged); None weighs every row 1, as before."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError as e:
        raise TableUnavailable("the fast table needs the `table` extra: pip install 'decisio[table]'") from e
    y = np.asarray(y, dtype=int)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)

    def fitted(c, Xa, ya, wa):
        m = LogisticRegression(C=c, max_iter=2000)
        m.fit(Xa, ya, sample_weight=None if sample_weight is None else wa)
        return m

    def probs(m, Xb):
        P = np.zeros((len(Xb), k))
        P[:, m.classes_] = m.predict_proba(Xb)
        return P

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # classes with fewer than 5 members in a fold; convergence notes
        folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y))
        loss = {}
        for c in CS:
            ll = []
            for tr, te in folds:
                P = probs(fitted(c, X[tr], y[tr], w[tr]), X[te])
                nll = -np.log(np.clip(P[np.arange(len(te)), y[te]], 1e-12, 1))
                ll.append(nll.mean() if sample_weight is None else float(np.average(nll, weights=w[te])))
            loss[c] = float(np.mean(ll))
        best = min(CS, key=lambda c: (loss[c], c))
        m = fitted(best, X, y, w)
    rec = {
        "format": FORMAT,
        # an earlier server checks only `encoder` and `revision`: an encoder table read with the head-and-tail rule
        # names its model with the rule, so a server that would read it with its own 512-token cut finds another
        # encoder, serves the task through decisio and says so, rather than answer on features not certified
        "encoder": legacy_encoder_name(features),
        "revision": features["revision"] if features else REVISION,
        "k": k,
        "classes": [int(c) for c in m.classes_],
        "binary": bool(m.coef_.shape[0] == 1),
        "coef": _pack(m.coef_),
        "intercept": _pack(m.intercept_),
        "C": best,
        "cv_logloss": {str(c): round(v, 5) for c, v in loss.items()},
        "n_examples": int(len(y)),
    }
    if features:
        rec["features"] = dict(features)
    return rec


def predict(rec: dict, X: np.ndarray) -> np.ndarray:
    """(n x k) probabilities of the stored table for embeddings X; an option the table never saw gets 0. A record with a
    calibration (a yes/no table's, `cross_fit_temperature`) divides the logits by its temperature first."""
    coef, b = unpack(rec["coef"]).astype(np.float64), unpack(rec["intercept"]).astype(np.float64)
    z = np.asarray(X, dtype=np.float64) @ coef.T + b
    z = z / temperature_of(rec)
    if rec["binary"]:
        p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
        q = np.stack([1.0 - p1, p1], axis=1)
    else:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        q = e / e.sum(axis=1, keepdims=True)
    P = np.zeros((len(q), rec["k"]))
    P[:, rec["classes"]] = q
    return P


def confidence(P: np.ndarray):
    """(confidence, prediction) of each row: the top probability and its option."""
    return P.max(axis=1), P.argmax(axis=1)


# ---- a yes/no table's probability -----------------------------------------------------------------------------------


def temperature_of(rec: dict) -> float:
    """The temperature a stored table's logits are divided by: its calibration's, or 1 (a choice table, and every table
    recorded before yes/no tables)."""
    cal = rec.get("calibration")
    return float(cal["temperature"]) if cal else 1.0


def temperature_for(z, yes, weights=None) -> tuple[float, float, float]:
    """(T, log loss at T = 1, log loss at T) for logits `z` of yes against the reference answers `yes` (true where the
    reference is yes), each row weighted by `weights` (None: 1): the T in `TEMPERATURES` that minimises the log loss of
    sigmoid(z / T) against Platt's (1999) targets, (N+ + 1) / (N+ + 2) for a yes and 1 / (N- + 2) for a no, which keep T
    finite where the logits separate the answers perfectly. Golden section on log T (the loss is convex in 1 / T)."""
    z = np.asarray(z, dtype=np.float64)
    yes = np.asarray(yes, dtype=bool)
    w = np.ones(len(z)) if weights is None else np.asarray(weights, dtype=np.float64)
    n_yes, n_no = int(yes.sum()), int((~yes).sum())
    target = np.where(yes, (n_yes + 1) / (n_yes + 2), 1.0 / (n_no + 2))

    def loss(log_t):
        s = z / math.exp(log_t)
        # -[t log sigmoid(s) + (1 - t) log sigmoid(-s)], computed stably
        return float(np.average(np.logaddexp(0.0, -s) * target + np.logaddexp(0.0, s) * (1.0 - target), weights=w))

    lo, hi = math.log(TEMPERATURES[0]), math.log(TEMPERATURES[1])
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(80):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
    log_t = (a + b) / 2
    return math.exp(log_t), loss(0.0), loss(log_t)


def cross_fit_temperature(X: np.ndarray, y, rec: dict, sample_weight=None, reference: str = "gold") -> dict:
    """The calibration of a two-class (yes/no) table, fitted on its FIT examples by cross-fitting, never on the examples
    its certificate is made on: the fit examples X (features) and y (their answer indices, 0 for yes) are split into K
    deterministic stratified folds (`FOLDS`, fewer when the smaller answer has fewer examples), a classifier of the
    table's own strength (`rec["C"]`, chosen on the fit examples) is fitted on K - 1 folds, its logit for yes is taken
    on the fold left out, and one temperature is fitted on all the out-of-fold logits against their answers
    (`temperature_for`), each weighted by `sample_weight` as in the table's fit. `reference` says what the answers are:
    "gold" (labelled examples), "decisio" (a learned table's logged rows: decisio's answers) or "decisio+gold".
    The served table stays the one fitted on all the fit examples (`rec`); P(yes) = sigmoid(logit / T). One parameter
    and no bias: the table's answer and the order of its confidences are unchanged. With fewer than two examples of
    either answer there is nothing to cross-fit, and T is 1. Needs scikit-learn."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    yes = y == 0
    folds = min(FOLDS, int(yes.sum()), int((~yes).sum()))
    out = {
        "method": "cross_fit",
        "folds": folds,
        "fitted_on": "fit",
        "reference": reference,
        "n": int(len(y)),
        "targets": "platt",
        "rule": "P(yes) = sigmoid(logit / temperature); the temperature is fitted by log loss against Platt's targets "
        "on out-of-fold logits of the fit examples (each fold's logit from a classifier fitted on the other folds), so "
        "it is fixed before any certification example is read; the answer and the order of the confidences are "
        "unchanged",
    }
    if folds < 2:
        return {**out, "folds": 0, "temperature": 1.0, "why": "fewer than two fit examples of one answer"}
    z = np.zeros(len(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence notes
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(X, y):
            m = LogisticRegression(C=rec["C"], max_iter=2000)
            m.fit(X[tr], y[tr], sample_weight=None if w is None else w[tr])
            score = m.decision_function(X[te])  # the logit of classes_[1], the answer no
            z[te] = -score if list(m.classes_) == [0, 1] else score
    t, before, after = temperature_for(z, yes, w)
    return {**out, "temperature": round(t, 6), "log_loss_before": round(before, 6), "log_loss_after": round(after, 6)}


def top_confidence(P: np.ndarray, commit_band=None) -> np.ndarray:
    """The top probability of each row of P (n x k): the confidence a certificate is read on. `commit_band` (lo, hi),
    for a yes/no table on a server whose --noul-commit rule is on: the top probability after the rule, as decisio's
    certified yes/no answers are read (a P(yes) strictly inside the band is reported at the band's edge on its own side,
    hi above 0.5 and lo otherwise)."""
    P = np.asarray(P, dtype=np.float64)
    if commit_band is None:
        return P.max(axis=1)
    lo, hi = commit_band
    py = P[:, 0]
    inside = (py > lo) & (py < hi)
    pc = np.where(inside, np.where(py > 0.5, hi, lo), py)
    return np.maximum(pc, 1.0 - pc)


# ---- the two-stage certificate ---------------------------------------------------------------------------------------


def cascade(cT, pT, y, cD, pD, alpha: float, delta: float) -> dict:
    """The thresholds of the cascade on the held-out examples: the table's `(cT, pT)` and decisio's `(cD, pD)`
    confidences and predictions against the gold options y. The table's threshold at (alpha, delta/2); decisio's on the
    requests the table defers, at (alpha, delta/2). A stage with no threshold that holds is None."""
    cT, pT, y, cD, pD = (np.asarray(a) for a in (cT, pT, y, cD, pD))
    tau_t = certify.threshold(cT, pT == y, alpha, delta / 2)
    table = cT >= tau_t if np.isfinite(tau_t) else np.zeros(len(y), bool)
    deferred = ~table
    tau_d = certify.threshold(cD[deferred], pD[deferred] == y[deferred], alpha, delta / 2) if deferred.any() else np.inf
    return {
        "table_threshold": float(tau_t) if np.isfinite(tau_t) else None,
        "decisio_threshold": float(tau_d) if np.isfinite(tau_d) else None,
        "table_share": float(table.mean()) if len(y) else 0.0,
        "n": int(len(y)),
    }


def table_stage(cT, pT, y, threshold, delta: float) -> dict:
    """The table's own stage on the held-out examples at its threshold: the share it answers, its errors among them and
    their upper bound at delta/2 (as `certificate` reports it)."""
    cT, pT, y = (np.asarray(a) for a in (cT, pT, y))
    by_table = cT >= threshold if threshold is not None else np.zeros(len(y), bool)
    answered, wrong = int(by_table.sum()), int((by_table & (pT != y)).sum())
    return {
        "table_share": float(by_table.mean()) if len(y) else 0.0,
        "answered": answered,
        "errors": wrong,
        "error_rate": wrong / answered if answered else None,
        "upper_bound": certify.upper_bound(wrong, answered, delta / 2) if answered else None,
    }


def served(cT, pT, cD, pD, thresholds: dict):
    """(answered by the table, certified, predicted option) for held-out or live confidences under the thresholds of
    `cascade`."""
    cT, pT, cD, pD = (np.asarray(a) for a in (cT, pT, cD, pD))
    tt, td = thresholds["table_threshold"], thresholds["decisio_threshold"]
    by_table = cT >= tt if tt is not None else np.zeros(len(cT), bool)
    certified = by_table | (~by_table & (cD >= td if td is not None else np.zeros(len(cT), bool)))
    return by_table, certified, np.where(by_table, pT, pD)


def certificate(alpha: float, delta: float, thresholds: dict, by_table, certified, correct) -> dict:
    """The numbers of the cascade's certification on the held-out examples, in the fields of a single-stage certificate:
    `threshold` is decisio's (on the requests the table defers), `table` the table's, the counts are the cascade's."""
    by_table, certified, correct = (np.asarray(a, dtype=bool) for a in (by_table, certified, correct))
    n = len(correct)
    stages = {}
    for name, mask, thr in (
        ("table", by_table, thresholds["table_threshold"]),
        ("decisio", certified & ~by_table, thresholds["decisio_threshold"]),
    ):
        answered = int(mask.sum())
        wrong = int((mask & ~correct).sum())
        stages[name] = {
            "threshold": thr,
            "certified_n": answered,
            "errors": wrong,
            "upper_bound": certify.upper_bound(wrong, answered, delta / 2) if answered else None,
        }
    cert_n, wrong = int(certified.sum()), int((certified & ~correct).sum())
    return {
        "procedure": certify.PROCEDURE
        + ", two stages (the table's threshold, then decisio's on what the table defers)",
        "alpha": alpha,
        "delta": delta,
        "n": n,
        "accuracy": float(correct.mean()) if n else None,
        "threshold": thresholds["decisio_threshold"],
        "certified": thresholds["table_threshold"] is not None,
        "table_threshold": thresholds["table_threshold"],
        "certified_share": cert_n / n if n else 0.0,
        "certified_n": cert_n,
        "errors_at_threshold": wrong,
        "error_rate_at_threshold": wrong / cert_n if cert_n else None,
        "upper_bound_at_threshold": None,
        "empirical_share_at_alpha": cert_n / n if n else 0.0,
        "stages": stages,
    }


def summary(c: dict, table_share: float) -> str:
    """The cascade's certificate in words."""
    st = c["stages"]
    return (
        f"Answers of the fast table whose top probability is at least {c['table_threshold']:.4f}, and answers of "
        f"decisio "
        f"(on the requests the table defers) whose top probability is at least "
        + (f"{c['threshold']:.4f}" if c["threshold"] is not None else "(no threshold holds: none)")
        + f", are certified: on the {c['n']} held-out examples that is {c['certified_share']:.0%} of the answers, "
        f"{c['errors_at_threshold']} of {c['certified_n']} wrong ({st['table']['certified_n']} answered by the table, "
        f"{st['table']['errors']} wrong; {st['decisio']['certified_n']} by decisio, {st['decisio']['errors']} wrong). "
        "The table's stage keeps its promise by the procedure's proof: with probability at least "
        f"{1 - c['delta'] / 2:.0%} over the draw of such examples, at most {c['alpha']:.0%} of its certified answers "
        "are wrong. Decisio's stage is certified on the requests the table's chosen threshold defers, from the same "
        "held-out examples, so its promise is supported by simulation (about 1 draw in 100 broke it), not proved. "
        "Both are for traffic like the examples and on this serving stack. "
        f"The table answers {table_share:.0%} of requests without asking the model."
    )


def timing_guard(table_ms, decisio_ms, table_share: float, min_saving: float = MIN_SAVING, deferred=None) -> dict:
    """Whether the table pays: the expected mean time per request with the table in front against decisio's own, both
    measured on the same requests one at a time. `table_ms` and `decisio_ms` are per-request milliseconds. With
    `deferred` (which requests the table defers) the expected time is the mean of the table's time plus decisio's on
    each deferred request, so a deferral of the slow requests is counted as it is; without it, the table's mean time
    plus decisio's mean on the share `table_share` it defers."""
    t, d = float(np.mean(table_ms)), float(np.mean(decisio_ms))
    if deferred is not None:
        expected = float(
            np.mean(np.asarray(table_ms, dtype=float) + np.asarray(deferred) * np.asarray(decisio_ms, dtype=float))
        )
    else:
        expected = t + (1.0 - table_share) * d
    saving = 1.0 - expected / d if d > 0 else 0.0
    return {
        "kept": bool(saving >= min_saving),
        "saving": round(saving, 4),
        "required": min_saving,
        "table_ms_mean": round(t, 2),
        "decisio_ms_mean": round(d, 2),
        "table_share": round(table_share, 4),
        "expected_ms_mean": round(expected, 2),
    }


# ---- one replica's use of the table -----------------------------------------------------------------------------


def cpu_model() -> str:
    """The host CPU's model name, as the host gives it."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5, check=False
        ).stdout.strip()
        if out:
            return out
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def timing_stack(engine_facts: dict, features: dict | None = None) -> dict:
    """What a timing of the table against decisio is true for: the host CPU, the engine and its card, and the student.
    The encoder (`features` of kind "encoder", or None): its model and revision, the ONNX file's hash, the runtime's
    version, its threads and the pool, the same fields as before the kinds, so a table timed then stays timed. A static
    student: its kind, model and revision and the libraries that compute it (no ONNX facts, threads or pool). A replica
    whose own differs does not trust the timing."""
    from importlib.metadata import PackageNotFoundError, version

    def lib(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    keep = ("engine", "vllm", "mlx", "mlx_lm", "gpu", "device", "machine", "quantization")
    host = {
        "cpu": cpu_model(),
        "cores": os.cpu_count(),
        "machine": platform.machine(),
        "engine": {k: engine_facts[k] for k in keep if k in engine_facts},
    }
    if features is not None and features["kind"] == "static":
        return {
            **host,
            "kind": "static",
            "model": features["model"],
            "revision": features["revision"],
            "runtime": "numpy",
            "numpy": lib("numpy"),
            "tokenizers": lib("tokenizers"),
        }
    return {
        **host,
        "encoder": ENCODER,
        "revision": REVISION,
        "runtime": "onnxruntime-fp32",
        "onnxruntime": lib("onnxruntime"),
        "tokenizers": lib("tokenizers"),
        "onnx_sha256": ONNX_SHA256[:16],
        "threads": THREADS,
        "encoders": ENCODERS,
    }


def stack_differences(a: dict, b: dict) -> list[str]:
    """The dotted names of the facts on which two timing stacks differ."""
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out += [f"{k}.{d}" for d in stack_differences(x, y)]
        elif x != y:
            out.append(k)
    return out


def agreement_baseline(disagree, seed: str) -> dict | None:
    """The record of a CUSUM on how often decisio's answer differs from the table's, for the held-out examples the table
    answered (`disagree` flags each one); None when there are too few of them, or the rate is too high to tell a rise
    from normal. The same rule as the monitor's other rate signals."""
    d = np.asarray(disagree, dtype=bool)
    if len(d) < 20:
        return None
    p0 = max(float(d.mean()), 1.0 / len(d))
    p_ok, p_bad = monitor.OK_FACTOR * p0, monitor.BAD_FACTOR * p0
    if p_bad > 0.5:
        return None
    return {
        "share": p0,
        "p_ok": p_ok,
        "p_bad": p_bad,
        "limit": monitor.cusum_limit(p_ok, p_bad, "agreement:" + seed),
        "false_alarm": monitor.FALSE_ALARM,
        "horizon": monitor.HORIZON,
    }


class Agreement:
    """A CUSUM on sampled table answers that decisio would have answered otherwise."""

    def __init__(self, base: dict):
        self.base = base
        self._up = math.log(base["p_bad"] / base["p_ok"])
        self._down = math.log((1 - base["p_bad"]) / (1 - base["p_ok"]))
        self.statistic, self.alerting, self.n, self.disagreed = 0.0, False, 0, 0

    def observe(self, disagree: bool) -> None:
        self.n += 1
        self.disagreed += bool(disagree)
        lim = self.base["limit"]
        self.statistic = min(8 * lim, max(0.0, self.statistic + (self._up if disagree else self._down)))
        if self.statistic > lim:
            self.alerting = True
        elif self.statistic < lim / 2:
            self.alerting = False

    def facts(self) -> dict:
        return {
            "statistic": self.statistic,
            "limit": self.base["limit"],
            "registered_disagreement": self.base["share"],
            "sampled": self.n,
            "disagreed": self.disagreed,
            "alerting": self.alerting,
        }


class Runtime:
    """One replica's use of one task's table: whether it answers, and why not when it does not.

    `active`: the table answers the requests it is sure of. `timing`: the table is not used and is being timed against
    decisio on this replica's own requests (the timing at registration was made on another stack, or the table stopped
    beating decisio under this replica's load); it comes back when 30 paired timings show the saving. `suspended`: the
    monitor's alarm fired (on this replica or another that shares the task store). `unavailable`: this server cannot run
    the table (it computes another feature function: another student, revision or long-state rule; or the `table` extra
    is missing). Not thread-safe by itself: the server calls it from the request path, one task at a time."""

    def __init__(self, rec: dict, stack: dict):
        self.rec = rec
        self.stack = stack
        self.min_saving = record_min_saving(rec)  # the bar it was kept under, which it must keep under load
        self.stack_sha = sha256(stack)
        self.share = rec["thresholds"]["table_share"]
        self.registered_ms = rec["timing"]["guard"]["decisio_ms_mean"]
        self.table_ms: collections.deque = collections.deque(maxlen=WINDOW)
        self.pairs: list[tuple[float, float]] = []
        self.requests = 0
        self.table_answers = 0
        self.last_guard = None
        self.retimed = None
        self.agreement = Agreement(rec["agreement"]) if rec.get("agreement") else None
        self.state, self.reason = "active", None
        self.features = record_features(rec)
        why = cannot_compute(self.features)
        if why:
            self.state, self.reason = "unavailable", why
        elif rec["timing"]["stack_sha256"] != self.stack_sha:
            self.state = "timing"
            diff = stack_differences(rec["timing"]["stack"], stack)
            self.reason = (
                f"the table was timed on another stack ({', '.join(diff)}): serving through decisio while this "
                "replica times the table on its own requests"
            )
        sus = rec.get("suspended")
        if sus:
            self.state, self.reason = "suspended", f"suspended by {sus.get('by', 'a replica')}: {sus.get('reason')}"

    # state changes

    def suspend(self, reason: str) -> None:
        self.state, self.reason = "suspended", reason

    def unavailable(self, reason: str) -> None:
        self.state, self.reason = "unavailable", reason

    @property
    def uses_table(self) -> bool:
        return self.state == "active"

    # the table's own latency, while it answers

    def table_answered(self, ms: float) -> None:
        """A request answered by the table in `ms`. Falls back to decisio when the rolling median no longer pays: the
        expected time per request with the table (its median on every request, decisio's registered mean on the share it
        defers) must still be the table's `min_saving` below decisio's registered mean."""
        self.table_answers += 1
        self.table_ms.append(ms)
        if len(self.table_ms) < MIN_LATENCY_SAMPLES:
            return
        p50 = float(np.median(self.table_ms))
        expected = p50 + (1.0 - self.share) * self.registered_ms
        bar = (1.0 - self.min_saving) * self.registered_ms
        if expected > bar:
            self.state = "timing"
            self.reason = (
                f"under this replica's load the table's median is {p50:.0f} ms against decisio's registered "
                f"{self.registered_ms:.0f} ms, so the expected time per request is {expected:.0f} ms, not "
                f"{bar:.0f} ms or less (a saving of {self.min_saving:.0%} required): serving through decisio while the "
                "table is timed again on this replica's requests"
            )
            self.table_ms.clear()
            self.pairs.clear()

    # re-timing, while the table is off for speed

    def shadow_due(self) -> bool:
        """Counts a request answered by decisio while the table is being timed; True when this one is to time the table
        too."""
        if self.state != "timing":
            return False
        self.requests += 1
        return self.requests % SHADOW_EVERY == 0

    def shadowed(self, table_ms: float, decisio_ms: float) -> None:
        """One request timed both ways: the table's time (embedding and classifier) and decisio's for the request."""
        self.pairs.append((table_ms, decisio_ms))
        if len(self.pairs) < SHADOW_N:
            return
        g = timing_guard([a for a, _ in self.pairs], [b for _, b in self.pairs], self.share, self.min_saving)
        self.last_guard = g
        self.pairs.clear()
        if g["kept"]:
            self.state, self.reason, self.retimed = "active", None, g
            self.table_ms.clear()

    # sampled agreement, while the table answers

    def sample_due(self) -> bool:
        return self.state == "active" and self.agreement is not None and self.table_answers % SAMPLE_EVERY == 0

    def facts(self) -> dict:
        out = {
            "state": self.state,
            "uses_table": self.uses_table,
            "table_share_registered": self.share,
            "min_saving": self.min_saving,
            "timed_on": self.rec["timing"]["stack_sha256"][:12],
            "this_stack": self.stack_sha[:12],
        }
        if self.reason:
            out["reason"] = self.reason
        if self.table_ms:
            out["table_ms_p50"] = round(float(np.median(self.table_ms)), 2)
            out["table_ms_samples"] = len(self.table_ms)
        if self.retimed:
            out["retimed"] = self.retimed
        if self.last_guard and not self.retimed:
            out["last_timing"] = self.last_guard
        if self.state == "timing":
            out["timing_progress"] = f"{len(self.pairs)} of {SHADOW_N} paired timings"
        if self.agreement:
            out["agreement"] = self.agreement.facts()
        else:
            out["agreement"] = {"armed": False, "why": "too few held-out table answers to tell a rise in disagreement"}
        return out
