# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The role tagger's interface, a stub for tests, and the wrapper for a token-classification encoder.

  Tagger                 the protocol: predict(state, context=None) -> Tags
  StubTagger(tags)       fixed answers: a mapping from the state text to its Tags, or a function
  HFTagger.load(path, revision=None)
                         an encoder (ModernBERT-base, DeBERTa-v3-small or any Hugging Face encoder) with the heads of
                         `decisio.calculator.cascade.model`, from a local directory or a Hub repository at a pinned
                         commit; needs torch and transformers, imported only here
  decode_bio(offsets, label_ids, probs, labels) -> [(start, end, label, p)]
                         character spans from per-token BIO labels

Determinism: the wrapper reads one text at a time (batch size 1), in eval mode, in float32 on the CPU unless told
otherwise, with fixed weights (a directory's `SHA256SUMS` is checked when present; a Hub repository needs a full commit
hash). A text longer than the model's window gives `Tags(note="too-long")`, never a truncated read.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Mapping, Protocol, runtime_checkable

from decisio.calculator.cascade.tags import Tags


@runtime_checkable
class Tagger(Protocol):
    def predict(self, state: str, context: str | None = None) -> Tags: ...


class StubTagger:
    """Fixed tags: `answers` maps a state text to its Tags (a text not in it gives `Tags(note="unknown-text")`), or is
    a function of (state, context)."""

    def __init__(self, answers: Mapping[str, Tags] | Callable[[str, str | None], Tags]):
        self.answers = answers

    @classmethod
    def from_rows(cls, rows) -> StubTagger:
        """The labelled tags of training-format rows, keyed by state: an oracle tagger."""
        return cls({r["state"]: Tags.from_row(r) for r in rows})

    def predict(self, state: str, context: str | None = None) -> Tags:
        if callable(self.answers):
            return self.answers(state, context)
        return self.answers.get(state) or Tags(spans=(), note="unknown-text")


def decode_bio(offsets, label_ids, probs, labels, text: str | None = None) -> list[tuple[int, int, str, float]]:
    """Character spans from token labels. `offsets` are (start, end) per token ((0, 0) for special tokens),
    `label_ids` the chosen label per token, `probs` its probability, `labels` the names (`O`, `B-role`, `I-role`).
    An `I-` token continues the open span of the same role, else starts one; with `text`, a `B-` token of the open
    span's role that continues the same word (it starts where the span ends, on a character that is not a space:
    `$45` `.` `44`) continues it too, since one word is one value. A span's probability is the lowest of its tokens'.
    Leading and trailing whitespace inside a token's offsets is not part of the span (the caller trims)."""
    spans: list[list] = []
    open_role = None
    for (s, e), lid, p in zip(offsets, label_ids, probs):
        if s == e:
            open_role = None
            continue
        name = labels[lid]
        if name == "O":
            open_role = None
            continue
        tag, role = name.split("-", 1)
        same_word = text is not None and spans and spans[-1][1] == s and not text[s].isspace()
        if open_role == role and (tag == "I" or same_word):
            spans[-1][1] = e
            spans[-1][3] = min(spans[-1][3], p)
        else:
            spans.append([s, e, role, p])
            open_role = role
    return [(s, e, r, float(p)) for s, e, r, p in spans]


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


_SHA = re.compile(r"[0-9a-f]{40}")


def check_sums(directory: Path) -> None:
    """Every file `SHA256SUMS` lists has that hash; raises ValueError on a mismatch or a missing file."""
    sums = directory / "SHA256SUMS"
    if not sums.exists():
        return
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = directory / name.lstrip("*")
        if not path.is_file():
            raise ValueError(f"tagger weights: {name} is listed in SHA256SUMS but missing")
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            raise ValueError(f"tagger weights: {name} does not match SHA256SUMS")


def resolve(path: str, revision: str | None = None) -> Path:
    """A local model directory, or a Hub repository downloaded at a full commit hash (never a branch or a tag)."""
    p = Path(path)
    if p.is_dir():
        return p
    if revision is None or not _SHA.fullmatch(revision):
        raise ValueError("a Hub tagger needs revision= a full 40-character commit hash, so the weights cannot change")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path, revision=revision))


class HFTagger:
    """A trained tagger directory (written by `model.save`): the encoder, its tokenizer and the heads."""

    def __init__(self, model, tokenizer, meta: dict, device: str = "cpu"):
        self.model, self.tokenizer, self.meta, self.device = model, tokenizer, meta, device

    @classmethod
    def load(cls, path: str, revision: str | None = None, device: str = "cpu", dtype: str = "float32") -> HFTagger:
        from decisio.calculator.cascade import model as M

        directory = resolve(path, revision)
        check_sums(directory)
        model, tokenizer, meta = M.load(directory, device, dtype)
        return cls(model, tokenizer, meta, device)

    def predict(self, state: str, context: str | None = None) -> Tags:
        from decisio.calculator.cascade import model as M

        return M.predict(self.model, self.tokenizer, self.meta, state, context, self.device)


def read_meta(directory: Path) -> dict:
    return json.loads((Path(directory) / "cascade_tagger.json").read_text())
