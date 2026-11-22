# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The readout's hidden state at serving time, for the per-task intent head (decisio.readout.intent_head).

The head needs h, the final-norm hidden state at the readout position (the vector the letter logits are read from),
which vLLM's generate path does not return. The serving engine runs decisio's hidden-readout class, and h is read one
of two ways (`vllm_engine.single_engine_reader`; docs/handoffs/tasks.md):

  tap        `TapHidden`, on every base with an in-process engine and one tensor-parallel rank
             (decisio.vllm_plugin.tap): the class keeps the answer-position rows and a logits processor stores those
             of the requests that carry a key, so a head question is the plain question's one request.
  reserved   `SingleEngineHidden`, with a separate engine process or several tensor-parallel ranks: the class writes h
             into reserved logit columns, read in one request per chunk of them (docs/design/hidden-state-readout.md).

The K letter log-probabilities are the engine's own readout of the row (Gemma, `Family.head_label_logprobs="engine"`)
or computed from h and the output layer's K label rows (Qwen),

    z  = h @ W_labels^T
    lp = log_softmax(z)

so a head is fitted and served on the same (lp, h), and `intent_head.apply_intent_head(lp, h, rec, options)` is the
served arithmetic exactly. `HFHiddenEngine` and `HFReservedHiddenEngine` are the CPU stand-ins (the hf backend).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from decisio.serve.engine_health import guarded
from decisio.serve.vllm_engine import ContextRegistrations, LettersEngine


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


class HiddenReadout:
    """Mixin: `readout(state, questions)` -> [(lp, h)] per question, from `hidden_rows(token id lists)` and
    `label_rows(label ids)`, both engine-specific."""

    def label_logits(self, h, lab):
        """The label logits from h, in float64 (deterministic, whatever the kernels), with the base's final soft cap
        when it has one (decisio.families)."""
        z = h.astype(np.float64) @ self.label_rows(lab).T
        family = getattr(self, "family", None)
        cap = family.softcap if family is not None else None
        return z if cap is None else np.tanh(z / cap) * cap

    def label_lp(self, h, lab):
        """The label log-probabilities from h (float32) for one row's label ids."""
        from decisio.readout.letters import allowed_ids, is_grouped, label_log_softmax

        if is_grouped(lab):  # several forms per label: the log-softmax over all of them, summed per label
            return label_log_softmax(self.label_logits(h, allowed_ids(lab)), lab)
        return log_softmax(self.label_logits(h, lab))

    def readout(self, state, questions):
        with self._lock:
            t0 = time.perf_counter()
            rows, _ = self._prepare_separate(state, questions)
            H = self.hidden_rows([ids for ids, _ in rows])
            out = []
            for (_, lab), h in zip(rows, H):
                h = np.asarray(h, dtype=np.float32)
                out.append((self.label_lp(h, lab), h))
            self.last_ms = (time.perf_counter() - t0) * 1000
        return out


def load_output_rows(model, names=("lm_head.weight",), revision=None):
    """The output layer's weight (vocabulary x hidden), from the checkpoint's own shards: the first of `names` the
    checkpoint has (a base with tied embeddings names its input embedding after lm_head.weight,
    decisio.families.Family.head_rows); `revision` pins a hub checkpoint. `model` is a local
    directory or a Hugging Face repo id, as vLLM accepts; for a repo id only the index and the shard holding
    `lm_head.weight` are fetched (from the local cache when vLLM has already downloaded them). A file missing
    from the repo is treated as absent (huggingface_hub's EntryNotFoundError); any other hub error (no network,
    an unknown or gated repo) propagates as raised. A checkpoint without the files, or without a separate
    `lm_head.weight` (tied embeddings), is refused with an error naming the model and the file."""
    from safetensors import safe_open

    def fetch(name):
        local = Path(model) / name
        if Path(model).is_dir():
            return local if local.exists() else None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(hf_hub_download(model, name, revision=revision))
        except EntryNotFoundError:
            return None

    wanted = " or ".join(names)
    index = fetch("model.safetensors.index.json")
    if index is not None:
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        name = next((n for n in names if n in weight_map), None)
        if name is None:
            raise ValueError(
                f"{model}: model.safetensors.index.json has no {wanted} (tied embeddings?); the intent "
                "head needs the output layer's own weight"
            )
        shard = weight_map[name]
    else:
        shard = "model.safetensors"
    path = fetch(shard)
    if path is None:
        raise FileNotFoundError(
            f"{model}: neither model.safetensors.index.json nor model.safetensors found"
            if index is None
            else f"{model}: {shard}, the shard the index names for {wanted}, is missing"
        )
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        name = next((n for n in names if n in keys), None)
        if name is None:
            raise ValueError(f"{model}: {shard} has no {wanted} (tied embeddings?)")
        return f.get_tensor(name)


class HFHiddenEngine(HiddenReadout):
    """The CPU stand-in: h from the HF model's final-norm hidden state at the last position, the label rows from its
    `lm_head` (float32 model). Not for measurement."""

    device = "cpu"  # the stand-in's; an engine built with `device` sets its own

    def __init__(
        self,
        model,
        pad_to="block",
        pad_where="front",
        block_size=64,
        revision=None,
        device="cpu",
        dtype="float32",
    ):
        import threading

        import torch
        from transformers import AutoTokenizer

        from decisio.serve.hf_letters import load_causal_lm

        self.mode, self.pad_token, self.pad_where = "hidden", 198, pad_where
        self.adapters = {}
        self.device = str(torch.device(device))
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = load_causal_lm(model, revision, dtype)
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model = self.model.eval()
        self.block_size = self.match_unit = block_size
        self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
        self._lock = threading.Lock()

    # the served row builder, unchanged, with the text engine's defaults for what it reads (the server sets fmt)
    pad_policy = LettersEngine.pad_policy
    fmt = LettersEngine.fmt
    _prepare_separate = LettersEngine._prepare_separate
    _build_rows = LettersEngine._build_rows
    read_plan = LettersEngine.read_plan
    _template_tail = LettersEngine._template_tail
    _labels = LettersEngine._labels

    def label_rows(self, lab):
        return self.model.lm_head.weight[list(lab)].detach().double().cpu().numpy()

    def hidden_rows(self, token_lists):
        import torch

        with torch.no_grad():
            return [
                guarded(self, self.model.model, input_ids=torch.tensor([ids], device=self.device))
                .last_hidden_state[0, -1]
                .float()
                .cpu()
                .numpy()
                for ids in token_lists
            ]

    def probe(self):
        """The backbone over a one-token prompt (engine_health)."""
        import torch

        with torch.no_grad():
            self.model.model(
                input_ids=torch.tensor([self.tok.encode("ok", add_special_tokens=False)[:1]], device=self.device)
            )

    def facts(self):
        return {"engine": "hf hidden (CPU stand-in, not for measurement)", "pad_unit": self.pad_unit}


class SingleEngineHidden(HiddenReadout):
    """h from the serving engine itself (docs/design/hidden-state-readout.md): the text engine runs decisio's
    hidden-readout model class, which writes [0, h] into d + 1 reserved logit columns; a head question is sent with
    exactly those ids allowed, and h is recovered from their log-probabilities (`recover_hidden_chunks`). No second copy
    of the weights.

    The label log-probabilities come one of two ways, by base (decisio.families.Family.head_label_logprobs):
    recompute (Qwen) recomputes them from h and the output layer's label rows as the engine computes them
    (`label_logits`, within 3.1e-8 of the engine on a card); engine (Gemma) reads them from the engine's own readout of
    the same prompt, one more request before the hidden-state chunks, because a recomputation cannot follow the soft
    cap's bf16 rounding on the card exactly (one bf16 step near the cap moved a probability by up to 0.062). Either
    way, fitting a head and serving it read the same function, and under `engine` the head's base log-probabilities
    are the plain readout's, exactly."""

    def __init__(self, engine, model, start=None, revision=None):
        from decisio.readout.letters import DEFAULT_FORMAT, MAX_LABELS, allowed_ids, label_groups, letter_labels
        from decisio.vllm_plugin.hidden import check_reserved, reserved_chunks

        self.engine, self.llm, self.tok, self._lock = engine, engine.llm, engine.tok, engine._lock
        self.mode = "hidden (serving engine, reserved logit columns)"
        family = getattr(engine, "family", None)
        self.family = family
        self.softcap = family.softcap if family is not None else None
        self.label_source = family.head_label_logprobs if family is not None else "recompute"
        rows = family.head_rows if family is not None else ("lm_head.weight",)
        self._W = load_output_rows(model, rows, revision)
        # every token any question of this server's prompt format may read (all forms of every label)
        fmt = getattr(engine, "fmt", DEFAULT_FORMAT)
        form = fmt.label_form()
        codes = letter_labels(self.tok, MAX_LABELS) if form == "spaced" else letter_labels(self.tok, MAX_LABELS, form)
        cands = [" yes", " no"] + [" " + c for c in codes]
        labels = allowed_ids(label_groups(self.tok, cands, fmt))
        self.reserved = check_reserved(self._W.shape[1], self._W.shape[0], labels, start)
        self.chunks = reserved_chunks(self._W.shape[1], self.reserved[0])

    def _prepare_separate(self, state, questions):
        return self.engine._prepare_separate(state, questions)

    def label_rows(self, lab):
        return self._W[list(lab)].double().numpy()

    def label_logits(self, h, lab):
        """The label logits the engine itself computes from h: the product rounded to bfloat16, the dtype of the served
        model's logits (on a card, 2026-09-30: this reproduces the engine's label probabilities within 3.1e-8 on 250
        intent items; in float64 they differ by up to 0.019; runs/2026-09-30_plugin-verification)."""
        import torch

        z = torch.from_numpy(h.astype(np.float64) @ self.label_rows(lab).T).float().to(torch.bfloat16)
        return softcap_bf16(z, self.softcap).double().numpy()

    def readout(self, state, questions):
        """[(label log-probabilities, h)] per question. Under `engine` the log-probabilities are the engine's own
        readout of the row (the plain path's score_prompts, so they equal what the question gets without a head),
        read just before the row's hidden-state chunks under the same lock; under `recompute` they are computed from h.

        The rows are the letters path's (`_prepare_separate`), and a state read with its task's context registers the
        context's boundary in the serving engine's prefix cache as the letters path does (vllm_engine
        .ContextRegistrations): once per context, ahead of the reads under --register-boundary before, after the
        response under after, checked for eviction on the request's first read. Registrations of other requests that
        are due go first, as before any engine call of the letters path. What the read said of the context is
        `read_info()`."""
        with self._lock:
            t0 = time.perf_counter()
            registrar = getattr(self.engine, "__dict__", {}).get("_registrar")
            if registrar is not None:
                registrar.run_due()
            prepared = self._prepare_separate(state, questions)
            rows = prepared[0]
            ctx = ContextRegistrations(self.engine, None)
            ctx.add(prepared, 0)
            ctx_ms = ctx.send()
            out, first = [], None
            for row in rows:
                if self.label_source == "engine":
                    (p,), info = self.engine.score_prompts([row])
                    with np.errstate(divide="ignore"):
                        lp = np.log(np.asarray(p, dtype=np.float64))
                    if first is None:
                        first = (info.get("cached_tokens") or [None])[0]
                    (h,) = self.hidden_rows([row[0]])
                    h = np.asarray(h, dtype=np.float32)
                else:
                    self._cached = None
                    (h,) = self.hidden_rows([row[0]])
                    h = np.asarray(h, dtype=np.float32)
                    lp = self.label_lp(h, row[1])
                    if first is None:
                        first = (self._cached or [None])[0]
                out.append((lp, h))
            ctx.settle(lambda k: first)
            facts = ctx.facts()
            self._read.info = {"context_ms": ctx_ms, "context_boundary": facts} if facts is not None else None
            self.last_ms = (time.perf_counter() - t0) * 1000
        return out

    @property
    def _read(self):
        if "_read_local" not in self.__dict__:
            import threading

            self._read_local = threading.local()
        return self._read_local

    def read_info(self) -> dict | None:
        """The calling thread's last `readout`: `context_ms`, the context's warm-up sent in it, and `context_boundary`,
        its facts, for a state read with a task's context (x-decisio-stages); None for one without."""
        return getattr(self._read, "info", None)

    def hidden_rows(self, token_lists):
        """One engine request per chunk of reserved ids (vLLM allows at most 1,024 ids per request), sent one at a time:
        identical prompts in one batch are not the same forward on the FP8 stack (their label probabilities differed by
        up to 0.59 on a card, 2026-09-30), so a state stitched from one batch's rows is no forward's state. Sent one at
        a time, the chunks of a prompt are the same forward (they agreed to 1e-7) and share its prefix through the
        cache."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        from decisio.vllm_plugin.hidden import recover_hidden_chunks

        rows, cached = [], []
        for ids in token_lists:
            lps = []
            for j, chunk in enumerate(self.chunks):
                sp = SamplingParams(
                    max_tokens=1, temperature=0.0, logprobs=len(chunk), allowed_token_ids=chunk, detokenize=False
                )
                (o,) = guarded(
                    self.engine, self.llm.generate, [TokensPrompt(prompt_token_ids=ids)], [sp], use_tqdm=False
                )
                if j == 0:  # what the row's first read took from the prefix cache (the context's eviction check)
                    cached.append(getattr(o, "num_cached_tokens", None) or 0)
                d = o.outputs[0].logprobs[0]
                lps.append([d[t].logprob for t in chunk])  # KeyError = a reserved column went missing
            rows.append(recover_hidden_chunks(lps).astype(np.float32))
        self._cached = cached
        return rows

    def probe(self):
        """The serving engine's probe: this reader is that engine (engine_health)."""
        self.engine.probe()

    def facts(self):
        return {
            "mode": self.mode,
            "reserved_ids": [self.reserved[0], self.reserved[-1]],
            "lm_head": list(self._W.shape),
            "label_logprobs": self.label_source,
            "requests_per_question": len(self.chunks) + (self.label_source == "engine"),
        }


class TapHidden(HiddenReadout):
    """h, and the tapped middle layers, from the serving engine's own readout request (decisio.vllm_plugin.tap; the
    head reader of every base when the engine runs in the server's process): the engine runs the base's hidden-readout
    class, which carries the tap, with the `HiddenTap` logits processor in this process, and a head question
    is the plain question's one request, carrying a key; the label log-probabilities are that request's own, so they
    equal the plain readout's exactly. `last_layers()`: per question
    of the calling thread's last read, {layer: state} (float32), for the multi-layer heads."""

    # the state-span means kept, one per (state's tokens, span, layers): a question about a state read before costs
    # no extra request
    STATE_MEANS_KEPT = 4096

    def __init__(self, engine, layers=None, state_mean_layers=()):
        from collections import OrderedDict

        from decisio.vllm_plugin import tap

        self.engine, self.llm, self.tok, self._lock = engine, engine.llm, engine.tok, engine._lock
        self.family = getattr(engine, "family", None)
        self.layers = tap.tap_layers() if layers is None else tuple(layers)
        self.state_mean_layers = tuple(sorted({int(x) for x in state_mean_layers}))
        missing = sorted(set(self.state_mean_layers) - set(self.layers))
        if missing:
            raise ValueError(f"state-span means of layers {missing}: the tap reads only {list(self.layers)}")
        self._state_means = OrderedDict()
        self.mode = "hidden (serving engine, tap in the same request)"
        self.reader = "tap"  # stored with every head fitted on this reader's reads (SystemOne.head_stale)

    def _prepare_separate(self, state, questions):
        return self.engine._prepare_separate(state, questions)

    @property
    def _read(self):
        if "_read_local" not in self.__dict__:
            import threading

            self._read_local = threading.local()
        return self._read_local

    def last_layers(self):
        return getattr(self._read, "layers", None)

    def last_state_means(self):
        """Per question of the calling thread's last read, {layer: the mean state over the request's state tokens}
        (float32), or None for a reader without state-span layers; a question whose state span was not found has
        {"error": why}."""
        return getattr(self._read, "state_means", None)

    def state_means(self, ids, labels, shared, state):
        """{layer: mean of that layer's state over the state's tokens} for one prompt row (`ids`, its `labels`, the
        row's shared prefix length `shared`): the span is the state's text inside the shared prefix
        (decisio.readout.state_span), read in an engine call of one request that computes its whole prompt up to the
        span's end (no prefix cache), so every state token is in the forward. Kept per state; the engine's lock must be
        held."""
        import hashlib

        from decisio.readout.state_span import state_span, state_text_of
        from decisio.vllm_plugin import tap

        start, end = state_span(self.tok, ids, state_text_of(state), within=shared, occurrence="last")
        key = (hashlib.sha256(np.asarray(ids[:end], dtype="<i8").tobytes()).hexdigest(), start, self.state_mean_layers)
        got = self._state_means.get(key)
        if got is not None:
            self._state_means.move_to_end(key)
            return got
        tap.span_begin(start, end, {layer: self.layers.index(layer) for layer in self.state_mean_layers})
        try:
            self.engine.score_prompts([(list(ids[:end]), labels)], skip_cache=True)
        finally:
            means = tap.span_end()
        self._state_means[key] = means
        while len(self._state_means) > self.STATE_MEANS_KEPT:
            self._state_means.popitem(last=False)
        return means

    def read_info(self) -> dict | None:
        """The calling thread's last `readout`: `context_ms` and `context_boundary` for a state read with a task's
        context (x-decisio-stages), as `SingleEngineHidden.read_info`; None for one without."""
        return getattr(self._read, "info", None)

    def readout(self, state, questions):
        """[(label log-probabilities, h)] per question, from one engine request per question row: the plain readout's
        own request, carrying the tap's key. A state read with its task's context registers the context's boundary in
        the prefix cache as the letters path and `SingleEngineHidden` do (vllm_engine.ContextRegistrations)."""
        import uuid

        from decisio.vllm_plugin import tap

        with self._lock:
            t0 = time.perf_counter()
            registrar = getattr(self.engine, "__dict__", {}).get("_registrar")
            if registrar is not None:
                registrar.run_due()
            prepared = self._prepare_separate(state, questions)
            rows = prepared[0]
            ctx = ContextRegistrations(self.engine, None)
            ctx.add(prepared, 0)
            ctx_ms = ctx.send()
            keys = [f"tap-{uuid.uuid4().hex}" for _ in rows]
            probs, info = self.engine.score_prompts(rows, extra_args=[{tap.TAP_KEY: k} for k in keys])
            first = (info.get("cached_tokens") or [None])[0] if isinstance(info, dict) else None
            ctx.settle(lambda k: first)
            facts = ctx.facts()
            self._read.info = {"context_ms": ctx_ms, "context_boundary": facts} if facts is not None else None
            out, layers = [], []
            for p, k in zip(probs, keys):
                row = tap.pop(k)
                if row is None:
                    raise RuntimeError(
                        "the engine stored no hidden state for this question: the tap needs the engine built with "
                        "DecisioGemma4UnifiedHiddenReadout and decisio.vllm_plugin.tap:HiddenTap, in this process"
                    )
                d = int(row.shape[-1]) // (1 + len(self.layers))
                h, mids = tap.split(row, d, self.layers)
                with np.errstate(divide="ignore"):
                    out.append((np.log(np.asarray(p, dtype=np.float64)), h))
                layers.append(mids)
            self._read.layers = layers
            self._read.state_means = None
            if self.state_mean_layers:  # the questions share the state: one span read (or none, when kept)
                from decisio.readout.state_span import StateSpanError

                try:
                    means = self.state_means(rows[0][0], rows[0][1], prepared[1], state)
                except StateSpanError as e:
                    means = {"error": str(e)}
                self._read.state_means = [means] * len(rows)
            self.last_ms = (time.perf_counter() - t0) * 1000
        return out

    def probe(self):
        self.engine.probe()

    def facts(self):
        return {
            "mode": self.mode,
            "layers": list(self.layers),
            "state_mean_layers": list(self.state_mean_layers),
            "label_logprobs": "engine",
            "requests_per_question": 1,
            "reader": "tap",
        }


def softcap_bf16(z, cap):
    """vLLM's LogitsProcessor soft cap on bf16 logits, step by step in bf16 (z / cap, tanh, * cap); None: unchanged."""
    if cap is None:
        return z
    import torch

    z = z.to(torch.bfloat16)
    return torch.tanh(z / cap) * cap


def reserved_logprobs(h, start=0):
    """What vLLM returns for each chunk of reserved ids of a hidden-readout question: the float32 log-softmax of the
    chunk's columns of [0, h] (the masked logits of its allowed set). For the CPU stand-in and the tests."""
    from decisio.vllm_plugin.hidden import reserved_chunks

    full = np.concatenate([np.zeros(1, dtype=np.float32), np.asarray(h, dtype=np.float32)])
    out = []
    for chunk in reserved_chunks(len(full) - 1, start):
        z = full[[i - start for i in chunk]]
        m = z.max()
        out.append((z - (m + np.log(np.exp(z - m, dtype=np.float32).sum(dtype=np.float32)))).astype(np.float32))
    return out


class HFReservedHiddenEngine(HFHiddenEngine):
    """The CPU stand-in of the single-engine route: the HF model's h sent through the same reserved-column
    log-probabilities and recovery as on vLLM. Not for measurement."""

    def hidden_rows(self, token_lists):
        from decisio.vllm_plugin.hidden import recover_hidden_chunks

        return [
            recover_hidden_chunks(reserved_logprobs(h)).astype(np.float32) for h in super().hidden_rows(token_lists)
        ]

    def facts(self):
        return {**super().facts(), "route": "reserved logit columns (emulated)"}
