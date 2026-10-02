# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Letters-readout serving on MLX (Apple silicon): the served default's text route on a Mac.

`MLXLettersEngine` subclasses `vllm_engine.LettersEngine`, as the CPU stand-in (`hf_letters`) does, and replaces only
what needs vLLM: model loading and the forward pass. Everything a request goes through before that is the served code:
state tokenization, the split into state and question, the front padding to the 1,056-token block, the label tokens,
derived questions, `/v1/answer` and `/v1/systemone`, the temperature, tasks, the intent head and abstention.

The prompts are byte for byte the vLLM path's:

- the tokenizer is the official one (`--tokenizer`, default `Qwen/Qwen3.6-35B-A3B-FP8`), not a conversion's: the
  mlx-community conversions of Qwen3.6-35B-A3B ship a different `tokenizer.json` and `tokenizer_config.json`;
- the padding is the served default's (front, to vLLM's 1,056-token block), which MLX's cache does not need but the
  served prompts carry; `--pad-to none` leaves it out.

Separate mode only: one forward per question. For each request the state prefix (the tokens every question shares,
padding included) is prefilled once into a fresh prompt cache and evaluated; each question then continues from a copy
of that cache, one question at a time. The Gated DeltaNet layers' cache entries are replaced, never written into, so a
copy shares them; the attention layers' key and value buffers are written in place, so a copy slices them to the prefix
and its first write allocates new ones. Every question, single-question requests included, goes through the prefix:
a question's answer does not depend on what else is in its request, bit for bit, and repeats bit for bit.

The letters are read at the last position: the softmax over the label logits (bf16 out of the output layer, then
float64). The final-norm hidden state at the same position comes out of the same forward, so the intent head
(`MLXHiddenReadout`) fits and serves on exactly the served readout, with no second weight copy and no extra request.

    from decisio.serve.mlx_engine import MLXLettersEngine
    eng = MLXLettersEngine("mlx-community/Qwen3.6-35B-A3B-4bit")
    out, info = eng.answer(state, [{"kind": "noul", "instructions": "Is this urgent?"}])

    python -m decisio.serve.vllm_engine --backend mlx --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8000
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import numpy as np

from decisio.readout.letters import allowed_ids, is_grouped, label_log_softmax
from decisio.serve.vllm_engine import PAD_PLACES, PAD_TOKEN, LettersEngine

OFFICIAL_TOKENIZER = "Qwen/Qwen3.6-35B-A3B-FP8"
# sha256 of the official tokenizer files (Qwen/Qwen3.6-35B-A3B-FP8 and Qwen/Qwen3.6-35B-A3B, identical)
OFFICIAL_TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
SERVED_BLOCK = 1056  # vLLM's block on the served default: --pad-to block pads to it, so the prompts are the same
PREFILL_STEP = 2048  # tokens per prefill chunk (bounds the attention layers' memory at long states)


def copy_cache(cache):
    """A prompt cache that continues from `cache` without changing it (see the module docstring)."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            n = ArraysCache(len(c.cache))
            n.cache = list(c.cache)
        elif isinstance(c, KVCache):
            n = KVCache()
            if c.keys is not None:
                n.keys, n.values, n.offset = c.keys[..., : c.offset, :], c.values[..., : c.offset, :], c.offset
        else:
            raise TypeError(f"no copy rule for prompt cache entries of type {type(c).__name__}")
        out.append(n)
    return out


def _cache_arrays(cache):
    from mlx_lm.models.cache import ArraysCache

    arrays = []
    for c in cache:
        parts = c.cache if isinstance(c, ArraysCache) else (c.keys, c.values)
        arrays += [a for a in parts if a is not None]
    return arrays


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer_file(tokenizer, name="tokenizer.json"):
    """The local path of a tokenizer's file: in its directory, or from the Hugging Face cache (fetched if missing)."""
    if Path(tokenizer).is_dir():
        p = Path(tokenizer) / name
        return p if p.exists() else None
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(tokenizer, name))
    except Exception:  # noqa: BLE001  (a repo without the file, or offline: reported as unknown in facts)
        return None


class MLXLettersEngine(LettersEngine):
    def __init__(
        self,
        model,
        tokenizer=OFFICIAL_TOKENIZER,
        pad_to="block",
        pad_token=PAD_TOKEN,
        pad_where="front",
        block_size=SERVED_BLOCK,
        prefill_step=PREFILL_STEP,
        warm_up=True,
    ):
        import mlx.core as mx
        from mlx_lm import load
        from transformers import AutoTokenizer

        if pad_where not in PAD_PLACES:
            raise ValueError(f"pad_where must be one of {PAD_PLACES}")
        self.mode, self.pad_token, self.pad_where = "separate", pad_token, pad_where
        self.adapters = {}
        self.model_name, self.tokenizer_name = str(model), str(tokenizer)
        self.model, _ = load(str(model))  # its own tokenizer is not used (see the module docstring)
        lm = getattr(self.model, "language_model", self.model)
        self._backbone = lm.model  # returns the final-norm hidden states
        self._head = lm.lm_head if hasattr(lm, "lm_head") else lm.model.embed_tokens.as_linear
        self.tok = AutoTokenizer.from_pretrained(str(tokenizer))
        self.block_size = self.match_unit = int(block_size)
        self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
        self.prefill_step = int(prefill_step)
        self._lock = threading.Lock()
        self._mx = mx
        if warm_up:
            self._warm_up()

    # ---- the forward pass ------------------------------------------------------------------------

    def _feed(self, ids, cache):
        """Feed `ids` into `cache` in chunks; the final-norm hidden state at the last position."""
        mx = self._mx
        h = None
        for a in range(0, len(ids), self.prefill_step):
            out = self._backbone(mx.array([ids[a : a + self.prefill_step]]), cache=cache)
            h = out[0, -1]
            mx.eval(h, *_cache_arrays(cache))
        return h

    def _score(self, rows, prefix):
        """[(label log-probabilities float64, hidden state float32)] per row, every row continuing from its first
        `prefix` tokens, which all rows share and which are prefilled once."""
        mx = self._mx
        if any(ids[:prefix] != rows[0][0][:prefix] for ids, _ in rows):
            raise ValueError("the rows do not share their first `prefix` tokens")
        if any(len(ids) <= prefix for ids, _ in rows):
            raise ValueError("every row needs at least one token after the shared prefix")
        base = self.model.make_cache()
        t0 = time.perf_counter()
        if prefix:
            self._feed(rows[0][0][:prefix], base)
        prefix_ms = (time.perf_counter() - t0) * 1000
        out = []
        for ids, lab in rows:
            h = self._feed(ids[prefix:], copy_cache(base))
            z = self._head(h[None])[0][mx.array(allowed_ids(lab))]
            z = np.array(z.astype(mx.float32), dtype=np.float64)
            if is_grouped(lab):  # several forms per label: their probabilities summed
                lp = label_log_softmax(z, lab)
            else:
                lp = z - z.max()
                lp -= np.log(np.exp(lp).sum())
            out.append((lp, np.array(h.astype(mx.float32))))
        return out, prefix_ms

    # ---- LettersEngine's engine interface --------------------------------------------------------

    def score_prompts(self, rows, adapter=None, warm=(), skip_cache=False, mm=None, mm_uuids=None, prefix=None):
        """Label distributions for fully-built prompts [(token ids, label ids)], as `LettersEngine.score_prompts`.

        `prefix`: the shared prefix length (the request's padded state prefix, as `_prepare_separate` returns it);
        None takes the rows' longest common prefix (none for one row). `warm` and `skip_cache` are accepted for the
        interface and have no effect: the prefix is always prefilled once, per call."""
        if mm:
            raise ValueError("the MLX engine serves the text route only (no image route in this phase)")
        if adapter:
            raise ValueError("the MLX engine serves no adapters")
        t0 = time.perf_counter()
        if prefix is None:
            prefix = 0
            if len(rows) > 1:
                first = rows[0][0]
                while all(len(ids) > prefix + 1 and ids[prefix] == first[prefix] for ids, _ in rows):
                    prefix += 1
        scored, prefix_ms = self._score(rows, prefix)
        probs = [np.exp(lp) / np.exp(lp).sum() for lp, _ in scored]
        return probs, {
            "warm_ms": 0.0,
            "prefix_ms": prefix_ms,
            "cached_tokens_mean": float(prefix),
            "engine_timing": [],
            "prompt_tokens_mean": float(np.mean([len(r[0]) for r in rows])) if rows else 0.0,
            "prompt_tokens": prefix + sum(len(r[0]) - prefix for r in rows),
            "engine_prompt_tokens": [len(r[0]) for r in rows],
            "cached_tokens": [prefix] * len(rows),
            "forward_ms": (time.perf_counter() - t0) * 1000,
        }

    def _answer_separate(self, requests, adapter):
        if adapter:
            raise ValueError("the MLX engine serves no adapters")
        results, infos = [], []
        for state, questions in requests:
            rows, P = self._prepare_separate(state, questions)
            probs, info = self.score_prompts(rows, prefix=P)
            results.append(probs)
            infos.append((P, info))
        info = {
            "warm_ms": 0.0,
            "prefix_ms": sum(i["prefix_ms"] for _, i in infos),
            "cached_tokens_mean": float(np.mean([c for _, i in infos for c in i["cached_tokens"]])),
            "prompt_tokens": sum(i["prompt_tokens"] for _, i in infos),
            "engine_prompt_tokens": [n for _, i in infos for n in i["engine_prompt_tokens"]],
            "cached_tokens": [c for _, i in infos for c in i["cached_tokens"]],
            "shared_prefix_tokens": infos[0][0] if len(infos) == 1 else [p for p, _ in infos],
            "questions": sum(len(r) for r in results),
        }
        return results, info

    def _answer_packed(self, requests):
        raise ValueError("the MLX engine serves separate mode only")

    def facts(self):
        import json
        import platform
        from importlib.metadata import version

        mx = self._mx
        config = Path(self.model_name) / "config.json"
        quant = None
        if config.exists():
            q = json.loads(config.read_text()).get("quantization") or {}
            quant = {k: v for k, v in q.items() if not isinstance(v, dict)} or None
        tj = tokenizer_file(self.tokenizer_name)
        tok_sha = _sha256(tj) if tj else None
        return {
            "engine": "mlx",
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "device": str(mx.default_device()),
            "machine": platform.machine(),
            "model": self.model_name,
            "quantization": quant,
            "tokenizer": self.tokenizer_name,
            "tokenizer_json_sha256": tok_sha,
            "official_tokenizer": tok_sha == OFFICIAL_TOKENIZER_SHA256,
            "mode": self.mode,
            "block_size": self.block_size,
            "pad_unit": self.pad_unit,
            "pad_where": self.pad_where if self.pad_unit else None,
            "prefill_step": self.prefill_step,
            "adapters": [],
        }


class MLXHiddenReadout:
    """The intent head's hidden state on the MLX engine: the final-norm hidden state at the answer position, from the
    same forward that gives the letters (`MLXLettersEngine._score`), so a head is fitted and served on exactly the
    served readout. No second weight copy, no extra request; the engine's lock serialises it with the text route."""

    def __init__(self, engine):
        self.engine, self.tok, self._lock = engine, engine.tok, engine._lock
        self.mode = "hidden (MLX engine, same forward as the letters)"

    def _prepare_separate(self, state, questions):
        return self.engine._prepare_separate(state, questions)

    def readout(self, state, questions):
        with self._lock:
            t0 = time.perf_counter()
            rows, P = self._prepare_separate(state, questions)
            scored, _ = self.engine._score(rows, P)
            self.last_ms = (time.perf_counter() - t0) * 1000
        return scored

    def facts(self):
        return {"mode": self.mode, "requests_per_question": 0}
