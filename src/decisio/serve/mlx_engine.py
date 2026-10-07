# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Letters-readout serving on MLX (Apple silicon): the served default's text route on a Mac.

`MLXLettersEngine` subclasses `vllm_engine.LettersEngine`, as the CPU stand-in (`hf_letters`) does, and replaces only
what needs vLLM: model loading and the forward pass. Everything a request goes through before that is the served code:
state tokenization, the split into state and question, the front padding to the 1,056-token block, the label tokens,
derived questions, `/v1/answer` and `/v1/systemone`, the temperature, tasks, the intent head and abstention.

The prompts are byte for byte the vLLM path's:

- the tokenizer is the base's official one (its checkpoint at its revision, `decisio.families`, found from the
  conversion's `model_type`; Qwen: `Qwen/Qwen3.6-35B-A3B-FP8`, Gemma 4: `google/gemma-4-12B-it` at `707f0a3b`), not a
  conversion's: the mlx-community
  conversions of Qwen3.6-35B-A3B ship a different `tokenizer.json` and `tokenizer_config.json`, and those of
  gemma-4-12B-it the chat template from before Google's 2026-07-15 fix; `--tokenizer` overrides it (`repo@revision`
  pins a revision);
- the padding is the served default's (front, to vLLM's 1,056-token block), which MLX's cache does not need but the
  served prompts carry; `--pad-to none` leaves it out.

Separate mode only: one forward per question. For each request the state prefix (the tokens every question shares,
padding included) is prefilled once into a fresh prompt cache and evaluated; each question then continues from a copy
of that cache, one question at a time. The Gated DeltaNet layers' cache entries are replaced, never written into, so a
copy shares them; the attention layers' key and value buffers are written in place, so a copy slices them to the prefix
and its first write allocates new ones; a sliding-window layer's ring buffer (Gemma 4) is copied whole, with its write
position, into new array objects, so writes into the copy leave the original as it was. Every question, single-question
requests included, goes through the prefix:
a question's answer does not depend on what else is in its request, bit for bit, and repeats bit for bit.

Across requests, the evaluated cache of each state prefix is kept (`PrefixCache`, least recently used first out,
bounded by `prefix_cache_mb` in MiB, the base's (decisio.families, mlx_prefix_cache_mb); 0 turns it off). A request
whose prefix tokens equal a kept entry's continues from that entry instead of prefilling it again. Exact by
construction: an entry is the cache the engine computed for exactly those tokens (compared in full, not by hash
alone), and it is never written into, since every question continues from a copy; the forward is deterministic, so a
hit gives the arrays a miss would compute.

The letters are read at the last position: the softmax over the label logits (bf16 out of the output layer, through
the model's final-logit softcap when it has one, as mlx-lm applies it: Gemma 4's 30; then float64). The final-norm
hidden state at the same position comes out of the same forward, so the intent head
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
from collections import OrderedDict
from pathlib import Path

import numpy as np

from decisio.families import QWEN, family_of
from decisio.readout.letters import allowed_ids, is_grouped, label_log_softmax
from decisio.serve.engine_health import guarded
from decisio.serve.vllm_engine import PAD_PLACES, PAD_TOKEN, LettersEngine

# sha256 of each base's official tokenizer.json (the base's checkpoint at its revision, decisio.families). Qwen:
# Qwen/Qwen3.6-35B-A3B-FP8 and Qwen/Qwen3.6-35B-A3B have identical files. Gemma 4: unchanged since the first upload; the
# revision pins the chat template (the mlx-community conversions ship the one from before Google's 2026-07-15 fix).
TOKENIZER_SHA256 = {
    "qwen3.6-35b-a3b": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "gemma-4-12b": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
}
OFFICIAL_TOKENIZER, OFFICIAL_TOKENIZER_SHA256 = QWEN.model, TOKENIZER_SHA256[QWEN.key]
SERVED_BLOCK = 1056  # vLLM's block on the served default: --pad-to block pads to it, so the prompts are the same
PREFILL_STEP = 2048  # tokens per prefill chunk (bounds the attention layers' memory at long states)
# the cross-request prefix cache's budget in MiB when a caller gives none (the server uses the base's)
PREFIX_CACHE_MB = 2048


def split_revision(tokenizer):
    """`repo@revision` -> (repo, revision); a local directory or a bare repository -> (it, None)."""
    t = str(tokenizer)
    if "@" in t and not Path(t).exists():
        name, rev = t.rsplit("@", 1)
        return name, rev
    return t, None


def copy_cache(cache):
    """A prompt cache that continues from `cache` without changing it (see the module docstring)."""
    from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            n = ArraysCache(len(c.cache))
            n.cache = list(c.cache)
        elif isinstance(c, RotatingKVCache):
            # the ring buffer whole, with its write position: a one-token step writes into the copy's arrays in place
            # and a longer one concatenates; either way the copy's arrays are new objects, and a write into a slice of
            # an MLX array does not reach the array it was sliced from, so the original keeps its values
            n = RotatingKVCache(max_size=c.max_size, keep=c.keep)
            if c.keys is not None:
                n.keys, n.values = c.keys[...], c.values[...]
            n.offset, n._idx = c.offset, c._idx
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


def tokenizer_file(tokenizer, name="tokenizer.json", revision=None):
    """The local path of a tokenizer's file: in its directory, or from the Hugging Face cache (fetched if missing)."""
    if Path(tokenizer).is_dir():
        p = Path(tokenizer) / name
        return p if p.exists() else None
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(tokenizer, name, revision=revision))
    except Exception:  # noqa: BLE001  (a repo without the file, or offline: reported as unknown in facts)
        return None


class PrefixCache:
    """Evaluated prompt caches of state prefixes, kept across requests: least recently used first out, bounded in bytes.
    Exact by construction (see the module docstring): an entry is returned only for exactly the token ids it was
    computed for, and callers never write into it (`copy_cache`)."""

    def __init__(self, max_bytes):
        self.max_bytes, self.entries, self.bytes, self.hits, self.misses = int(max_bytes), OrderedDict(), 0, 0, 0

    @staticmethod
    def _key(ids):
        return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()

    def get(self, ids):
        e = self.entries.get(self._key(ids))
        if e is None or not np.array_equal(e[0], np.asarray(ids, dtype=np.int64)):
            self.misses += 1
            return None
        self.entries.move_to_end(self._key(ids))
        self.hits += 1
        return e[1]

    def put(self, ids, cache):
        n = sum(a.nbytes for a in _cache_arrays(cache))
        if n > self.max_bytes:
            return
        key = self._key(ids)
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[2]
        while self.entries and self.bytes + n > self.max_bytes:
            self.bytes -= self.entries.popitem(last=False)[1][2]
        self.entries[key] = (np.asarray(ids, dtype=np.int64), cache, n)
        self.bytes += n

    def facts(self):
        return {
            "max_mb": self.max_bytes / 2**20,
            "entries": len(self.entries),
            "mb": round(self.bytes / 2**20, 1),
            "hits": self.hits,
            "misses": self.misses,
        }


class MLXLettersEngine(LettersEngine):
    def __init__(
        self,
        model,
        tokenizer=None,
        family=None,
        pad_to="block",
        pad_token=PAD_TOKEN,
        pad_where="front",
        block_size=SERVED_BLOCK,
        prefill_step=PREFILL_STEP,
        prefix_cache_mb=PREFIX_CACHE_MB,
        warm_up=True,
    ):
        import mlx.core as mx
        from mlx_lm import load
        from transformers import AutoTokenizer

        if pad_where not in PAD_PLACES:
            raise ValueError(f"pad_where must be one of {PAD_PLACES}")
        self.mode, self.pad_token, self.pad_where = "separate", pad_token, pad_where
        self.adapters = {}
        self.model_name = str(model)
        # the base (decisio.families), from the conversion's config.json model_type unless given; Qwen for any other
        self.family = family or family_of(str(model))
        self.official_tokenizer_sha256 = TOKENIZER_SHA256.get(self.family.key)
        if tokenizer is None:
            self.tokenizer_name, self.tokenizer_revision = self.family.model, self.family.revision
        else:
            self.tokenizer_name, self.tokenizer_revision = split_revision(tokenizer)
        self.model, _ = load(str(model))  # its own tokenizer is not used (see the module docstring)
        lm = getattr(self.model, "language_model", self.model)
        self._backbone = lm.model  # returns the final-norm hidden states
        self._head = lm.lm_head if hasattr(lm, "lm_head") else lm.model.embed_tokens.as_linear
        # the model's final-logit softcap where it has one (Gemma 4: 30), applied as mlx-lm applies it
        self.softcap = getattr(lm, "final_logit_softcapping", None)
        self.tok = AutoTokenizer.from_pretrained(self.tokenizer_name, revision=self.tokenizer_revision)
        self.block_size = self.match_unit = int(block_size)
        self.pad_unit = None if not pad_to else (self.block_size if pad_to == "block" else int(pad_to))
        self.prefill_step = int(prefill_step)
        self.prefix_cache = PrefixCache(prefix_cache_mb * 2**20) if prefix_cache_mb else None
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
        `prefix` tokens, which all rows share and which are prefilled once (or taken from the prefix cache); also the
        prefix's time and whether it came from the cache."""
        mx = self._mx
        if any(ids[:prefix] != rows[0][0][:prefix] for ids, _ in rows):
            raise ValueError("the rows do not share their first `prefix` tokens")
        if any(len(ids) <= prefix for ids, _ in rows):
            raise ValueError("every row needs at least one token after the shared prefix")
        t0 = time.perf_counter()
        pre = rows[0][0][:prefix]
        base = self.prefix_cache.get(pre) if (prefix and self.prefix_cache is not None) else None
        hit = base is not None
        if not hit:
            base = self.model.make_cache()
            if prefix:
                self._feed(pre, base)
                if self.prefix_cache is not None:
                    self.prefix_cache.put(pre, base)
        prefix_ms = (time.perf_counter() - t0) * 1000
        out = []
        for ids, lab in rows:
            h = self._feed(ids[prefix:], copy_cache(base))
            z = self._head(h[None])[0][mx.array(allowed_ids(lab))]
            if self.softcap is not None:  # elementwise in the logits' dtype, so the labels' values of the capped vector
                z = mx.tanh(z / self.softcap) * self.softcap
            z = np.array(z.astype(mx.float32), dtype=np.float64)
            if is_grouped(lab):  # several forms per label: their probabilities summed
                lp = label_log_softmax(z, lab)
            else:
                lp = z - z.max()
                lp -= np.log(np.exp(lp).sum())
            out.append((lp, np.array(h.astype(mx.float32))))
        return out, prefix_ms, hit

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
        scored, prefix_ms, hit = guarded(self, self._score, rows, prefix)
        probs = [np.exp(lp) / np.exp(lp).sum() for lp, _ in scored]
        return probs, {
            "warm_ms": 0.0,
            "prefix_ms": prefix_ms,
            "prefix_cache_hit": hit,
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
            "prefix_cache_hits": sum(bool(i["prefix_cache_hit"]) for _, i in infos),
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

    def probe(self):
        """The backbone over a one-token prompt into a new cache (engine_health)."""
        self._feed(self.tok.encode("ok", add_special_tokens=False)[:1], self.model.make_cache())

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
        tj = tokenizer_file(self.tokenizer_name, revision=self.tokenizer_revision)
        tok_sha = _sha256(tj) if tj else None
        return {
            "engine": "mlx",
            "base": self.family.key,
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "device": str(mx.default_device()),
            "machine": platform.machine(),
            "model": self.model_name,
            "quantization": quant,
            "tokenizer": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_json_sha256": tok_sha,
            "official_tokenizer": tok_sha == self.official_tokenizer_sha256,
            "final_logit_softcap": self.softcap,
            "mode": self.mode,
            "block_size": self.block_size,
            "pad_unit": self.pad_unit,
            "pad_where": self.pad_where if self.pad_unit else None,
            "prefill_step": self.prefill_step,
            "prefix_cache": self.prefix_cache.facts() if self.prefix_cache is not None else None,
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
            scored, _, _ = guarded(self.engine, self.engine._score, rows, P)
            self.last_ms = (time.perf_counter() - t0) * 1000
        return scored

    def probe(self):
        """The serving engine's probe: this reader is that engine (engine_health)."""
        self.engine.probe()

    def facts(self):
        return {"mode": self.mode, "requests_per_question": 0}
